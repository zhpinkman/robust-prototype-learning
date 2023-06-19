from IPython import embed
from transformers import EarlyStoppingCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
import pandas as pd
import argparse
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

dataset_to_max_length = {
    "imdb": 512,
    "dbpedia": 512,
    "ag_news": 64,
}

dataset_to_num_labels = {
    "imdb": 2,
    "dbpedia": 9,
    "ag_news": 4,
}


def preprocess_data(tokenizer, dataset, args):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=dataset_to_max_length[args.dataset],
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return tokenized_dataset


def load_data(data_dir, mode):
    test_names = [
        file
        for file in os.listdir(data_dir)
        if (file.startswith("test_") and file.endswith("csv"))
    ]

    test_dfs = [pd.read_csv(os.path.join(data_dir, file)) for file in test_names]

    adv_attack_names = [
        file for file in os.listdir(data_dir) if file.startswith("adv_")
    ]
    adv_attack_dfs = [
        pd.read_csv(os.path.join(data_dir, file)) for file in adv_attack_names
    ]
    if mode == "test":
        dataset = DatasetDict(
            {
                **{
                    file[: file.find(".csv")]: Dataset.from_pandas(df)
                    for file, df in zip(test_names, test_dfs)
                },
                **{
                    file[: file.find(".csv")]: Dataset.from_pandas(df)
                    for file, df in zip(adv_attack_names, adv_attack_dfs)
                },
            }
        )
    else:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        # if args.dataset == "dbpedia":
        #     train_df = train_df.sample(frac=0.1)
        #     print("number of labels in train: ", len(train_df["label"].unique()))
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                **{
                    file[: file.find(".csv")]: Dataset.from_pandas(df)
                    for file, df in zip(test_names, test_dfs)
                },
                **{
                    file[: file.find(".csv")]: Dataset.from_pandas(df)
                    for file, df in zip(adv_attack_names, adv_attack_dfs)
                },
            }
        )
    return dataset


def compute_metrics(eval_preds):
    from datasets import load_metric

    metric = load_metric("accuracy")
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    try:
        predictions = np.argmax(predictions, axis=1)
        # use classification report from sklearn
        from sklearn.metrics import classification_report

        print(classification_report(labels, predictions, digits=3))
    except Exception as e:
        print(e)
        embed()
        exit()
    return metric.compute(predictions=predictions, references=labels)


def main(args):
    if args.mode == "train":
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=dataset_to_num_labels[args.dataset],
            ignore_mismatched_sizes=True,
        )
        if "bart" in args.model_checkpoint:
            num_dec_layers = len(model.base_model.decoder.layers)
            for i in range(num_dec_layers):
                model.base_model.decoder.layers[i].requires_grad_(False)
            model.base_model.decoder.layers[num_dec_layers - 1].requires_grad_(False)

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
            num_labels=dataset_to_num_labels[args.dataset],
            ignore_mismatched_sizes=True,
        )

    dataset = load_data(args.data_dir, args.mode)
    tokenized_dataset = preprocess_data(tokenizer, dataset, args)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_dir=args.log_dir,
        report_to=None,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        logging_strategy="steps",
        save_steps=args.logging_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        load_best_model_at_end=True,
        # metric_for_best_model="eval_accuracy",
        # greater_is_better=True,
        learning_rate=1e-5,
    )

    # os.environ["WANDB_PROJECT"] = f"bert-{args.data_dir.split('/')[-1]}"

    if args.mode == "train":
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test_paraphrased"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        # trainer.evaluate(tokenized_dataset["test"])

        trainer.train()
        trainer.save_model(args.model_dir)

    elif args.mode == "test":
        trainer = Trainer(
            model,
            training_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                # EarlyStoppingCallback(
                #     early_stopping_patience=3, early_stopping_threshold=0.01
                # )
            ],
        )
        splits_for_test_and_adv_in_dataset = [
            key
            for key in tokenized_dataset.keys()
            if (key.startswith("test_") or key.startswith("adv_"))
        ]
        for split in splits_for_test_and_adv_in_dataset:
            print("results for split: ", split)
            trainer.evaluate(tokenized_dataset[split])
            embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )
    parser.add_argument("--model_checkpoint", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=float, default=3)

    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--logging_steps", type=int, default=100)

    args = parser.parse_args()
    main(args)
