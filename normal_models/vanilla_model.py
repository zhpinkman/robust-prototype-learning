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

# import train_test split
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

dataset_to_max_length = {
    "imdb": 512,
    "dbpedia": 512,
    "ag_news": 64,
    "sst2": 64,
    "olid": 64,
}

dataset_to_num_labels = {"imdb": 2, "dbpedia": 9, "ag_news": 4, "sst2": 2, "olid": 2}


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


def load_one_data(data_dir, mode, file, split_training_data):
    df = pd.read_csv(os.path.join(data_dir, file))
    if split_training_data:
        df_texts = df["text"].tolist()
        df_labels = df["label"].tolist()
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df_texts, df_labels, test_size=0.1, random_state=42
        )
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_texts, train_labels, test_size=0.1, random_state=42
        )
        train_df = pd.DataFrame(
            {"text": train_texts, "label": train_labels, "split": "train"}
        )
        val_df = pd.DataFrame({"text": val_texts, "label": val_labels, "split": "val"})
        test_df = pd.DataFrame(
            {"text": test_texts, "label": test_labels, "split": "test"}
        )
        return DatasetDict(
            {
                split: Dataset.from_pandas(df)
                for split, df in zip(
                    ["train", "val", "test"], [train_df, val_df, test_df]
                )
            }
        )
    else:
        return DatasetDict({file[: file.find(".csv")]: Dataset.from_pandas(df)})


def load_data(data_dir, mode):
    test_names = [
        file
        for file in os.listdir(data_dir)
        if (
            (file.startswith("test") and file.endswith("csv"))
            or (file.startswith("val") and file.endswith("csv"))
        )
    ]

    test_dfs = [pd.read_csv(os.path.join(data_dir, file)) for file in test_names]

    adv_attack_names = [file for file in os.listdir(data_dir) if file.startswith("adv")]
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
        from sklearn.model_selection import train_test_split

        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

        indices_to_pick = []
        for label in train_df["label"].unique():
            sub_df = train_df[train_df["label"] == label]
            sub_df_sample = sub_df.sample(
                n=min(10000, sub_df.shape[0]), random_state=42, replace=False
            )
            indices_to_pick.extend(sub_df_sample.index.tolist())
        train_df = train_df.loc[indices_to_pick]
        train_df = train_df.sample(frac=1, replace=False).reset_index(drop=True)

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
            print("Freezing all layers except the last layer for BART")
            num_enc_layers = len(model.base_model.encoder.layers)

            for i in range(num_enc_layers):
                model.base_model.encoder.layers[i].requires_grad_(False)
            model.base_model.encoder.layers[num_enc_layers - 1].requires_grad_(True)
        elif "bert" in args.model_checkpoint:
            print("Freezing all layers except the last layer for BERT")
            num_enc_layers = len(model.base_model.encoder.layer)

            for i in range(num_enc_layers):
                model.base_model.encoder.layer[i].requires_grad_(False)
            model.base_model.encoder.layer[num_enc_layers - 1].requires_grad_(True)
        elif "electra" in args.model_checkpoint:
            print("Freezing all layers except the last layer for ELECTRA")
            num_enc_layers = len(model.base_model.encoder.layer)

            for i in range(num_enc_layers):
                model.base_model.encoder.layer[i].requires_grad_(False)
            model.base_model.encoder.layer[num_enc_layers - 1].requires_grad_(True)

        embed()
        exit()

    else:
        print(
            "Loading model from: ",
            args.model_dir,
            "with num labels: ",
            dataset_to_num_labels[args.dataset],
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
        )

    if args.test_file is not None:
        dataset = load_one_data(
            args.data_dir, args.mode, args.test_file, args.split_training_data
        )
    else:
        dataset = load_data(args.data_dir, args.mode)

    tokenized_dataset = preprocess_data(tokenizer, dataset, args)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        eval_accumulation_steps=20,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=256,
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
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        learning_rate=args.learning_rate,
    )
    all_results_for_test_and_adv = {}

    if args.mode == "train":
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["val"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3, early_stopping_threshold=0.01
                )
            ],
        )
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        # trainer.evaluate(tokenized_dataset["test"])

        trainer.train()
        # trainer.evaluate(tokenized_dataset["test"])
        trainer.save_model(args.model_dir)

    elif args.mode == "test":

        trainer = Trainer(
            model,
            training_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        splits_for_test_and_adv_in_dataset = [
            key
            for key in tokenized_dataset.keys()
            if (key.startswith("test") or key.startswith("adv"))
        ]

        for split in splits_for_test_and_adv_in_dataset:
            print("results for split: ", split)
            returned_metrics = trainer.evaluate(tokenized_dataset[split])
            all_results_for_test_and_adv[split] = returned_metrics

    return all_results_for_test_and_adv


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
    parser.add_argument("--test_file", type=str, required=False)

    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--split_training_data", action="store_true")

    args = parser.parse_args()
    main(args)
