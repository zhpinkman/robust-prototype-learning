from IPython import embed
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


def preprocess_data(tokenizer, dataset, args):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return tokenized_dataset


def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

    test_names = [
        file
        for file in os.listdir(data_dir)
        if (file.startswith("test") and file.endswith("csv"))
    ]
    
    test_dfs = [pd.read_csv(os.path.join(data_dir, file)) for file in test_names]

    adv_attack_names = [file for file in os.listdir(data_dir) if file.startswith("adv")]
    adv_attack_dfs = [
        pd.read_csv(os.path.join(data_dir, file)) for file in adv_attack_names
    ]

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
    predictions = np.argmax(predictions, axis=1)
    # use classification report from sklearn
    from sklearn.metrics import classification_report

    print(classification_report(labels, predictions))
    return metric.compute(predictions=predictions, references=labels)


def main(args):
    if args.mode == "train":
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            num_labels=args.num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, num_labels=args.num_labels, ignore_mismatched_sizes=True
        )

    dataset = load_data(args.data_dir)
    tokenized_dataset = preprocess_data(tokenizer, dataset, args)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_dir=args.log_dir,
        report_to="wandb",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        logging_strategy="steps",
        save_steps=args.logging_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.logging_steps,
        load_best_model_at_end=True,
    )

    os.environ["WANDB_PROJECT"] = f"bert-{args.data_dir.split('/')[-1]}"

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.mode == "train":
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        trainer.evaluate(tokenized_dataset["test"])
        trainer.train()
        trainer.save_model(args.model_dir)

    elif args.mode == "test":
        splits_for_test_and_adv_in_dataset = [
            key
            for key in tokenized_dataset.keys()
            if (key.startswith("test") or key.startswith("adv"))
        ]
        for split in splits_for_test_and_adv_in_dataset:
            print("results for split: ", split)
            trainer.evaluate(tokenized_dataset[split])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--num_labels",
        type=int,
        required=True,
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
    parser.add_argument("--num_epochs", type=int, default=3)

    parser.add_argument("--log_dir", type=str, default="./logs")

    parser.add_argument("--max_length", type=int, default=250)

    parser.add_argument("--logging_steps", type=int)

    args = parser.parse_args()
    main(args)
