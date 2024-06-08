import pandas as pd
import numpy as np
import json
import os
from argparse import Namespace
import sys
from IPython import embed
from vanilla_model import main as train_model
import argparse

Models_directory = "/scratch/zhivar/robust-prototype-learning/normal_models/models"
augmented_Models_directory = (
    "/scratch/zhivar/robust-prototype-learning/normal_models/augmented_models"
)
os.environ["WANDB_MODE"] = "offline"


def get_batch_size(dataset):
    dataset_to_batch_size = {"imdb": 16, "dbpedia": 16, "ag_news": 256, "sst2": 256}
    return dataset_to_batch_size[dataset]


def process_condition_training(architecture, dataset, attack_type):
    data_dir = f"/scratch/zhivar/robust-prototype-learning/datasets/{dataset}_dataset"
    model_checkpoint = os.path.join(Models_directory, f"{dataset}_{architecture}")
    model_dir = os.path.join(
        augmented_Models_directory, f"{dataset}_{attack_type}_{dataset}_{architecture}"
    )
    log_dir = "/scratch/zhivar/robust-prototype-learning/normal_models/augmented_logs"
    if not os.path.exists(model_checkpoint):
        print(f"Model checkpoint not found: {model_checkpoint}")
        return None

    args = Namespace(
        mode="train",
        batch_size=get_batch_size(dataset),
        logging_steps=20,
        num_epochs=3,
        dataset=dataset,
        data_dir=data_dir,
        model_checkpoint=model_checkpoint,
        model_dir=model_dir,
        log_dir=log_dir,
        test_file=f"adv_{attack_type}.csv",
        learning_rate=5e-5,
        split_training_data=True,
    )

    train_model(args)
    return


def process_condition_evaluation(architecture, dataset, attack_type):
    data_dir = f"/scratch/zhivar/robust-prototype-learning/datasets/{dataset}_dataset"
    model_checkpoint = os.path.join(Models_directory, f"{dataset}_{architecture}")
    model_dir = os.path.join(
        augmented_Models_directory, f"{dataset}_{attack_type}_{dataset}_{architecture}"
    )
    log_dir = "/scratch/zhivar/robust-prototype-learning/normal_models/augmented_logs"
    if not os.path.exists(model_checkpoint):
        print(f"Model checkpoint not found: {model_checkpoint}")
        return None

    args = Namespace(
        mode="test",
        batch_size=256,
        logging_steps=20,
        num_epochs=3,
        dataset=dataset,
        data_dir=data_dir,
        model_dir=model_dir,
        log_dir=log_dir,
        test_file=f"adv_{attack_type}.csv",
        learning_rate=5e-5,
        split_training_data=True,
    )

    results = train_model(args)
    return results


def main(args):
    all_results = []

    for architecture in [
        "ModelTC/bart-base-mnli",
        "google/electra-base-discriminator",
        "prajjwal1/bert-medium",
    ]:
        for dataset in ["dbpedia", "imdb", "ag_news"]:
            for attack_type in [
                "textfooler",
                "textbugger",
                "deepwordbug",
                "pwws",
                "bae",
            ]:
                if args.mode == "train":
                    process_condition_training(
                        architecture,
                        dataset,
                        attack_type,
                    )
                else:
                    results = process_condition_evaluation(
                        architecture,
                        dataset,
                        attack_type,
                    )
                    all_results.append(
                        {
                            "architecture": architecture,
                            "dataset": dataset,
                            "attack_type": attack_type,
                            "results": results,
                        }
                    )

    try:
        with open(
            "all_results_from_non_pbn_models_static_augmented_training.json", "w"
        ) as f:
            json.dump(all_results, f)
    except Exception as e:

        import joblib

        joblib.dump(
            all_results, "all_results_from_non_pbn_models_static_augmented_training.pkl"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    args = parser.parse_args()
    main(args)
