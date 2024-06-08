import pandas as pd
import numpy as np
import json
import os
from argparse import Namespace
import sys
from IPython import embed
from vanilla_model import main as evaluate_model

Models_directory = "/scratch/zhivar/robust-prototype-learning/normal_models/models"
os.environ["WANDB_MODE"] = "offline"


# def get_batch_size(dataset):
#     dataset_to_batch_size = {"imdb": 16, "dbpedia": 16, "ag_news": 256, "sst2": 256}
#     return dataset_to_batch_size[dataset]


batch_size = 256


def process_condition(architecture, dataset, attack_type):
    data_dir = f"/scratch/zhivar/robust-prototype-learning/datasets/{dataset}_dataset"
    model_checkpoint = os.path.join(Models_directory, f"{dataset}_{architecture}")
    if not os.path.exists(model_checkpoint):
        print(f"Model checkpoint not found: {model_checkpoint}")
        return None

    all_results = {}
    for condition in ["test", "adv"]:

        args = Namespace(
            dataset=dataset,
            mode="test",
            model_dir=model_checkpoint,
            num_epochs=10,
            logging_steps=100,
            log_dir="/scratch/zhivar/robust-prototype-learning/normal_models/logs",
            test_file=f"{condition}_{attack_type}.csv",
            data_dir=data_dir,
            batch_size=batch_size,
            split_training_data=False,
            learning_rate=1e-4,
        )

        results = evaluate_model(args)
        all_results[condition] = results
    return all_results


if os.path.exists("all_results_from_non_pbn_models_static.json"):
    with open("all_results_from_non_pbn_models_static.json", "r") as f:
        all_results = json.load(f)
        f.close()
    already_existing_conditions = set()
    for result in all_results:
        already_existing_conditions.add(
            (
                result["architecture"],
                result["dataset"],
                result["attack_type"],
            )
        )
else:
    already_existing_conditions = set()
    all_results = []

for architecture in [
    "ModelTC/bart-base-mnli",
    "google/electra-base-discriminator",
    "prajjwal1/bert-medium",
]:
    for dataset in ["dbpedia", "imdb", "ag_news", "sst2"]:
        attack_type_list = (
            ["glue"]
            if dataset == "sst2"
            else ["pwws", "textfooler", "textbugger", "deepwordbug", "bae"]
        )
        for attack_type in attack_type_list:
            if (architecture, dataset, attack_type) in already_existing_conditions:
                print("Skipping", architecture, dataset, attack_type)
                continue
            condition_results = process_condition(
                architecture,
                dataset,
                attack_type,
            )
            if condition_results is not None:
                all_results.append(
                    {
                        "architecture": architecture,
                        "dataset": dataset,
                        "attack_type": attack_type,
                        "results": condition_results,
                    }
                )

try:
    with open("all_results_from_non_pbn_models_static.json", "w") as f:
        json.dump(all_results, f)
except Exception as e:

    import joblib

    joblib.dump(all_results, "all_results_from_non_pbn_models_static.pkl")
