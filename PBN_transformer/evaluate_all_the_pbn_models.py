import pandas as pd
import numpy as np
import json
import os
from argparse import Namespace
import sys
from evaluate_model import main as evaluate_model
from IPython import embed

Models_directory = "/scratch/zhivar/robust-prototype-learning/PBN_transformer/Models"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


batch_size = 256


def process_condition(
    architecture, dataset, attack_type, p1_lamb, p2_lamb, p3_lamb, num_proto, is_cosine
):
    data_dir = f"/scratch/zhivar/robust-prototype-learning/datasets/{dataset}_dataset"
    if is_cosine:
        model_checkpoint = f"{architecture}_{dataset}_model_{p1_lamb}_{p2_lamb}_{p3_lamb}_{num_proto}_cosine"
    else:
        model_checkpoint = (
            f"{architecture}_{dataset}_model_{p1_lamb}_{p2_lamb}_{p3_lamb}_{num_proto}"
        )
    if not os.path.exists(os.path.join(Models_directory, model_checkpoint)):
        return None

    all_results = {}
    for condition in ["test", "adv"]:

        args = Namespace(
            architecture=architecture,
            test_file=f"{condition}_{attack_type}.csv",
            data_dir=data_dir,
            dataset=dataset,
            batch_size=batch_size,
            num_prototypes=num_proto,
            modelname=model_checkpoint,
            model="ProtoTEx",
            use_cosine_dist=is_cosine,
        )

        results = evaluate_model(args)
        all_results[condition] = results
    return all_results


if os.path.exists("all_results_from_pbn_models_static.json"):
    with open("all_results_from_pbn_models_static.json", "r") as f:
        all_results = json.load(f)
        f.close()
    already_existing_conditions = set()
    for result in all_results:
        if "is_cosine" in result.keys():
            is_cosine = result["is_cosine"]
        else:
            is_cosine = False
        already_existing_conditions.add(
            (
                result["architecture"],
                result["dataset"],
                result["attack_type"],
                result["p1_lamb"],
                result["p2_lamb"],
                result["p3_lamb"],
                result["num_proto"],
                is_cosine,
            )
        )
else:
    all_results = []
    already_existing_conditions = set()

is_cosine = True
for architecture in ["BART", "ELECTRA", "BERT"]:
    for dataset in ["dbpedia", "imdb", "ag_news", "sst2"]:
        attack_type_list = (
            ["glue"]
            if dataset == "sst2"
            else ["pwws", "textfooler", "textbugger", "deepwordbug", "bae"]
        )
        for attack_type in attack_type_list:
            for p1_lamb in [0.9]:
                for p2_lamb in [0.9]:
                    for p3_lamb in [0.9]:
                        for num_proto in [16]:
                            if (
                                architecture,
                                dataset,
                                attack_type,
                                p1_lamb,
                                p2_lamb,
                                p3_lamb,
                                num_proto,
                                is_cosine,
                            ) in already_existing_conditions:
                                print("condition already found")
                                continue
                            condition_results = process_condition(
                                architecture,
                                dataset,
                                attack_type,
                                p1_lamb,
                                p2_lamb,
                                p3_lamb,
                                num_proto,
                                is_cosine,
                            )
                            if condition_results is not None:
                                all_results.append(
                                    {
                                        "architecture": architecture,
                                        "dataset": dataset,
                                        "attack_type": attack_type,
                                        "p1_lamb": p1_lamb,
                                        "p2_lamb": p2_lamb,
                                        "p3_lamb": p3_lamb,
                                        "num_proto": num_proto,
                                        "is_cosine": is_cosine,
                                        "results": condition_results,
                                    }
                                )

try:
    with open("all_results_from_pbn_models_static.json", "w") as f:
        json.dump(all_results, f)
except Exception as e:

    import joblib

    joblib.dump(all_results, "all_results_from_pbn_models_static.pkl")
