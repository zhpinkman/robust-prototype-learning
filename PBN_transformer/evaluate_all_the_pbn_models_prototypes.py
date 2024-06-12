import pandas as pd
import numpy as np
import json
import os
import joblib
from argparse import Namespace
import sys
from inference_and_explanations import main as inference
from IPython import embed

Models_directory = "/scratch/zhivar/robust-prototype-learning/PBN_transformer/Models"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


batch_size = 256


def process_condition(
    architecture, dataset, attack_type, p1_lamb, p2_lamb, p3_lamb, num_proto
):
    data_dir = f"/scratch/zhivar/robust-prototype-learning/datasets/{dataset}_dataset"
    model_checkpoint = (
        f"{architecture}_{dataset}_model_{p1_lamb}_{p2_lamb}_{p3_lamb}_{num_proto}"
    )
    if not os.path.exists(os.path.join(Models_directory, model_checkpoint)):
        return None

    all_results = {}
    args = Namespace(
        architecture=architecture,
        test_file="train.csv",
        data_dir=data_dir,
        dataset=dataset,
        batch_size=batch_size,
        num_prototypes=num_proto,
        modelname=model_checkpoint,
        use_cosine_dist=False,
        model="ProtoTEx",
        mode="prototypes",
    )
    results = inference(args)
    all_results["prototypes"] = results

    for condition in ["test", "adv"]:

        args = Namespace(
            architecture=architecture,
            test_file=f"{condition}_{attack_type}.csv",
            data_dir=data_dir,
            dataset=dataset,
            batch_size=batch_size,
            num_prototypes=num_proto,
            modelname=model_checkpoint,
            use_cosine_dist=False,
            model="ProtoTEx",
            mode="data",
        )

        results = inference(args)
        all_results[condition] = results
    return all_results


output_directory = "prototypes_analysis_logs"

json_files_already_computed = [
    file for file in os.listdir(output_directory) if file.endswith(".json")
]


for architecture in ["BART"]:
    for dataset in ["ag_news", "sst2", "dbpedia", "imdb"]:
        attack_type_list = (
            ["glue"]
            if dataset == "sst2"
            else ["pwws", "textfooler", "textbugger", "deepwordbug", "bae"]
        )
        for attack_type in attack_type_list:
            for p1_lamb in [0.9, 0.0, 10.0]:
                for p2_lamb in [0.9, 0.0, 10.0]:
                    for p3_lamb in [0.9, 0.0, 10.0]:
                        for num_proto in [4, 2, 8, 16, 64]:
                            if p1_lamb == 0.0 and p2_lamb == 0.0 and p3_lamb == 0.0:
                                continue
                            json_file_name_to_save = os.path.join(
                                output_directory,
                                f"{architecture}_{dataset}_{attack_type}_{p1_lamb}_{p2_lamb}_{p3_lamb}_{num_proto}.json",
                            )
                            if json_file_name_to_save in json_files_already_computed:
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
                            )
                            if condition_results is not None:
                                all_results = {
                                    "architecture": architecture,
                                    "dataset": dataset,
                                    "attack_type": attack_type,
                                    "p1_lamb": p1_lamb,
                                    "p2_lamb": p2_lamb,
                                    "p3_lamb": p3_lamb,
                                    "num_proto": num_proto,
                                    "results": condition_results,
                                }

                                try:
                                    with open(json_file_name_to_save, "w") as f:
                                        json.dump(all_results, f)
                                        f.close()
                                    print(
                                        "Saved the json file containing the prototypes"
                                    )
                                except Exception as e:
                                    print("error in saving the json file")
                                    print(e)
                                    continue
