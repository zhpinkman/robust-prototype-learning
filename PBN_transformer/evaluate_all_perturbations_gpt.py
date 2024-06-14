import pandas as pd
import numpy as np
import json
import os
from argparse import Namespace
import sys
from IPython import embed
from openai import OpenAI
from tqdm.auto import tqdm


batch_size = 256

mappings_from_label_to_ids = {
    "imdb": {"negative": 0, "positive": 1},
    "ag_news": {"world": 0, "sports": 1, "business": 2, "science": 3},
    "dbpedia": {
        "agent": 0,
        "work": 1,
        "place": 2,
        "species": 3,
        "unitofwork": 4,
        "event": 5,
        "sportsseason": 6,
        "device": 7,
        "topicalconcept": 8,
    },
    "sst2": {"negative": 0, "positive": 1},
}


def label_2_ids_to_str(dictionary):
    return (
        "{"
        + ", ".join([f'"{key}": {value}' for key, value in dictionary.items()])
        + "}"
    )


def prompt_gpt_model(text, dataset):

    client = OpenAI()

    content = f"Considering the following text, '{text}', return which label it should be associated with, from the following labels, {label_2_ids_to_str(mappings_from_label_to_ids[dataset])}.\nONLY return the index of the label in a \\\\boxed{{}}."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ],
        temperature=0,
        max_tokens=32,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


def process_condition(
    dataset,
    attack_type,
):
    data_dir = f"/scratch/zhivar/robust-prototype-learning/datasets/{dataset}_dataset"

    all_results = {}
    for condition in ["test", "adv"]:

        predictions = []
        df = pd.read_csv(os.path.join(data_dir, f"{condition}_{attack_type}.csv"))
        df = df.sample(100, random_state=42)
        texts = df["text"].values
        labels = df["label"].values

        predictions = [
            prompt_gpt_model(text, dataset) for text in tqdm(texts, leave=False)
        ]

        all_results[condition] = {
            "predictions": predictions,
            "labels": labels,
            "texts": texts,
        }
    return all_results


if os.path.exists("all_results_from_perturbations_gpt_models.json"):
    with open("all_results_from_perturbations_gpt_models.json", "r") as f:
        all_results = json.load(f)
        f.close()
    already_existing_conditions = set()
    for result in all_results:
        already_existing_conditions.add((result["dataset"], result["attack_type"]))
else:
    all_results = []
    already_existing_conditions = set()

    for dataset in tqdm(["dbpedia", "imdb", "ag_news", "sst2"], leave=False):
        attack_type_list = (
            ["glue"]
            if dataset == "sst2"
            else ["pwws", "textfooler", "textbugger", "deepwordbug", "bae"]
        )
        for attack_type in tqdm(attack_type_list, leave=False):
            if (
                dataset,
                attack_type,
            ) in already_existing_conditions:
                print("condition already found")
                continue
            condition_results = process_condition(
                dataset,
                attack_type,
            )
            if condition_results is not None:
                all_results.append(
                    {
                        "dataset": dataset,
                        "attack_type": attack_type,
                        "results": condition_results,
                    }
                )

                try:
                    with open(
                        "all_results_from_perturbations_gpt_models.json", "w"
                    ) as f:
                        json.dump(all_results, f)
                except Exception as e:

                    import joblib

                    joblib.dump(
                        all_results, "all_results_from_perturbations_gpt_models.pkl"
                    )
