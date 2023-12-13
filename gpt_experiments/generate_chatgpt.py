from datasets import load_dataset
import datasets
import glob
from gpt_extractor import generate_chat_completion
from prompts import prompts
from tqdm import tqdm
import os
import pandas as pd
import csv
import time


def sample(dataset_split):
    adv_datasets = [
        i
        for i in glob.glob(f"../datasets/{dataset_split}_dataset/adv_*.csv")
        if "protocnn" not in i
    ]
    test_datasets = [
        i
        for i in glob.glob(f"../datasets/{dataset_split}_dataset/test_*.csv")
        if "protocnn" not in i
    ]

    system_prompt = "You are a helpful assistant that answers the question of the user and follows the specific template provided by the user."

    for data_path in test_datasets + adv_datasets:
        if not os.path.exists(
            os.path.dirname(data_path.replace(dataset_split, f"{dataset_split}_gpt"))
        ):
            print(
                "Creating: ",
                os.path.dirname(
                    data_path.replace(dataset_split, f"{dataset_split}_gpt")
                ),
            )
            os.mkdir(
                os.path.dirname(
                    data_path.replace(dataset_split, f"{dataset_split}_gpt")
                )
            )

        print("-" * 100)
        print("Starting with: ", data_path)
        print("-" * 100)

        dataset = datasets.Dataset.from_pandas(pd.read_csv(data_path))

        with open(data_path.replace(dataset_split, f"{dataset_split}_gpt"), "a") as f:
            csv_writer = csv.writer(f)
            for index, i in enumerate(tqdm(dataset)):
                user_prompt = prompts[dataset_split].format(text=i["text"])
                try:
                    completion = generate_chat_completion(
                        system_prompt, user_prompt, max_tokens=5, model="gpt-3.5-turbo"
                    )
                except Exception as e:
                    print(e, "\n\nSleeping for 10 seconds")
                    time.sleep(10)
                    completion = generate_chat_completion(
                        system_prompt, user_prompt, max_tokens=5, model="gpt-3.5-turbo"
                    )
                csv_writer.writerow([i["text"], completion, i["label"]])


if __name__ == "__main__":
    for ds in ["imdb", "ag_news", "dbpedia"]:
        sample(ds)
        print("Done with ", ds, " dataset.")
        print("-" * 100)
