import pandas as pd
import glob
from sklearn.metrics import classification_report, accuracy_score
import json

import warnings

warnings.filterwarnings("ignore")

mappings = {
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
}


def mapper(x, split):
    return mappings[split][x] if x in mappings[split] else -1


def evaluate(dataset_split):
    results = []

    adv_datasets = [
        i
        for i in glob.glob(f"../datasets/{dataset_split}_gpt_dataset/adv_*.csv")
        if "protocnn" not in i
    ]
    test_datasets = [
        i
        for i in glob.glob(f"../datasets/{dataset_split}_gpt_dataset/test_*.csv")
        if "protocnn" not in i
    ]

    with open("results.txt", "w+") as f:
        for data_path in test_datasets + adv_datasets:
            dataset = pd.read_csv(data_path, names=["text", "pred", "label"])

            dataset["pred"] = dataset["pred"].str.lower()
            dataset["pred"] = dataset["pred"].str.strip()

            if "imdb" in data_path:
                dataset["pred"] = dataset["pred"].apply(mapper, args=("imdb",))
            elif "ag_news" in data_path:
                dataset["pred"] = dataset["pred"].apply(mapper, args=("ag_news",))
            elif "dbpedia" in data_path:
                dataset["pred"] = dataset["pred"].apply(mapper, args=("dbpedia",))

            print(
                "Classification report for: ",
                data_path.split("/")[-1],
                ": ",
                accuracy_score(dataset["label"], dataset["pred"]),
            )
            results.append(
                {
                    f"""{data_path.split("/")[-1]}""": classification_report(
                        dataset["label"], dataset["pred"], output_dict=True
                    )
                }
            )

        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    for ds in ["imdb", "dbpedia", "ag_news"]:
        evaluate(ds)
        print("Done with ", ds, " dataset.")
        print("-" * 100)
