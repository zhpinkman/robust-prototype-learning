from IPython import embed
import argparse
import textattack
import transformers
from datasets import Dataset
import os
import pandas as pd


dbpedia_dataset_classes = [
    "Agent",
    "Work",
    "Place",
    "Species",
    "UnitOfWork",
    "Event",
    "SportsSeason",
    "Device",
    "TopicalConcept",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack_type", type=str, default="textfooler", help="attack type"
    )
    parser.add_argument("--dataset", type=str, default="imdb", help="dataset to use")
    parser.add_argument(
        "--mode", type=str, default="attack", choices=["attack", "read"]
    )
    parser.add_argument("--model_checkpoint", type=str)

    args = parser.parse_args()

    log_file = f"logs/log_{args.dataset}_{args.attack_type}_{args.model_checkpoint.replace('.', '').replace('/', '_')}.csv"
    summary_file = f"summaries/summary_{args.dataset}_{args.attack_type}_{args.model_checkpoint.replace('.', '').replace('/', '_')}.json"

    if args.mode == "attack" or not os.path.exists(log_file):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model_checkpoint
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_checkpoint)

        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer
        )

        if args.dataset in ["dbpedia", "olid"]:
            dataset = textattack.datasets.HuggingFaceDataset(
                Dataset.from_pandas(pd.read_csv(f"{args.dataset}_dataset/test.csv"))
            )
        else:
            dataset = textattack.datasets.HuggingFaceDataset(args.dataset, split="test")

        if args.attack_type == "textfooler":
            attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
        elif args.attack_type == "textbugger":
            attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
        elif args.attack_type == "deepwordbug":
            attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
        elif args.attack_type == "a2t":
            attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper)
        elif args.attack_type == "checklist":
            attack = textattack.attack_recipes.CheckList2020.build(model_wrapper)
        elif args.attack_type == "hotflip":
            attack = textattack.attack_recipes.HotFlipEbrahimi2017.build(model_wrapper)
        elif args.attack_type == "iga":
            attack = textattack.attack_recipes.IGAWang2019.build(model_wrapper)
        elif args.attack_type == "bae":
            attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)
        elif args.attack_type == "input_reduction":
            attack = textattack.attack_recipes.InputReductionFeng2018.build(
                model_wrapper
            )
        elif args.attack_type == "kuleshov":
            attack = textattack.attack_recipes.Kuleshov2017.build(model_wrapper)
        elif args.attack_type == "swarm":
            attack = textattack.attack_recipes.PSOZang2020.build(model_wrapper)
        elif args.attack_type == "pwws":
            attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
        elif args.attack_type == "clare":
            attack = textattack.attack_recipes.CLARE2020.build(model_wrapper)
        elif args.attack_type == "pruthi":
            attack = textattack.attack_recipes.Pruthi2019.build(model_wrapper)
        print("Loaded attack and dataset")

        # Attack 20 samples with CSV logging and checkpoint saved every 5 interval

        attack_args = textattack.AttackArgs(
            random_seed=1234,
            num_successful_examples=800,
            shuffle=True,
            log_to_csv=log_file,
            log_summary_to_json=summary_file,
            checkpoint_interval=None,
            checkpoint_dir="checkpoints",
            disable_stdout=True,
            parallel=True,
        )
        print("Created attack")
        attacker = textattack.Attacker(attack, dataset, attack_args)

        print("Attacking")
        attacker.attack_dataset()

    if not os.path.exists(f"{args.dataset}_dataset"):
        os.makedirs(f"{args.dataset}_dataset")
        train_dataset = textattack.datasets.HuggingFaceDataset(
            args.dataset, split="train"
        )
        sentences = []
        labels = []
        for text, label in train_dataset:
            sentences.append(text["text"])
            labels.append(label)
        pd.DataFrame({"text": sentences, "label": labels}).to_csv(
            f"{args.dataset}_dataset/train.csv", index=False
        )
    resulted_df = pd.read_csv(log_file)
    resulted_df = resulted_df[resulted_df["result_type"] == "Successful"]
    test_sentences = resulted_df["original_text"].tolist()
    test_labels = resulted_df["ground_truth_output"].tolist()
    adv_sentences = resulted_df["perturbed_text"].tolist()

    pd.DataFrame(
        {
            "original_text": test_sentences,
            "perturbed_text": adv_sentences,
            "label": test_labels,
        }
    ).to_csv(
        f"{args.dataset}_dataset/summaries/adv_{args.attack_type}_{args.model_checkpoint.replace('/', '_')}.csv",
        index=False,
    )
    print("Saved adv dataset")


if __name__ == "__main__":
    main()
