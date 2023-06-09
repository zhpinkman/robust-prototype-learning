from IPython import embed
import argparse
import textattack
from textattack.transformations import WordSwapRandomCharacterDeletion, BackTranslation
import transformers
from datasets import Dataset
import os
import pandas as pd
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from datasets import Dataset
import argparse
from IPython import embed
from models.protoconv.lit_module import ProtoConvLitModule
import numpy as np
import json
from models.protoconv.data_visualizer import DataVisualizer
from utils import plot_html
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset
from datasets import Features, Value
from torch.utils.data import DataLoader
import torch
import warnings

warnings.filterwarnings("ignore")

from textattack.augmentation import (
    CLAREAugmenter,
    BackTranslationAugmenter,
    CharSwapAugmenter,
    CheckListAugmenter,
    DeletionAugmenter,
    EasyDataAugmenter,
    EmbeddingAugmenter,
    WordNetAugmenter,
)


def load_preprocess(dataset, tokenizer):
    dataset_file = os.path.join("../../datasets", f"{dataset}_dataset", "test.csv")

    try:
        dataset = Dataset.from_pandas(pd.read_csv(dataset_file))

    except Exception as e:
        embed()

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # all_datasets = all_datasets.map(preprocess_function, batched=True)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack_type", type=str, default="textfooler", help="attack type"
    )
    parser.add_argument("--dataset", type=str, default="imdb", help="dataset to use")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name", default="bert-base-uncased", type=str, required=False
    )

    args = parser.parse_args()

    model = ProtoConvLitModule.load_from_checkpoint(
        checkpoint_path=args.model_checkpoint
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(
        model=model, tokenizer=tokenizer
    )

    test_dataset = load_preprocess(args.dataset, tokenizer)

    dataset = textattack.datasets.HuggingFaceDataset(test_dataset)

    if args.attack_type == "textfooler":
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif args.attack_type == "textbugger":
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    elif args.attack_type == "deepwordbug":
        attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)

    print("Loaded attack and dataset")

    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval

    log_file = f"log_{args.dataset}_{args.attack_type}_{args.model_checkpoint.replace('.', '').replace('/', '_')}.csv"
    summary_file = f"summary_{args.dataset}_{args.attack_type}_{args.model_checkpoint.replace('.', '').replace('/', '_')}.json"

    attack_args = textattack.AttackArgs(
        random_seed=1234,
        num_successful_examples=800,
        shuffle=True,
        log_to_csv=log_file,
        log_summary_to_json=summary_file,
        checkpoint_interval=None,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
        parallel=False,
    )
    print("Created attack")
    attacker = textattack.Attacker(attack, dataset, attack_args)

    if args.mode == "attack":
        print("Attacking")
        attacker.attack_dataset()

    embed()

    # if not os.path.exists(f"{args.dataset}_dataset"):
    #     os.makedirs(f"{args.dataset}_dataset")
    #     train_dataset = textattack.datasets.HuggingFaceDataset(
    #         args.dataset, split="train"
    #     )
    #     sentences = []
    #     labels = []
    #     for text, label in train_dataset:
    #         sentences.append(text["text"])
    #         labels.append(label)
    #     pd.DataFrame({"text": sentences, "label": labels}).to_csv(
    #         f"{args.dataset}_dataset/train.csv", index=False
    #     )
    # resulted_df = pd.read_csv(log_file)
    # resulted_df = resulted_df[resulted_df["result_type"] == "Successful"]
    # test_sentences = resulted_df["original_text"].tolist()
    # test_labels = resulted_df["ground_truth_output"].tolist()
    # adv_sentences = resulted_df["perturbed_text"].tolist()

    # pd.DataFrame(
    #     {
    #         "original_text": test_sentences,
    #         "perturbed_text": adv_sentences,
    #         "label": test_labels,
    #     }
    # ).to_csv(
    #     f"{args.dataset}_dataset/adv_{args.attack_type}_{args.model_checkpoint.replace('/', '_')}.csv",
    #     index=False,
    # )


if __name__ == "__main__":
    main()
