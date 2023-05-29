import argparse
import textattack
from textattack.transformations import WordSwapRandomCharacterDeletion, BackTranslation
import transformers
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)

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
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack_type", type=str, default="textfooler", help="attack type"
    )
    parser.add_argument("--dataset", type=str, default="imdb", help="dataset to use")

    args = parser.parse_args()

    # print(args.attack_type)

    # from textattack.augmentation import CLAREAugmenter

    # augmenter = CLAREAugmenter(pct_words_to_swap=0.2, transformations_per_example=5)
    # s = "I'd love to go to Japan but the tickets are 500 dollars"
    # print(augmenter.augment(s))

    # dataset = pd.read_csv("test.csv")
    # sentences = dataset["sentence"].tolist()

    if args.dataset == "agnews":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-ag-news"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "textattack/bert-base-uncased-ag-news"
        )
    elif args.dataset == "imdb":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-imdb"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "textattack/bert-base-uncased-imdb"
        )

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    if args.dataset == "agnews":
        dataset = textattack.datasets.HuggingFaceDataset("ag_news", split="test")
    elif args.dataset == "imdb":
        dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

    if args.attack_type == "textfooler":
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif args.attack_type == "textbugger":
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)

    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        random_seed=1234,
        num_examples=200,
        shuffle=True,
        log_to_csv=f"log_{args.dataset}_{args.attack_type}.csv",
        log_summary_to_json=f"summary_{args.dataset}_{args.attack_type}.json",
        checkpoint_interval=None,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
        parallel=True,
    )
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()


if __name__ == "__main__":
    main()
