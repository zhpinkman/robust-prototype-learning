from IPython import embed
import argparse
import textattack
from textattack.transformations import WordSwapRandomCharacterDeletion, BackTranslation
import transformers
import os
import pandas as pd
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack_type", type=str, default="textfooler", help="attack type"
    )
    parser.add_argument("--dataset", type=str, default="imdb", help="dataset to use")
    parser.add_argument("--mode", type = str)

    args = parser.parse_args()

    # print(args.attack_type)

    # from textattack.augmentation import CLAREAugmenter

    # augmenter = CLAREAugmenter(pct_words_to_swap=0.2, transformations_per_example=5)
    # s = "I'd love to go to Japan but the tickets are 500 dollars"
    # print(augmenter.augment(s))

    # dataset = pd.read_csv("test.csv")
    # sentences = dataset["sentence"].tolist()

    if args.dataset == "ag_news":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "../normal_models/models/ag_news"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "../normal_models/models/ag_news"
        )
    elif args.dataset == "imdb":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "../normal_models/models/imdb"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "../normal_models/models/imdb"
        )

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    dataset = textattack.datasets.HuggingFaceDataset(args.dataset, split="test")

    if args.attack_type == "textfooler":
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif args.attack_type == "textbugger":
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)

    print("Loaded attack and dataset")

    # Attack 20 samples with CSV logging and checkpoint saved every 5 interval
    attack_args = textattack.AttackArgs(
        random_seed=1234,
        num_successful_examples=200,
        shuffle=True,
        log_to_csv=f"log_{args.dataset}_{args.attack_type}.csv",
        log_summary_to_json=f"summary_{args.dataset}_{args.attack_type}.json",
        checkpoint_interval=None,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
        parallel=True,
    )
    print("Created attack")
    attacker = textattack.Attacker(attack, dataset, attack_args)
    print("Attacking")
    
    if args.mode == "attack":
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
    resulted_df = pd.read_csv(f"log_{args.dataset}_{args.attack_type}.csv")
    resulted_df = resulted_df[resulted_df["result_type"] == "Successful"]
    test_sentences = [i.replace("[", "").replace("]", "") for i in resulted_df["original_text"].tolist()]
    test_labels = resulted_df["ground_truth_output"].tolist()
    adv_sentences = [i.replace("[", "").replace("]", "") for i in resulted_df["perturbed_text"].tolist()]
    adv_labels = resulted_df["ground_truth_output"].tolist()
    pd.DataFrame({"text": test_sentences, "label": test_labels}).to_csv(
        f"{args.dataset}_dataset/test_{args.attack_type}.csv", index=False
    )
    pd.DataFrame({"text": adv_sentences, "label": adv_labels}).to_csv(
        f"{args.dataset}_dataset/adv_{args.attack_type}.csv", index=False
    )
    


if __name__ == "__main__":
    main()
