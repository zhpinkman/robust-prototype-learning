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
import torch
from transformers import AutoTokenizer
from models import ProtoTEx
from models_electra import ProtoTEx_Electra
from models_bert import ProtoTEx_BERT

import sys

sys.path.append("../datasets")
import configs
import utils

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
    parser.add_argument("--mode", type=str)
    parser.add_argument("--model_checkpoint", type=str)
    ########
    parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true")
    # parser.add_argument("--nli_dataset", help="check if the dataset is in nli
    # format that has sentence1, sentence2, label", action="store_true")
    parser.add_argument("--num_prototypes", type=int, default=16)
    parser.add_argument("--model", type=str, default="ProtoTEx")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default="3e-5")

    # Wandb parameters
    parser.add_argument("--project", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--nli_intialization", type=str, default="Yes")
    parser.add_argument("--none_class", type=str, default="No")
    parser.add_argument("--curriculum", type=str, default="No")
    parser.add_argument("--augmentation", type=str, default="No")
    parser.add_argument("--architecture", type=str, default="BART")

    args = parser.parse_args()

    log_file = f"logs/log_{args.dataset}_{args.attack_type}_{args.model_checkpoint.replace('.', '').replace('/', '_')}.csv"
    summary_file = f"summaries/summary_{args.dataset}_{args.attack_type}_{args.model_checkpoint.replace('.', '').replace('/', '_')}.json"

    if os.path.exists(log_file):
        print("Log file: {0} already exists".format(log_file))
        return

    if args.architecture == "BART":
        tokenizer = AutoTokenizer.from_pretrained("ModelTC/bart-base-mnli")
    elif args.architecture == "ELECTRA":
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    elif args.architecture == "BERT":
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium")
    else:
        print(f"Invalid backbone architecture: {args.architecture}")

    if args.model == "ProtoTEx":
        print("ProtoTEx best model: {0}".format(args.num_prototypes))
        if args.architecture == "BART":
            print(f"Using backone: {args.architecture}")
            torch.cuda.empty_cache()
            model = ProtoTEx(
                num_prototypes=args.num_prototypes,
                class_weights=None,
                n_classes=configs.dataset_to_num_labels[args.dataset],
                max_length=configs.dataset_to_max_length[args.dataset],
                bias=False,
                special_classfn=True,
                p=1,  # p=0.75,
                batchnormlp1=True,
            )
        elif args.architecture == "ELECTRA":
            model = ProtoTEx_Electra(
                num_prototypes=args.num_prototypes,
                class_weights=None,
                n_classes=configs.dataset_to_num_labels[args.dataset],
                max_length=configs.dataset_to_max_length[args.dataset],
                bias=False,
                special_classfn=True,
                p=1,  # p=0.75,
                batchnormlp1=True,
            )
        elif args.architecture == "BERT":
            model = ProtoTEx_BERT(
                num_prototypes=args.num_prototypes,
                class_weights=None,
                n_classes=configs.dataset_to_num_labels[args.dataset],
                max_length=configs.dataset_to_max_length[args.dataset],
                bias=False,
                special_classfn=True,
                p=1,  # p=0.75,
                batchnormlp1=True,
            )
        else:
            print(f"Invalid backbone architecture: {args.architecture}")

    print(f"Loading model checkpoint: Models/{args.model_checkpoint}")
    pretrained_dict = torch.load(f"Models/{args.model_checkpoint}")
    # Fiter out unneccessary keys
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            print(f"Skipping weights for: {k}")
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    all_datasets = utils.load_dataset(
        data_dir=f"../datasets/{args.dataset}_dataset",
        tokenizer=tokenizer,
        max_length=configs.dataset_to_max_length[args.dataset],
    )
    print(all_datasets.keys())

    test_dl = torch.utils.data.DataLoader(
        all_datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids": torch.LongTensor([i["input_ids"] for i in batch]),
            "attention_mask": torch.Tensor([i["attention_mask"] for i in batch]),
            "label": torch.LongTensor([i["label"] for i in batch]),
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    (
        total_loss,
        mac_prec,
        mac_recall,
        mac_f1_score,
        accuracy,
        y_true,
        y_pred,
    ) = utils.evaluate(test_dl, model_new=model)
    if mac_f1_score < 0.6:
        print("This model is not accurate enough in the first place")
        model = model.to("cpu")
        if not os.path.exists(log_file):
            open(log_file, "w").close()
        if not os.path.exists(summary_file):
            open(summary_file, "w").close()
        return

    # put on cpu
    model = model.to("cpu")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(model)
    # model = model.to(device)

    if not os.path.exists(log_file):
        open(log_file, "w").close()
    if not os.path.exists(summary_file):
        open(summary_file, "w").close()

    if args.mode == "attack":
        model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(
            model=model, tokenizer=tokenizer
        )

        if args.dataset == "dbpedia":
            dataset = textattack.datasets.HuggingFaceDataset(
                Dataset.from_pandas(
                    pd.read_csv(f"../datasets/{args.dataset}_dataset/test.csv")
                )
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
        f"{args.dataset}_dataset/adv_{args.attack_type}_{args.model_checkpoint.replace('/', '_')}.csv",
        index=False,
    )
    print("Saved adv dataset")


if __name__ == "__main__":
    main()
