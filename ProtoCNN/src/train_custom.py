import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse


import string
import warnings

import pandas as pd
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies.ddp import DDPStrategy
from models.protoconv.data_visualizer import DataVisualizer
from utils import plot_html
import os

from sklearn.model_selection import train_test_split
from models.protoconv.lit_module import ProtoConvLitModule
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import functional as F
from torch.nn.utils.rnn import pad_sequence

# from utils import plot_html
from IPython import embed

warnings.simplefilter("ignore")
seed_everything(0)

from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Features, Value


def load_datasets(dataset_name, dir_path):
    def data_loader(split_name, path):
        print("PATH", path)
        features = Features(
            {
                "text": Value("string"),
                "label": Value("float"),
            }
        )

        dataset = load_dataset(
            "csv",
            data_files={
                split_name: path,
            },
            delimiter=",",
            column_names=[
                "text",
                "label",
            ],
            skiprows=1,
            features=features,
            keep_in_memory=True,
        )
        return dataset

    train_dataset = data_loader("train", os.path.join(dir_path, "train.csv"))["train"]
    val_dataset = data_loader("test", os.path.join(dir_path, "test.csv"))["test"]
    return train_dataset, val_dataset


# tokenizer = tokenizer.train_new_from_iterator(train_dataset, vocab_size=10_000)


def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", default="bert-base-uncased", required=False, type=str
    )
    parser.add_argument("--batch_size", default=32, required=False, type=int)
    parser.add_argument(
        "--dataset_name", default="imdb_dataset", required=False, type=str
    )
    parser.add_argument("--num_labels", default=2, required=False, type=int)
    parser.add_argument(
        "--dir_path", default="../../datasets/", required=False, type=str
    )
    parser.add_argument("--type", default="", required=False, type=str)
    parser.add_argument('--use_bigger', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dir_path = os.path.join(args.dir_path, args.dataset_name)

    train_dataset, val_dataset = load_datasets(args.dataset_name, dir_path)
    train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=25000)
    val_dataset = val_dataset.map(preprocess_function, batched=True, batch_size=25000)

    train_dataset = DataLoader(
        train_dataset.shuffle().with_format("torch"), batch_size=args.batch_size
    )
    validation_dataset = DataLoader(
        val_dataset.with_format("torch"), batch_size=args.batch_size
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{args.dataset_name}",
        filename=f"{args.type}-" + "{epoch_0:02d}-{val_loss_0:.4f}-{val_acc_0:.4f}",
        save_weights_only=True,
        save_top_k=1,
        monitor="val_acc_0",
        mode="max",
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor=f"val_loss_0",
            patience=10,
            verbose=True,
            mode="min",
            min_delta=0.005,
        ),
        model_checkpoint,
    ]

    # LR: 5e-3 works best for ag_news
    # Lr: 1e-3:2.5, 5e-3:2.56, 1e-2:best for logic
    # LR: 1e-2 is best for dbpedia
    separation_loss = False if args.type == "without_separation_loss" else True
    clustering_loss = False if args.type == "without_clustering_loss" else True
    use_dynamic_prototypes = False if args.type == "without_dynamic" else True
    with_linear = "linear" if args.type == "with_linear" else "log"

    model = ProtoConvLitModule(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=500,
        fold_id=0,
        lr=1e-3,
        num_labels=args.num_labels,
        pc_conv_filters=64,
        pc_conv_filter_size=5,
        itos={y: x for x, y in tokenizer.vocab.items()},
        verbose_proto=False,
        pc_dynamic_number=use_dynamic_prototypes,
        use_clustering_loss=clustering_loss,
        use_separation_loss=separation_loss,
        pc_sim_func=with_linear,
        use_larger_version=args.use_bigger,
    )

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print("number of trainable parameters:", params)

    trainer = Trainer(
        max_epochs=35, callbacks=callbacks, deterministic=True, num_sanity_val_steps=0
    )
    trainer.fit(
        model, train_dataloaders=train_dataset, val_dataloaders=validation_dataset
    )

    model = ProtoConvLitModule.load_from_checkpoint(model_checkpoint.best_model_path)
