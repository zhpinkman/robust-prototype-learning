import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd




import string
import warnings

import pandas as pd
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",)
batch_size = 32
dataset_name = "imdb"
num_labels = 2
dir_path = f"../../datasets/{dataset_name}/"
mappings = ["Agent", "Work", "Place", "Species", "UnitOfWork", "Event", "SportsSeason", "Device", "TopicalConcept"]

try:
    dataset = load_dataset(dataset_name, keep_in_memory=True)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
except Exception as e:
    def mapper(x):
        return {"label": mappings.index(x["label"])}

    def data_loader(split_name, path):
        features = Features(
            {
                "text": Value("string"),
                "label": Value("string"),
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
    train_dataset = train_dataset.map(mapper)

    val_dataset = data_loader("test", os.path.join(dir_path, "test.csv"))["test"]
    val_dataset = val_dataset.map(mapper)



# tokenizer = tokenizer.train_new_from_iterator(train_dataset, vocab_size=10_000)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

train_dataset = DataLoader(train_dataset.shuffle().with_format("torch"), batch_size=batch_size)
validation_dataset = DataLoader(val_dataset.with_format("torch"), batch_size=batch_size)
# visual_val_dataset = DataLoader(val_dataset.with_format("torch"), batch_size=1)

model_checkpoint = ModelCheckpoint(dirpath=f'checkpoints/{dataset_name}', filename='{epoch_0:02d}-{val_loss_0:.4f}-{val_acc_0:.4f}',
                                   save_weights_only=True, save_top_k=1, monitor='val_acc_0', mode='max')

callbacks = [
    LearningRateMonitor(logging_interval='epoch'),
    EarlyStopping(monitor=f'val_loss_0', patience=10, verbose=True, mode='min', min_delta=0.005),
    model_checkpoint
]

# LR: 5e-3 works best for ag_news
# Lr: 1e-3:2.5, 5e-3:2.56, 1e-2:best for logic
# LR: 1e-2 is best for dbpedia

model = ProtoConvLitModule(vocab_size=tokenizer.vocab_size, embedding_dim=300, fold_id=0, lr=1e-3, num_labels=num_labels, pc_conv_filters=64, pc_conv_filter_size=5,
                           itos={y: x for x, y in tokenizer.vocab.items()}, verbose_proto=False)


trainer = Trainer(max_epochs=35, callbacks=callbacks, deterministic=True, num_sanity_val_steps=0)
trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=validation_dataset)

# embed()
model = ProtoConvLitModule.load_from_checkpoint(model_checkpoint.best_model_path)
# data_visualizer = DataVisualizer(model)
# plot_html(data_visualizer.visualize_prototypes())
# plot_html(data_visualizer.visualize_random_predictions(visual_val_dataset, n=5))