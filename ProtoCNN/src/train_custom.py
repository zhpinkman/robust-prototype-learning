import string
import warnings

import pandas as pd
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import train_test_split
# from torchtext import data
# from torchtext.data import BucketIterator
# from torchtext.vocab import GloVe

# from dataframe_dataset import DataFrameDataset
# from models.protoconv.data_visualizer import DataVisualizer
from models.protoconv.lit_module import ProtoConvLitModule
import torchtext
from torchtext.datasets import IMDB, SST2
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import functional as F
# from utils import plot_html

warnings.simplefilter("ignore")
seed_everything(0)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# glove_vocab = torchtext.vocab.GloVe("6B", cache="vectors/")
batch_size = 32

train_datapipe = IMDB(split="train")
dev_datapipe = IMDB(split="test")


# vocab_transform = torchtext.transforms.VocabTransform(vocab(glove_vocab.stoi))
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(yield_tokens(iter(train_datapipe)), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
def apply_transform(x):
    # return text_transform(x[0]), x[1]
    # print(x[0])
    # return x[0], glove_vocab.get_vecs_by_tokens(x[1][:].split(" "))
    tokenized = vocab(x[1].split(" "))
    truncated = torchtext.functional.truncate(tokenized, 128)
    
    return x[0], truncated 

to_tensor = torchtext.transforms.ToTensor()


train_datapipe = train_datapipe.map(apply_transform)
# train_datapipe = train_datapipe.map(to_tensor)
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["label", "text"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(apply_transform)
# dev_datapipe = dev_datapipe.map(to_tensor)
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["label", "text"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)



model_checkpoint = ModelCheckpoint(dirpath='checkpoints/', filename='{epoch_0:02d}-{val_loss_0:.4f}-{val_acc_0:.4f}',
                                   save_weights_only=True, save_top_k=1, monitor='val_acc_0')

callbacks = [
    LearningRateMonitor(logging_interval='epoch'),
    EarlyStopping(monitor=f'val_loss_0', patience=10, verbose=True, mode='min', min_delta=0.005),
    model_checkpoint
]

model = ProtoConvLitModule(vocab_size=len(vocab), embedding_dim=300, fold_id=0, lr=1e-3,
                           itos=vocab.get_itos(), verbose_proto=False)


trainer = Trainer(max_epochs=35, callbacks=callbacks, deterministic=True, num_sanity_val_steps=0)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader)
