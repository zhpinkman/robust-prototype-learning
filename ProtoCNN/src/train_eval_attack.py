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
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# from utils import plot_html
from IPython import embed

warnings.simplefilter("ignore")
seed_everything(0)

# glove_vocab = torchtext.vocab.GloVe("6B", cache="vectors/")
# glove_vocab = vocab(glove_vocab.stoi)

# def yield_tokens(data_iter):
#     for _, text in data_iter:
#         yield tokenizer(text)


batch_size = 32

# train_datapipe = IMDB(split="train")
# dev_datapipe = IMDB(split="test")


# # vocab_transform = torchtext.transforms.VocabTransform(vocab(glove_vocab.stoi))
# tokenizer = get_tokenizer('basic_english')
# vocab = build_vocab_from_iterator(yield_tokens(iter(train_datapipe)), specials=["<unk>", "<pad>"])
# vocab.set_default_index(vocab["<unk>"])

# # Transform the raw dataset using non-batched API (i.e apply transformation line by line)
# def apply_transform(x):
#     # return text_transform(x[0]), x[1]
#     # print(x[0])
#     # return x[0], glove_vocab.get_vecs_by_tokens(x[1][:].split(" "))
#     tokenized = vocab(x[1].split(" "))
#     truncated = torchtext.functional.truncate(tokenized, 128)
    
#     return x[0], truncated 

# to_tensor = torchtext.transforms.ToTensor()


# train_datapipe = train_datapipe.map(apply_transform)
# # train_datapipe = train_datapipe.map(to_tensor)
# train_datapipe = train_datapipe.batch(batch_size)
# train_datapipe = train_datapipe.rows2columnar(["label", "text"])
# train_dataloader = DataLoader(train_datapipe, batch_size=None)

# dev_datapipe = dev_datapipe.map(apply_transform)
# # dev_datapipe = dev_datapipe.map(to_tensor)
# dev_datapipe = dev_datapipe.batch(batch_size)
# dev_datapipe = dev_datapipe.rows2columnar(["label", "text"])
# dev_dataloader = DataLoader(dev_datapipe, batch_size=None)

# class CustomDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = pd.read_csv(csv_file)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Assuming your CSV has two columns: 'input' and 'target'
#         input_data = self.data.iloc[idx]['text']
#         tokenized = vocab(input_data.split(" "))
#         truncated = torchtext.functional.truncate(tokenized, 128)
#         # truncated = torchtext.functional.to_tensor(truncated, padding_value=1)
#         target = self.data.iloc[idx]['label']
#         return {"label": target, "text": truncated}

# def collate_fn(batch):
#     # Sort the batch by input_data length (descending order)
#     batch = sorted(batch, key=lambda x: len(x["text"]), reverse=True)
#     input_data, targets = zip(*batch)
#     print(input_data)
    
#     # Pad the input_data sequences
#     input_data = [torch.Tensor(data) for data in input_data]
#     input_data = pad_sequence(input_data, batch_first=True, padding_value=1)
    
#     # Convert targets to tensors
#     targets = torch.Tensor(targets)
    
#     return input_data, targets

# dataset = CustomDataset("/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/data/perturbed_dataset.csv")
# dev_dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# embed()

from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import Features, Value

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",)

dataset = load_dataset("imdb", keep_in_memory=True)
train_dataset = dataset["train"]
# val_dataset = dataset["test"]

features = Features(
        {
            # "cogency_mean": Value("float"),
            # "effectiveness_mean": Value("float"),
            # "reasonableness_mean": Value("float"),
            # "text": Value("string"),
            # "title": Value("string"),
            # "similar": Value("string"),
            "text": Value("string"),
            "label": Value("float"),
        }
    )

val_dataset = load_dataset(
    "csv",
    data_files={
        "val": "/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/data/textbugger_imdb.csv",
    },
    delimiter=",",
    column_names=[
        "text", "label"
    ],
    skiprows=1,
    features=features,
    keep_in_memory=True,
)["val"]


# tokenizer = tokenizer.train_new_from_iterator(train_dataset, vocab_size=10_000)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

train_dataset = DataLoader(train_dataset.shuffle().with_format("torch"), batch_size=batch_size)
validation_dataset = DataLoader(val_dataset.with_format("torch"), batch_size=batch_size)
visual_val_dataset = DataLoader(val_dataset.with_format("torch"), batch_size=1)


# embed()

model_checkpoint = ModelCheckpoint(dirpath='checkpoints/', filename='{epoch_0:02d}-{val_loss_0:.4f}-{val_acc_0:.4f}',
                                   save_weights_only=True, save_top_k=1, monitor='val_acc_0')

callbacks = [
    LearningRateMonitor(logging_interval='epoch'),
    EarlyStopping(monitor=f'val_loss_0', patience=10, verbose=True, mode='min', min_delta=0.005),
    model_checkpoint
]

model = ProtoConvLitModule(vocab_size=tokenizer.vocab_size, embedding_dim=300, fold_id=0, lr=1e-3,
                           itos={y: x for x, y in tokenizer.vocab.items()}, verbose_proto=False)


trainer = Trainer(max_epochs=35, callbacks=callbacks, deterministic=True, num_sanity_val_steps=0, strategy=DDPStrategy(find_unused_parameters=True),)
trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=validation_dataset)

# embed()
model = ProtoConvLitModule.load_from_checkpoint(model_checkpoint.best_model_path)
data_visualizer = DataVisualizer(model)
plot_html(data_visualizer.visualize_prototypes())
plot_html(data_visualizer.visualize_random_predictions(visual_val_dataset, n=5))