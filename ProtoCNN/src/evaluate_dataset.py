import argparse

from models.protoconv.lit_module import ProtoConvLitModule
from models.protoconv.data_visualizer import DataVisualizer
from utils import plot_html

from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import Features, Value
from torch.utils.data import DataLoader

def load_preprocess(tokenizer_name="bert-base-uncased", ds_path=""):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    features = Features(
            {
                "text": Value("string"),
                "label": Value("float"),
            }
        )

    val_dataset = load_dataset(
        "csv",
        data_files={
            "val": ds_path,
        },
        delimiter=",",
        column_names=[
            "text", "label"
        ],
        skiprows=1,
        features=features,
        keep_in_memory=True,
    )["val"]

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    val_dataset = val_dataset.map(preprocess_function, batched=True)
    visual_val_dataset = DataLoader(val_dataset.with_format("torch"), batch_size=1)
    return visual_val_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        default="dataset.csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_checkpoint",
        default="model_checkpoint.ckpt",
        type=str,
        required=True,
    )

    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str, required=False)
    args = parser.parse_args()
    visual_val_dataset = load_preprocess(ds_path=args.ds_path, tokenizer_name=args.tokenizer_name)
    model = ProtoConvLitModule.load_from_checkpoint(checkpoint_path=args.model_checkpoint)
    data_visualizer = DataVisualizer(model)
    plot_html(data_visualizer.visualize_prototypes())
    plot_html(data_visualizer.visualize_random_predictions(visual_val_dataset, n=5))