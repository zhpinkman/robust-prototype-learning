import argparse
from IPython import embed
from models.protoconv.lit_module import ProtoConvLitModule
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

def load_preprocess(dataset_dir, tokenizer_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    features = Features(
            {
                "text": Value("string"),
                "label": Value("float"),
            }
        )
    
    test_adv_files_paths = [
        file for file in os.listdir(dataset_dir) if ((file.endswith(".csv")) and (file.startswith("test") or file.startswith("adv")))
    ]


    try:
        all_datasets = load_dataset(
            "csv",
            data_files={
                file_name[:file_name.find(".csv")]: os.path.join(dataset_dir, file_name)
                for file_name in test_adv_files_paths
            },
            delimiter=",",
            column_names=[
                "text", "label"
            ],
            skiprows=1,
            features=features,
            keep_in_memory=True,
        )
        
    except Exception as e:
        embed()

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    all_datasets = all_datasets.map(preprocess_function, batched=True)
    for dataset_name in all_datasets.keys():
        yield dataset_name, DataLoader(all_datasets[dataset_name].with_format("torch"), batch_size=1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_checkpoint",
        default="model_checkpoint.ckpt",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--num_labels',
        type=int,
        required=True,
    )

    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str, required=False)
    args = parser.parse_args()

    model = ProtoConvLitModule.load_from_checkpoint(checkpoint_path=args.model_checkpoint)
    # data_visualizer = DataVisualizer(model)
    # plot_html(data_visualizer.visualize_prototypes())
    # plot_html(data_visualizer.visualize_random_predictions(visual_val_dataset, n=5))
    
    for name, loader in load_preprocess(tokenizer_name=args.tokenizer_name, dataset_dir=args.dataset_dir):
            
        print('Testing model on the validation set on the {} dataset'.format(name))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                all_labels.extend(batch["label"].cpu().numpy())
                outputs = model(batch["input_ids"].to(device))
                preds = torch.softmax(outputs.logits, dim=1).argmax(dim=1) if args.num_labels > 2 else torch.round(torch.sigmoid(outputs.logits))
                all_preds.extend(preds.cpu().numpy())
        all_preds = [int(i) for i in all_preds]
                
        from sklearn.metrics import classification_report
        print(classification_report(all_labels, all_preds, zero_division=0, digits=4))