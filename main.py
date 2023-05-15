import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import math
import torch.utils.data as data_utils
import torch.nn.functional as F
from Models import *
from train_utils import *
from transformers import BertTokenizer
import torch.utils.data as utils
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="sst2", help="data directory")
    parser.add_argument("--use_max_length", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01, help="initial_learning_rate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument(
        "--h", type=int, default=2, help="dimension of the hidden layer"
    )
    parser.add_argument(
        "--scale", type=float, default=2, help="scaling factor for distance"
    )
    parser.add_argument(
        "--reg", type=float, default=0.1, help="regularization coefficient"
    )

    args, _ = parser.parse_known_args()

    def reshape_dataset(dataset, height, width):
        new_dataset = []
        for k in range(0, dataset.shape[0]):
            new_dataset.append(np.reshape(dataset[k], [1, height, width]))

        return np.array(new_dataset)

    class LoadDataset(Dataset):
        def __init__(self, data, target, transform=None):
            self.data = torch.from_numpy(data).float()
            self.target = torch.from_numpy(target).long()
            self.transform = transform

        def __getitem__(self, index):
            x = self.data[index]
            y = self.target[index]

            if self.transform:
                x = self.transform(x)

            return x, y

        def __len__(self):
            return len(self.data)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset_info = DatasetInfo(
        data_dir=args.data_dir, use_max_length=args.use_max_length
    )
    train_dataset, val_dataset, test_dataset = load_dataset(
        dataset_info=dataset_info, data_dir=args.data_dir, tokenizer=tokenizer
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, collate_fn=val_dataset.collate_fn
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=test_dataset.collate_fn
    )

    train_num = len(train_dataset)
    test_num = len(test_dataset)

    model = Net(args.h, args.num_classes, args.scale)
    model = model.cuda()

    lrate = args.lr
    optimizer_s = optim.SGD(
        model.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-4
    )

    num_epochs = 10

    # plotsFileName = "./plots/mnist+"  # Filename to save plots. Three plots are updated with each epoch; Accuracy, Loss and Error Rate
    # csvFileName = "./stats/mnist_log.csv"  # Filename to save training log. Updated with each epoch, contains Accuracy, Loss and Error Rate

    print(model)

    train_model(
        model,
        optimizer_s,
        lrate,
        num_epochs,
        args.reg,
        train_dl,
        test_dl,
        train_num,
        test_num,
        # plotsFileName,
        # csvFileName,
    )
