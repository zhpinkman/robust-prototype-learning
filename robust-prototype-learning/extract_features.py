import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import math
import torch.utils.data as data_utils
import torch.nn.functional as F
from Models import *
from train_utils import *
import numpy as np
import glob
from transformers import AutoTokenizer
import os
import pickle


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="imdb", help="data directory")
    parser.add_argument("--use_max_length", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)

    args, _ = parser.parse_known_args()

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # dataset_info = DatasetInfo(
    #     data_dir=args.data_dir, use_max_length=args.use_max_length
    # )
    (
        train_dataset,
        #  val_dataset,
        test_dataset,
    ) = load_dataset(data_dir=args.data_dir, tokenizer=tokenizer)

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    # val_dl = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=val_dataset.collate_fn,
    # )
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )

    train_num = len(train_dataset)
    test_num = len(test_dataset)

    model = Net(2, 2, 2)

    def copy_data(m, i, o):
        my_embedding.copy_(o)

    model = torch.load("final_model.pt")
    model = model.cuda()
    print(model)
    layer = model._modules.get("fc3")
    extracted_features = []
    true_labels = []

    i = 0
    for (
        input_ids,
        attention_mask,
        label,
    ) in test_dl:
        i += 1
        print("working on {} input".format(i))

        my_embedding = torch.zeros(1, 2)

        def copy_data(m, i, o):
            my_embedding.copy_(o)

        h = layer.register_forward_hook(copy_data)
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        label = label.cuda()
        test_outputs = model(input_ids, attention_mask)
        h.remove()
        my_embedding = my_embedding.squeeze(0)
        my_embedding = my_embedding.detach().numpy()
        extracted_features.append(my_embedding)
        true_labels.append(label.cpu().data.numpy()[0])

        np.save("features", extracted_features)
        np.save("labels", true_labels)
