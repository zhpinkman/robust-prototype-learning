import torch
import wandb

import os
import torch
import torch
import torch.nn.functional as F
import pandas as pd
import json

from tqdm import tqdm

# from data_utils import *
from Models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomNonBinaryClassDataset(torch.utils.data.Dataset):
    def __init__(
        self, sentences, labels, tokenizer, max_length: int, dataset_type: str
    ):
        try:
            if dataset_type == "classification":
                sentences = [str(i) for i in sentences]
                inputs = tokenizer(
                    sentences,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
            elif dataset_type == "nli":
                sentences1 = [str(i) for i in sentences[0]]
                sentences2 = [str(i) for i in sentences[1]]

                inputs = tokenizer(
                    sentences1,
                    sentences2,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
        except Exception as e:
            print(e)
            from IPython import embed

            embed()
            exit()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        self.x = input_ids
        self.attn_mask = attention_mask
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.attn_mask[idx], self.y[idx]

    def collate_fn(self, batch):
        return (
            torch.LongTensor([i[0] for i in batch]),
            torch.Tensor([i[1] for i in batch]),
            torch.LongTensor([i[2] for i in batch]),
        )


def load_classification_dataset(df, tokenizer, dataset_type, max_length):
    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()

    dataset = CustomNonBinaryClassDataset(
        sentences=sentences,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_type=dataset_type,
    )

    return dataset


def load_dataset(data_dir, tokenizer, dataset_type="classification", max_length=512):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    # val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_df_split_0 = train_df[train_df["label"] == 0].sample(1000)
    train_df_split_1 = train_df[train_df["label"] == 1].sample(1000)

    train_df = pd.concat([train_df_split_0, train_df_split_1])

    # shuffle train dataframe
    # train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if dataset_type == "classification":
        return [
            load_classification_dataset(df, tokenizer, dataset_type, max_length)
            for df in [
                train_df,
                # val_df,
                test_df,
            ]
        ]
    else:
        raise Exception("Dataset type not supported")


def lr_scheduler(optimizer, init_lr, epoch):
    for param_group in optimizer.param_groups:
        if epoch == 20 or epoch == 25:
            param_group["lr"] = param_group["lr"] / 10

        if epoch == 0:
            param_group["lr"] = init_lr

        print("Current learning rate is {}".format(param_group["lr"]))

    return optimizer


def train_model(
    model,
    optimizer_s,
    lrate,
    num_epochs,
    reg,
    lambd,
    train_loader,
    test_loader,
    dataset_train_len,
    dataset_test_len,
):
    epochs = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    best_acc = 0.0

    begin_epoch = False

    for epoch in range(num_epochs):
        begin_epoch = True
        model.train()
        epochs.append(epoch)
        optimizer = lr_scheduler(optimizer_s, lrate, epoch)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("*" * 70)
        running_loss = 0.0
        running_corrects = 0.0
        train_batch_ctr = 0.0

        for input_ids, attention_mask, label in tqdm(
            train_loader, leave=False, total=len(train_loader)
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            features, centers, distance, outputs = model(
                input_ids, attention_mask, begin_epoch
            )

            _, preds = torch.max(distance, 1)

            loss1 = F.nll_loss(outputs, label)
            loss2 = regularization(features, centers, label)
            loss3 = centers_divergence_loss(centers)

            loss = loss1 + reg * loss2 + lambd * loss3
            if begin_epoch:
                embed()
            begin_epoch = False

            loss.backward()

            optimizer.step()
            train_batch_ctr += 1.0

            running_loss += loss.item()

            running_corrects += torch.sum(preds == label.data)

            epoch_acc = float(running_corrects) / (float(dataset_train_len))

        print(
            "Train corrects: {} Train samples: {} Train accuracy: {}".format(
                running_corrects, (dataset_train_len), epoch_acc
            )
        )
        train_acc.append(epoch_acc)
        train_loss.append(1.0 * running_loss / train_batch_ctr)

        print(
            "Train loss: {}".format(
                train_loss[epoch],
            )
        )

        # model.eval()
        # test_running_corrects = 0.0
        # test_batch_ctr = 0.0
        # test_running_loss = 0.0

        # for input_ids, attention_mask, label in tqdm(
        #     test_loader, leave=False, total=len(test_loader)
        # ):
        #     with torch.no_grad():
        #         input_ids = input_ids.to(device)
        #         attention_mask = attention_mask.to(device)
        #         label = label.to(device)

        #         features, centers, distance, test_outputs = model(
        #             input_ids, attention_mask
        #         )
        #         _, predicted_test = torch.max(distance, 1)

        #         loss1 = F.nll_loss(test_outputs, label)
        #         loss2 = regularization(features, centers, label)
        #         loss3 = centers_divergence_loss(centers)

        #         loss = loss1 + reg * loss2 + lambd * loss3

        #         test_running_loss += loss.item()
        #         test_batch_ctr += 1

        #         test_running_corrects += torch.sum(predicted_test == label.data)
        #         test_epoch_acc = float(test_running_corrects) / float(dataset_test_len)

        # if test_epoch_acc > best_acc:
        #     torch.save(model, "best_model.pt")
        #     best_acc = test_epoch_acc

        # test_acc.append(test_epoch_acc)
        # test_loss.append(1.0 * test_running_loss / test_batch_ctr)
        wandb.log(
            {
                "Train Loss": train_loss[epoch],
                # "Test Loss": test_loss[epoch],
                "Train Accuracy": train_acc[epoch],
                # "Test Accuracy": test_acc[epoch],
            }
        )
        # print(
        #     "Test corrects: {} Test samples: {} Test accuracy {}".format(
        #         test_running_corrects, (dataset_test_len), test_epoch_acc
        #     )
        # )

        print("*" * 70)

    # torch.save(model, "final_model.pt")
