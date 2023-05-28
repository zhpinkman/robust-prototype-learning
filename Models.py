from __future__ import print_function
import argparse
from collections import defaultdict
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm
from transformers import RobertaModel


class Net(nn.Module):
    def __init__(self, num_hidden_units=2, num_classes=2, s=2, train_dataset=None):
        super(Net, self).__init__()

        self.encoder = RobertaModel.from_pretrained("xlm-roberta-base")

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.dropout_3 = nn.Dropout(0.1)

        self.batchnorm1_1 = nn.BatchNorm1d(256)
        self.batchnorm1_2 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(768, 256)
        self.prelu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.prelu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_hidden_units)
        # self.prelu3 = nn.PReLU()

        self.scale = s

        self.dce = dce_loss(num_classes, num_hidden_units)
        self.init_dce_loss(train_dataset)

    def init_dce_loss(self, train_dataset):
        train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )

        train_dl_bert_features = defaultdict(list)

        model = self.eval()
        model = model.cuda()
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(
                train_dl, total=len(train_dl)
            ):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                x1, _, _, _ = model(input_ids, attention_mask)
                x1 = x1.cpu()
                for one_bert_feature, label in zip(x1, labels):
                    train_dl_bert_features[label.item()].append(one_bert_feature)

        model = model.cpu()

        train_dl_bert_features_mean = {
            key: torch.stack(train_dl_bert_features[key]).mean(dim=0)
            for key in train_dl_bert_features.keys()
        }
        train_dl_bert_features_mean = torch.stack(
            [train_dl_bert_features_mean[0], train_dl_bert_features_mean[1]]
        ).T

        self.dce.centers.data = train_dl_bert_features_mean

    def forward(self, input_ids, attention_mask, begin_epoch=False):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[1]
        x = self.dropout_1(x)

        x = self.fc1(x)
        x = self.batchnorm1_1(x)
        x = self.prelu1(x)

        x = self.fc2(x)
        x = self.batchnorm1_2(x)
        x = self.prelu2(x)

        x1 = self.fc3(x)

        centers, x = self.dce(x1)

        output = F.log_softmax(self.scale * x, dim=1)

        if begin_epoch:
            embed()

        return x1, centers, x, output


class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim, init_weight=True):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(
            torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True
        )
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = features - torch.t(centers)[labels]

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance


def centers_divergence_loss(
    centers,
    threshold: float = 0.02,
):
    distance = centers.t()[0] - centers.t()[1]
    distance = torch.sum(torch.pow(distance, 2))
    loss = torch.max(torch.Tensor([distance - threshold, 0]).cuda())
    return loss
