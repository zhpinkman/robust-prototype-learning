from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from transformers import BertModel


class Net(nn.Module):
    def __init__(self, num_hidden_units=2, num_classes=2, s=2):
        super(Net, self).__init__()

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        self.dropout = nn.Dropout(0.1)

        # Write down some linear layers starting from the output of the bert model and after two layers getting to the num_hidden_units
        self.fc1 = nn.Linear(768, 256)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(256, 128)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(128, num_hidden_units)

        self.scale = s

        # self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # self.prelu1_1 = nn.PReLU()
        # self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        # self.prelu1_2 = nn.PReLU()
        # self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # self.prelu2_1 = nn.PReLU()
        # self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.prelu2_2 = nn.PReLU()
        # self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        # self.prelu3_1 = nn.PReLU()
        # self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        # self.prelu3_2 = nn.PReLU()
        # self.preluip1 = nn.PReLU()

        # self.ip1 = nn.Linear(128 * 3 * 3, num_hidden_units)
        self.dce = dce_loss(num_classes, num_hidden_units)

    def forward(self, input_ids, attention_mask):
        x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
        x = self.dropout(x)
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        x1 = self.fc3(x)

        # x = self.prelu1_1(self.conv1_1(x))
        # x = self.prelu1_2(self.conv1_2(x))
        # x = F.max_pool2d(x, 2)
        # x = self.prelu2_1(self.conv2_1(x))
        # x = self.prelu2_2(self.conv2_2(x))
        # x = F.max_pool2d(x, 2)
        # x = self.prelu3_1(self.conv3_1(x))
        # x = self.prelu3_2(self.conv3_2(x))
        # x = F.max_pool2d(x, 2)
        # x = x.view(-1, 128 * 3 * 3)

        # x1 = self.preluip1(self.fc3(x))
        centers, x = self.dce(x1)
        output = F.log_softmax(self.scale * x, dim=1)
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
