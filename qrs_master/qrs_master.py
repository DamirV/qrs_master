import argparse
import os
import numpy as np
import math
import itertools
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import torch
import data_loader

num_epochs = 100
num_classes = 1
batch_size = 10
learning_rate = 0.001
DATA_PATH = "D:\\Projects\\ecg_gan_experiments-master\\Dataset\\"
MODEL_STORE_PATH = "D:\\Projects\\ecg_gan_experiments-master\\Models\\"
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


class QrsMaster(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(61, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def tryToTrain():
    epochs = 10
    data = data_loader.QrsDataset()
    dataLoader = DataLoader(data, batch_size=15, shuffle=True)
    qrs_master = QrsMaster()
    optimizer = torch.optim.SGD(qrs_master.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
       for batch_idx, (data, target) in enumerate(dataLoader):
       data, target = Variable(data), Variable(target)


if __name__ == "main":



