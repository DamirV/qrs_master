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
        super(QrsMaster, self).__init__()
        self.fc1 = nn.Linear(61, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def tryToTrain():
    epochs = 10
    batch_size = 15
    learning_rate = 0.01
    dataLoader = data_loader.QrsDataset()
    net = QrsMaster()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch_idx in range(batch_size):
            data, target = dataLoader.__getitem__(batch_idx)

            data = np.array(data)
            target = np.array(target)

            data = torch.from_numpy(data)
            target = torch.from_numpy(target)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            print("epoch = " + epoch + "/" + epochs)
            print("output = " + output + " ,target = " + target + " ,loss = " + loss)

    return net


if __name__ == "__main__":
    a = tryToTrain()
