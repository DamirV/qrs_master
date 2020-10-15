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
    epochs = 3
    learning_rate = 0.01
    dataloader = data_loader.QrsDataset()
    dataloader = DataLoader(dataloader, batch_size=4)
    net = QrsMaster()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        for i, (data, target) in enumerate(dataloader):

            target = target.view(1, -1)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                %(epoch + 1, epochs, i + 1, len(dataloader), loss.item()))

    return net


if __name__ == "__main__":
    a = tryToTrain()
