#!/user/bw2762/.conda/envs/testbed_2/bin/python

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x = x.view(-1, 28*28)
        # print(x.shape)
        # print(self.fc1.weight.dtype)
        # print(x.dtype)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class MLP_experiment():

    def __init__(self,in_dim,out_dim,first_batch):

        self.model = MLP(in_dim,out_dim)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.step(first_batch.x,first_batch.y)

    def step(self,features,labels):
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).float()
        self.optimizer.zero_grad()
        outputs = self.model(features)
        # print(outputs.shape)
        # print(labels.dtype)
        # print(outputs.dtype)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

    def get_params(self):
        parameters = [p.data for p in self.model.parameters()]
        flat_parameters = torch.cat([p.view(-1) for p in parameters])
        return flat_parameters

    def predict(self,x):
        x = torch.from_numpy(x).float()
        return self.model.forward(x)