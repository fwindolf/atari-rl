from models.base import ModelBase

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

# adapted from https://github.com/dxyang/DQN_pytorch/blob/master/model.py

class Dueling_DQN(ModelBase):
    def __init__(self, in_channels, num_actions):
        super().__init__(in_channels, num_actions)

        self.num_actions = num_actions

        # base network with 3 conv layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,          out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,          out_channels=64, kernel_size=3, stride=1)

        # Advantage head
        self.fc1_adv = nn.Linear(in_features=6 * 6 * 64, out_features=512)
        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)

        # Value head
        self.fc1_val = nn.Linear(in_features=6 * 6 * 64, out_features=512)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

        if self.is_cuda():
            self.cuda()


    def forward(self, x):

        # base network with 3 conv layer
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        # Advantage head
        adv = self.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)

        # Value head
        val = self.relu(self.fc1_val(x))
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        # Combination to Q values
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)

        return x