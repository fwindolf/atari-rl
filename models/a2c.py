from models.base import ModelBase

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

# adapted from https://github.com/lnpalmer/A2C/blob/master/models.py

class A2C(ModelBase):
    def __init__(self, in_channels, num_actions):
        super().__init__(in_channels, num_actions)

        self.num_actions = num_actions

        # base network with 3 conv layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,          out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,          out_channels=64, kernel_size=3, stride=1)

        # Shared FC Layer
        self.fc = nn.Linear(in_features=6*6*64, out_features=512)

        # Policy head
        self.fc_pi = nn.Linear(in_features=512, out_features=num_actions)

        # Value head
        self.fc_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

        self.fc_pi.weight.data = self.ortho_weights(self.fc_pi.weight.size(), scale=0.1)
        self.fc_val.weight.data = self.ortho_weights(self.fc_val.weight.size())

        if self.is_cuda():
            self.cuda()

    def forward(self, x):

        batch_size = x.size(0)

        # base network with 3 conv layer
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        # Shared FC Layer
        x = self.relu(self.fc(x))

        # Policy head
        pi = self.relu(self.fc_pi(x))

        # Value head
        val = self.relu(self.fc_val(x))

        return pi, val

    def ortho_weights(self, shape, scale=1.):
        """ PyTorch port of ortho_init from baselines.a2c.utils """

        shape = tuple(shape)

        if len(shape) == 2:
            flat_shape = shape[1], shape[0]
        elif len(shape) == 4:
            flat_shape = (np.prod(shape[1:]), shape[0])
        else:
            raise NotImplementedError

        a = np.random.normal(0., 1., flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.transpose().copy().reshape(shape)

        if len(shape) == 2:
            return torch.from_numpy((scale * q).astype(np.float32))

        if len(shape) == 4:
            return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))

        return None