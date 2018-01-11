from models.base import ModelBase

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(ModelBase):
    """
    DQN convolutional network that produces Q-values
    
    num_inputs (int)  : The number of channels of the input frame array
    num_actions (int) : The number of actions in action_space 
    """
    def __init__(self, num_inputs, num_actions):
        super().__init__(num_inputs, num_actions)
        
        # TODO: This only works for certain input resolutions! (80,80)

        # Network definition
        self.conv1 = nn.Conv2d(num_inputs, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)        
        self.fc4 = nn.Linear(7 * 7 * 32, 512)
        self.fc5 = nn.Linear(512, num_actions)
        
        if self.is_cuda:
            self.cuda()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
    
