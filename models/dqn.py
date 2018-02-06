from models.base import ModelBase

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from models.caps.capsule import Capsule

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
        self.dp1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.dp2 = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.dp3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(7 * 7 * 32, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        if self.is_cuda:
            x= x.cuda()
            
        x = F.relu(self.dp1(self.bn1(self.conv1(x))))
        x = F.relu(self.dp2(self.bn2(self.conv2(x))))
        x = F.relu(self.dp3(self.bn3(self.conv3(x))))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

    
class DQNLinear(ModelBase):
    """
    DQN linear network that produces Q-values

    num_inputs (int)  : The number of channels of the input frame array
    num_actions (int) : The number of actions in action_space
    hidden_size (int) : The number of neurons in the hidden layer
    """
    def __init__(self, num_inputs, num_actions, hidden_size):        
        num_inputs, num_actions = int(num_inputs), int(num_actions)        
        super().__init__(num_inputs, num_actions)
        self.l1 = nn.Linear(num_inputs, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        if self.is_cuda:
            x = x.cuda()
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class DQNCapsNet(ModelBase):
    """
    DQN Capsule network with 3 layers that produces Q-values

    num_inputs (int)  : The number of channels of the input frame array
    num_actions (int) : The number of actions in action_space
    conv_output (int) : The number of outputs from the convolutional network
    conv_kernel (int) : The kernel size of the convolutional network
    conv_stride (int) : The stride of the convolutional network
    primary_num (int) : The number of primary capsules
    primary_size (int): The unit size of the primary capsules
    num_routing (int) : The number of routing iterations per epoch
    """
    def __init__(self, num_input, num_actions, conv_output=64, conv_kernel=9, conv_stride=1, 
                 primary_num=16, primary_size=32768, num_routing=2):

        super().__init__(num_input, num_actions)

        # Layer 1 : Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=num_input, out_channels=conv_output,
                               padding=0, kernel_size=conv_kernel, stride=conv_stride)
        self.relu = nn.ReLU(inplace=True)

        # Primary Layer
        # Conv2d with squash activation
        self.primary = Capsule(in_unit=0, in_channel = conv_output, num_unit = primary_num,                               
                               unit_size = primary_size, use_routing=False, num_routing=num_routing)

        # Output layer
        # Capsule layer with dynamic routing
        self.output = Capsule(in_unit = primary_num, in_channel = primary_size, num_unit=1,
                              unit_size = num_actions, use_routing=True, num_routing=num_routing)

    def forward(self, x):
        if self.is_cuda:
            x = x.cuda()        
        # Forward Pass
        conv1 = self.conv1(x)
        conv1 = self.relu(conv1)

        primary_caps_out = self.primary(conv1)
        output = self.output(primary_caps_out)
        
        output = output.squeeze().unsqueeze(0)

        return output

    
    
        
    