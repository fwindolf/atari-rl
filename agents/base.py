import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.autograd import Variable
from torchvision import models

class AgentBase:
    def __init__(self):       
        self.actions = None
        self.memory = None
        self.model = None
    
    def next_action(self, observation):
        """
        Select the next action via epsilon-greedy exploration
        
        observation (frame) : The current state of the screen        
        return (int)        : The best action calculated by the agent
        """        
        if np.random.rand() < self.eps:
            return self.actions.sample() # random action from action space
        else:
            return self.model.predict(Variable(observation, volatile=True)).data.max(1).cpu()
        