import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.autograd import Variable

class AgentBase:
    def __init__(self, screen):     
        self.screen = screen
        self.actions = None
        self.memory = None
        self.model = None
        
    def initialize(self):
        """
        Initialize the replay buffer and model
        """
        raise NotImplemented()
        
    def play(self, screen, num_sequences):
        """
        Play in the environment to achieve the highest score
        """
        raise NotImplemented()             
    
    def next_action(self, observation):
        """
        Select the next action via epsilon-greedy exploration
        
        observation (frame) : The current state of the screen        
        return (int)        : The best action calculated by the agent
        """        
        raise NotImplemented()
        
        