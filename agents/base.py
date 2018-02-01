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
        
    def initialize(self, screen, num_frames=0):
        """
        Initialize the replay buffer and model
        """
        pass # not necessarily needed
     
    def step(self, screen, save=False, random=False, render=False):
        """
        Step once in the environment
        """
        raise NotImplemented()
        
    def play(self, screen, max_duration=10000, save=False):
        """
        Play in the environment to achieve the highest score
        """
        raise NotImplemented()      
        
    def optimize(self, optimizer, screen, batchsize, data=None):
        """
        Optimize the model with either data or from stepping
        """
        raise NotImplemented()
    
    def next_action(self, observation, random=False):
        """
        Select the next action
        
        observation (frame) : The current state of the screen        
        random (bool)       : Should a random action be used
        return (int)        : The best action calculated by the agent
        """        
        raise NotImplemented()
        
        