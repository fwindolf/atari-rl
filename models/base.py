import torch
import torch.nn as nn

import os

class ModelBase(nn.Module):
    """
    Base module for models 
    """
    def __init__(self, num_inputs, num_actions):
        """
        Create an new Modelbase (non-functional)
        
        num_inputs (int)  : The size of the input array
        num_actions (int) : The number of actions the agent can take
        """
        super(ModelBase, self).__init__()
        
        self.inputs = num_inputs
        self.actions = num_actions
        
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda
       
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        if path[0] == '/':
            path = path[1:] # remove leading / because we want to write to subfolder
            
        directory = os.path.join(os.getcwd(), path[:path.rfind('/')]) 
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self, path) 
        
    def predict(self, observation):
        """
        Predict the next action
        frame (observation) : The observation for which a new action is required
        return (action)     : The predicted action
        """
        output = self.forward(observation)
        _, prediction = torch.max(output, 1, keepdim=True)
        return prediction
    
    