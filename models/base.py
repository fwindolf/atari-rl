import torch
import torch.nn as nn

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
        print('Saving model... %s' % path)
        torch.save(self.model, path) 