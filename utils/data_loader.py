import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class AtariGrandChallengeDataset(Dataset):
    """
    Atari Grand Challenge dataset
    www.atarigrandchallenge.com/data
    """
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): Path to the csv file with annotations
        root_dir (string): Directory with all the images
        transform        : Function hook to transform data
        """
    
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # TODO 
        sample = { 
            'observation' : obs, 
            'reward' : reward,
            'action' : action,
            'next_observation' : next_obs,
            'done_mask' : done
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    
    
    
    