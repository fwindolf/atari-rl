import os
from os import path
import torch
import panda as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils

class AtariGrandChallengeDataset(Dataset):
    """
    Atari Grand Challenge dataset
    www.atarigrandchallenge.com/data
    """
    
    def __init__(self, root_dir, game, transform=None):
        """        
        root_dir (string): Directory with all the images
        game (string)    : Name of the game that is played
        transform        : Function hook to transform data
        """
        self.root_dir = root_dir
        self.game = game
        self.transform = transform
        self.data = list()      
        self.traj = dict()
        
    def __load_trajectories(self, max_files=None):
        
        traj_path = path.join(self.root_dir, 'trajectories')
        game_path = path.join(traj_path, self.game)
        
        idx = 0 # running index for data
        
        files = os.listdir(game_path)        
        if max_files is not None:
            files = files[:max_files]
        
        for filename in files:
            if filename.endswith('.txt'):
                # Create data for each of the <idx>.txt files
                idx = int(filename.split('.')[0])
                
                # Save the lines of the txt files
                with open(path.join(game_path, filename)) as f:
                data = csv.reader(f, delimiter=',')
                for rows in data:
                    # frame,reward,score,terminal,action 
                    frame = int(row[0])
                    reward = int(row[1])
                    score = int(row[2])
                    done = bool(row[3])
                    action = int(row[4]) # or __get_action_name?
                                        
                    self.data.append(dict(
                        frame=frame,
                        reward=reward, 
                        score=score,
                        done=done,
                        action=action))
                    
                    data_idx = len(self.data) - 1
                    self.traj[data_idx] = (idx, frame) # save to be able to load screen later
                
    def __load_screen(self, idx, frame):
        screen_path = path.join(self.root_dir, 'screens', self.game, idx)
        try:
            return io.imread(path.join(screen_path, str(frame) + '.png'))
        except:
            return None        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):               
        data = self.data[idx]
        
        # observation, reward, action, next_observation, done
        # get the screens        
        f_idx, frame = self.traj[idx]
        obs = self.__load_screen(f_idx, frame)
        next_obs = self.__load_screen(f_idx, frame + 1)
        
        if next_obs is None and data.done is False:
            raise AssertionError('Unfinished sequence has no next observation!')

        sample = dict(
            observation=obs,
            reward=data.reward, 
            done=data.done,
            next_observation=next_obs)            
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    
    
    
    