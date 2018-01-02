import os
from os import path
import csv
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

class AtariGrandChallengeDataset(Dataset):
    """
    Atari Grand Challenge dataset
    www.atarigrandchallenge.com/data
    
    Uses lazy loading of screens
    Possibly (re)using memmaps would be more efficient
    """
    
    def __init__(self, root_dir, game, transform=None, max_files=None):
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
        
        self.__load_trajectories(max_files)
        
    def split(self, train, valid):
        """
        Split the dataset into training, validation and test data
        
        train (int): Percentage of training data
        valid (int): Percentage of validation data
        """
        assert(train + valid < 1)
        
        indices = np.random.permutation(len(self))
        num_train = int(train * len(self))
        num_valid = int(valid * len(self))
        idx_train = indices[:num_train]
        idx_valid = indices[num_train:num_train + num_valid]
        idx_test = indices[num_train + num_valid:]
      
        # make new datasets out of the splits
        ds_train = AtariGrandChallengeDataset(self.root_dir, self.game, self.transform, 0)
        ds_train.data = self.data[idx_train]
        ds_train.traj = self.traj
        
        ds_valid = AtariGrandChallengeDataset(self.root_dir, self.game, self.transform, 0)
        ds_valid.data = self.data[idx_valid]
        ds_valid.traj = self.traj
        
        ds_test = AtariGrandChallengeDataset(self.root_dir, self.game, self.transform, 0)
        ds_test.data = self.data[idx_test]
        ds_test.traj = self.traj
        
        return ds_train, ds_valid, ds_test
    
        
        
    def __load_trajectories(self, max_files):
        """
        Load the trajectories from disk
        Trajectories are organized in .txt files that contain associated frames, rewards, ... in
        a csv-like structure
        """
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
                    next(data) # skip db id
                    next(data) # skip header
                    for row in data:
                        # frame,reward,score,terminal,action 
                        frame = int(row[0])
                        reward = int(row[1])
                        score = int(row[2])
                        done = bool(int(row[3]))
                        action = int(row[4]) # or __get_action_name?

                        self.data.append(dict(
                            frame=frame,
                            reward=reward, 
                            score=score,
                            done=done,
                            action=action))

                        data_idx = len(self.data) - 1
                        self.traj[data_idx] = (idx, frame) # save to be able to load screen later
         
        self.data = np.array(self.data)
                
    def __load_screen(self, idx, frame):
        """
        Load the associated screen for the frame in <idx>.txt
        """
        screen_path = path.join(self.root_dir, 'screens', self.game, str(idx))
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
            reward=data['reward'], 
            done=data['done'],
            action=data['action'],
            next_observation=next_obs)            
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample