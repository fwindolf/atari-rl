from os import path, listdir

import agc.dataset as dataset
import agc.util as util
import numpy as np
from skimage import io 

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class AGCDataSet(Dataset):
    """
    Atari Grand Challenge dataset
    www.atarigrandchallenge.com/data
    """    
    def __init__(self, datadir, game, history_len, gamma=0.8, transform=None, screen=None, reuse=False):
        self.data_set = dataset.AtariDataset(datadir)
        if game in ['spaceinvaders', 'mspacman', 'pinball']:
            self.game=game
        else:
            print('Available games are spaceinvaders, mspacman, and pinball')
        
        self.history_len = history_len
        self.reuse = reuse
        self.screen = screen
        self.action_meanings = self.screen.get_action_meaning() # mapping from idx to string
        
        self.transform = transform
        self.screens_dir = path.join(datadir, 'screens', self.game)
        
        if self.reuse:
            self.screens = dict()
            self.screens[self.game] = dict()
            
        self.valid_trajectories= listdir(path.join(datadir,'screens', self.game))    
        
        self.__discount_rewards(self.game, gamma)
        
        self.trajectory = 0
        self.next() # advance to random trajectory
    
    def __len__(self):
        return len(self.data_set.trajectories[self.game][self.trajectory])
    
    def __get_screen(self, game, trajectory, idx):
        
        if self.reuse and idx in self.screens[game][trajectory]:
            return self.screens[game][trajectory][idx]
        
        img_name = path.join(self.screens_dir, str(trajectory), str(idx) + '.png')
        
        # if this screen does not exist just return None
        try:
            image = io.imread(img_name)        
        except:
            return None
        
        if self.screen is not None:
            image = self.screen.output_float(image) # same format as from env
        
        # save in case of re-usage
        if self.reuse:
            self.screens[game][trajectory][idx] = image
        
        return image
    
    def __get_observations(self, game, trajectory, idx):        
        
        # 1. observation
        frame = self.__get_screen(game,trajectory, idx)
        if frame is None:
            # we cant prevent from breaking anyways
            print(game, trajectory, idx)
        
        # get history_len frames before and at idx
        observation = np.zeros([self.history_len] + list(frame.shape), dtype=np.float32)
        hlen = self.history_len - 1
        
        missing_frames = max(0, hlen - idx)
        # missing frames stay 0, else fill with screens
        for o_idx in range(missing_frames, hlen):    
            observation[o_idx] = self.__get_screen(game, trajectory, idx - hlen + o_idx)
            
        observation[hlen] = frame
        
        # 2. next_observation
        next_frame = self.__get_screen(game, trajectory, idx + 1)
        
        # get history_len frames before and from idx + 1
        next_observation = np.zeros_like(observation, dtype=np.float32)
        
        if next_frame is not None:
            next_observation = np.roll(observation, -1) # shift everything one to back
            next_observation[-1] = next_frame
        
        return observation, next_observation
    
    def __getitem__(self,idx):        
        # get data for frame at idx
        data = self.data_set.trajectories[self.game][self.trajectory][idx] 
        obs, next_obs = self.__get_observations(self.game, self.trajectory, idx)
        
        # resolve action into [0, ...] and not as indexing the actions array 
        code = self.screen.get_action_code(data['action'])
        try:
            action = self.action_meanings.index(code) 
        except ValueError:
            # invalid action translates to noop in game
            action = 0        
        
        sample = (obs, action, data['reward'], next_obs)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __discount_rewards(self, game, gamma):
        """
        Iterate over all frames and discount the reward so learning is faster.
        Modifies all trajectories in memory.
        """ 
        for trajectory in self.valid_trajectories:            
            data = self.data_set.trajectories[game][int(trajectory)]
            
            next_reward = 0
            # reverse through the frames and discount
            for i in reversed(range(len(data))):
                reward = data[i]['reward'] + gamma * next_reward
                data[i]['reward'] = reward
                next_reward = reward
    
    def next(self):
        """
        Switch to a random trajectory folder
        """
        # delete screens dict to save memory
        if self.reuse:
            del self.screens[self.game][self.trajectory]
            
        self.trajectory = int(np.random.choice(self.valid_trajectories))
        
        # create new screens dict
        if self.reuse:
            self.screens[self.game][self.trajectory] = dict() 
        
            
    def raw(self, idx):
        raise NotImplemented()
        
    def sequence(self, idx):
        raise NotImplemented()
        
        
        
        