import os
from os import path
import csv
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils.transition import Transition

class AtariGrandChallengeDataset(Dataset):
    """
    Atari Grand Challenge dataset
    www.atarigrandchallenge.com/data
    
    Uses lazy loading of screens
    Possibly (re)using memmaps would be more efficient
    """
    
    def __init__(self, root_dir, game, history_len, transform=None, screen=None, max_files=None):
        """        
        root_dir (string): Directory with all the images
        game (string)    : Name of the game that is played
        transform        : Function hook to transform data
        """
        self.root_dir = root_dir
        self.game = game
        self.history_len = history_len
        self.transform = transform
        self.data = list()      
        self.traj = dict()
        
        
        self.screen = screen # for transforming data to consistent format
        self.screens = dict() # cache screens
                
        self.action_meanings = self.screen.get_action_meaning()
        
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
        ds_train = AtariGrandChallengeDataset(self.root_dir, self.game, self.history_len, self.transform, self.screen, 0)
        ds_train.data = self.data[idx_train]
        ds_train.traj = self.traj
        
        ds_valid = AtariGrandChallengeDataset(self.root_dir, self.game, self.history_len, self.transform, self.screen, 0)
        ds_valid.data = self.data[idx_valid]
        ds_valid.traj = self.traj
        
        ds_test = AtariGrandChallengeDataset(self.root_dir, self.game, self.history_len, self.transform, self.screen, 0)
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
        # try to get from cache
        try: 
            return self.screens[idx]
        except:
             pass # doesnt matter if we didnt cache it
        
        screen_path = path.join(self.root_dir, 'screens', self.game, str(idx))
        try:
            self.screens[idx] = io.imread(path.join(screen_path, str(frame) + '.png'))             
            return self.screens[idx]
        except:
            return None # doesnt exist
        
    def __len__(self):
        return len(self.data)
    
    def __getdata(self, idx):
        data = self.data[idx]
        # lazily load screens
        p_idx, f_idx = self.traj[idx]
        frame = self.__load_screen(p_idx, f_idx)
        next_frame = self.__load_screen(p_idx, f_idx + 1)
        
        return Transition(frame, data['action'], data['reward'], data['done'], next_frame)
    
    def raw(self, idx):
        """
        Return the raw data at this index as Transition
        idx (in)            : The data index
        return (Transition) : Consists of frame, action, reward, done, next_frame
        """
        return self.__getdata(idx)
    
    def __getsequence(self, idx):        
        sample = self.__getdata(idx)
        
        if self.transform:
            sample = self.transform(sample)
        
        # unwrap
        frame = sample.observation
        action = sample.action
        reward = sample.reward
        done = sample.done
        next_frame = sample.next_observation
        
        # get the action codes as 0-n like from the environment
        code = self.screen.get_action_code(action)
        action = self.action_meanings.index(code) 
        
        # create new array for our observation
        observation = np.empty([self.history_len] + list(frame.shape), dtype='uint8')
        play_idx, frame_idx = self.traj[idx] # the play the frame belongs to        
        
        # fill the observation
        hlen = self.history_len - 1
        missing_frames = max(0, hlen - frame_idx)
        for o_idx in range(0, missing_frames):            
            observation[o_idx] = np.zeros_like(frame, dtype='uint8')
                        
        for o_idx in range(missing_frames, hlen):    
            observation[o_idx] = self.__load_screen(play_idx, frame_idx - hlen + o_idx)
            
        observation[hlen] = frame
        
        # create new array for next_observation
        next_observation = np.empty(observation.shape)
        for idx in range(len(observation)- 1):
            next_observation[idx] = observation[idx + 1]
        
        next_observation[hlen] = next_frame        
        
        return observation, action, reward, done, next_observation
    
    def sequence(self, idx):
        """
        Get the raw data for a sequence of history_len frames, and the associated action, reward, 
        done and a next_frame
        idx (int)      : The data index
        return (tuple) : A tuple consiting of observation (frames * history_len), action, 
                         reward, done, and next_frame
        """
        return self.__getsequence(idx)
    
    def __getitem__(self, idx):               
        """
        Function used by the dataloader
        Get a sequence of history_len frames, and the associated action, reward and next_frame
        with the observation and next_frames encoded like in screen
        
        return (tuple): observation (history_len * frame.shape), action
        """        
        observation, action, reward, done, next_observation = self.__getsequence(idx)
        
        obs = np.empty([len(observation)] + list(self.screen.get_shape()[1:]), dtype='float')
        for o_idx in range(len(observation)):
            obs[o_idx] = self.screen.output_float(observation[o_idx])
        
        return obs, action
        
        
    
        
        