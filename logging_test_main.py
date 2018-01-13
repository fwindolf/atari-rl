import gym

from agents.dqn import DQNAgent
from utils.screen import SpaceInvaderScreen
from utils.data_loader import AtariGrandChallengeDataset
from train.solver import Solver

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np

#%reload_ext autoreload
#%autoreload 2

#%matplotlib inline

screen = SpaceInvaderScreen()
frame = screen.reset()
plt.imshow(frame);

data_dir = '/mnt/c/Users/sc_to/Documents/IN2346_Deep_Leaning_Exercise/Final_Project/atari_data'
dataset = AtariGrandChallengeDataset(data_dir, 'spaceinvaders', history_len=10, screen=screen, max_files=2)
data_train, data_valid, data_test = dataset.split(0.7, 0.2,'OVERFIT')
print('Size of Training Data: %d' % len(data_train))

agent = DQNAgent(screen, mem_size=1000)
agent.initialize(800)

batchsize = 1

train_loader = DataLoader(data_train, batch_size=batchsize, num_workers=4)
val_loader = DataLoader(data_valid, batch_size=batchsize, num_workers=4)


logfile_path = '/mnt/c/Users/sc_to/Documents/IN2346_Deep_Leaning_Exercise/Final_Project/dl4cv/logfile.log'
console_log_level = 'DEBUG'
solver = Solver(optim.Adam, CrossEntropyLoss(), batchsize, logfile_path = logfile_path, log_level=console_log_level)
solver.train_offline(agent, train_loader, val_loader, num_epochs=50, log_nth=0)
#solver.train_online(agent, screen)
#solver.play(agent, screen)