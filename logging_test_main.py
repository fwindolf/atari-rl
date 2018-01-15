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
from math import sqrt, ceil


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    # determine grid size
    (N, C, H, W) = Xs.shape  # 4D shape
    Xs = Xs.reshape(N * C, H, W)  # reshape to 3D
    (N, H, W) = Xs.shape  # eD shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def show_net_weights(net):
    fig = plt.figure(figsize=(20, 20))
    cnn_weight1 = net.state_dict()['conv1.weight'].numpy()

    plt.imshow(visualize_grid(cnn_weight1))
    plt.gca().axis('off')
    plt.show()
    fig.savefig('cnn1_weights.pdf')



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
solver.train_offline(agent, train_loader, val_loader, num_epochs=100, log_nth=0)
#solver.train_online(agent, screen)
#solver.play(agent, screen)

# show weights and save to file

# get wright names/dict
# print(agent.model.state_dict().keys())
show_net_weights(agent.model)