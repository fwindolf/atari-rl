import logging
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.utils.data import DataLoader

from torchvision import transforms, utils

from agents.dqn import DQNAgent
from utils.screen import CartPoleScreen
from train.solver import Solver

HISTORY_LEN = 4
BATCHSIZE = 32
INITIAL_LEN = 1
EPOCHS = 1000
DATA_PATH = '/mnt/e/data/atari_v2_release'
GAME = 'cartpole'

logger = logging.getLogger("Main")
logger.setLevel(20) # INFO

logger.info("Training for DQN Base")
screen = CartPoleScreen()

logger.info("Creating Agent")
agent = DQNAgent(screen, history_len=HISTORY_LEN, mem_size=0, loss=SmoothL1Loss())
logger.info("GPU available: %r" % torch.cuda.is_available())
logger.info("Model on GPU: %r" % agent.model.is_cuda)

solver = Solver(optim.RMSprop, None, batchsize=BATCHSIZE, log_level='INFO') # Loss doesnt do anything for dataset learning

logger.info("Creating Baseline scores")
best_score, mean_score, mean_dur = solver.play(agent, screen, num_sequences=INITIAL_LEN)
logger.info("Baseline with best score %d (Mean %d in %d frames)" % (best_score, mean_score, mean_dur))

logger.info("Starting Training ")
solver.train_online(agent, screen, num_epochs=EPOCHS, learning_rate=2e-7)

logger.info("Creating Benchmark scores")
best_score, mean_score, mean_dur = solver.play(agent, screen, num_sequences=INITIAL_LEN)
logger.info("Benchmark with best score %d (Mean %d in %d frames)" % (best_score, mean_score, mean_dur))

logger.info("Saving Model")
agent.model.save('/output/dqn_base_cartpole.pt')
logger.info("Done!")
