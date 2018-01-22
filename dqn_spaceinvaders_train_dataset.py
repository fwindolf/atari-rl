import logging
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from agents.dqn import DQNAgent
from utils.screen import SpaceInvaderScreen
from utils.data_loader import AtariGrandChallengeDataset
from train.solver import Solver


HISTORY_LEN = 4
BATCHSIZE = 32
MEMORY_SIZE = 1000000
INITIAL_SIZE = 500
INITIAL_LEN = 1
EPOCHS = 10000
DATA_PATH = '/mnt/e/data/atari_v2_release'
GAME = 'spaceinvaders'

logger = logging.getLogger("Main")
logger.setLevel(20) # INFO

logger.info("Training for DQN Base")
screen = SpaceInvaderScreen()

logger.info("Opening dataset")
dataset = AtariGrandChallengeDataset(DATA_PATH, GAME, history_len=HISTORY_LEN, screen=screen)

logger.info("Creating Agent")
agent = DQNAgent(screen, history_len=HISTORY_LEN, mem_size=MEMORY_SIZE)
logger.info("GPU available: %r" % torch.cuda.is_available())
logger.info("Model on GPU: %r" % agent.model.is_cuda)

data_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
solver = Solver(optim.Adam, CrossEntropyLoss(), batchsize=BATCHSIZE, log_level='INFO')

logger.info("Creating Baseline scores")
best_score, mean_score, mean_dur = solver.play(agent, screen, num_sequences=INITIAL_LEN)
logger.info("Baseline with best score %d (Mean %d in %d frames)" % (best_score, mean_score, mean_dur))

logger.info("Starting Training ")
solver.train_dataset(agent, screen, data_loader, num_epochs=EPOCHS, learning_rate=0.00025)

logger.info("Saving Model")
agent.model.save('/models/dqn_base.pt')
self.logger.info("Done!")
