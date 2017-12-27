from agents.base import AgentBase
from utils.replay_buffer import ReplayBuffer
from models.dqn import DQN

import numpy as np

class DQNAgent(AgentBase):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        
        self.actions = action_space
        self.memory = ReplayBuffer(size=1000000, history_len=1000)
        self.model = DQN(num_inputs, self.action_space.n)
        
    