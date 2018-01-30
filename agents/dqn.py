"""Agents based on deep Q-networks as proposed by Deepmind for atari games."""

from agents.base import AgentBase
from models.dqn import DQN
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utils.replay_buffer import ReplayBuffer


class DQNAgent(AgentBase):
    """A DQN agent implementation according to the DeepMind paper."""

    def __init__(self, screen, history_len=10, gamma=0.8, loss=CrossEntropyLoss(), memory=None, model=None):
        """Initialize agent."""
        super().__init__(screen)
        self.history_len = history_len
        if memory is None:
            self.memory = ReplayBuffer(100000, history_len, screen)
        else:
            self.memory = memory

        if model is None:
            self.model = DQN(history_len, screen.get_actions())
        else:
            self.model = model            
        
        if torch.cuda.is_available():
            self.model.cuda()

        self.gamma = gamma
        self.loss = loss
        
        # for stepping
        self.steps = 0
        self.obs = None
        self.state_buffer = None
        self.dur = 0 
    
    def __epsilon_linear(self, epsilon, epsilon_end=0.05, decay=2000):
        """
        Linearily decay epsilon from epsilon to epsilon_end
        """
        eps_step = (epsilon - epsilon_end) / decay
        return epsilon - self.step * eps_step
    
    def __epsilon(self, epsilon, epsilon_end=0.05, decay=200):
        """
        Exponentially decay epsilon from epsilon to epsilon_end
        """
        return epsilon_end + (epsilon - epsilon_end) * np.exp(-1. * self.steps / decay)

    def next_action(self, observation, epsilon=0.9, random=False):
        """
        Choose the next action to take based on an observation
        
        Interpolates in a way that at epoch 0 the initial epsilon is used,
        and 80% of epochs passed, epsilon is 0.
        
        observation : sequence of frames in screen output format        
        """        
        eps_threshold = self.__epsilon(epsilon)
        
        if random or np.random.rand() <= eps_threshold:
            return torch.LongTensor([[self.screen.sample_action()]])
        else:
            observation = Variable(observation, volatile=True).type(torch.FloatTensor)
            return self.model(observation).data.max(1)[1].view(1, 1)
  
    def optimize(self, optimizer, screen, batchsize, data=None):
        """Train the model.

        Everything that relates to state, action, reward, next_state comes from
        replay memory.

        All we do is
        1. Predict the Q values for state (sequence of frames) from the model
        2. Predict the Q values fro next_state (sequence of frames) from the model
        3. Calculate the targetQ values via bellman equation (target = reward +
        discount * max Q(next_state))
        4. Do backpropagation with the Q(state) and the targetQ values

        Returns the train loss history.

        Args:
            optimizer : the optimizer.
            screen : wrapper for the environment
            batchsize (int) : the size of a batch
            data : use this data instead of sampling from memory
        """        
        if data is None:
            if not self.memory.can_sample(batchsize):
                return np.array([0])
            
            batch_state, batch_action, batch_reward, batch_next_state = self.memory.sample(batchsize)
            batch_state = Variable(torch.cat(batch_state))
            batch_action = Variable(torch.cat(batch_action))
            batch_reward = Variable(torch.cat(batch_reward))
            batch_next_state = Variable(torch.cat(batch_next_state))            
        else:
            batch_state, batch_action, batch_reward, batch_next_state = data
            batch_obs = Variable(obs.float())
            batch_action = Variable(action.long()) 
            batch_reward = Variable(reward.float())
            batch_next_obs = Variable(next_obs.float())

        if self.model.is_cuda:
            batch_state, batch_next_state = batch_state.cuda(), batch_next_state.cuda()
            batch_action, batch_reward = batch_action.cuda(), batch_reward.cuda()

        # current Q values are estimated by NN for all actions
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # expected Q values are estimated from actions which gives maximum Q value
        max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (self.gamma * max_next_q_values)    

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # backpropagation of loss to NN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.data.cpu().numpy()
    
    def __encode_frame(self, frame):
        # if this is an image, make sure it is float 
        if np.issubdtype(self.obs.dtype, np.integer):
            self.obs = self.obs.astype(np.float32) / 255.
        
        # create FloatTensor
        if len(self.obs.shape) == 1:
            state = torch.FloatTensor(self.obs)
        elif len(self.obs.shape) == 2:
            state = torch.FloatTensor(self.obs).unsqueeze(0)
        else:
            state = torch.FloatTensor(self.obs)       
            
        # return with batchsize 0
        return state.unsqueeze(0)
    
    def step(self, screen, save=False, random=False, render=False):
        """
        Stepwise playing (with state)
        
        Args:
            screen : wrapper for the environment
            epoch (int) : current epoch (for epsilon calculation)
            max_epoch (int) : last epoch (for epsilon calculation)
            save (bool) : Write to replay memory
        """
        # Initialize
        if self.obs is None:            
            self.obs = screen.reset() # new sequence
            self.rew = 0 # accumulated
            self.dur = 0 # accumulated
        
        
        # if this is an image, make sure it is float 
        if np.issubdtype(self.obs.dtype, np.integer):
            self.obs = self.obs.astype(np.float32) / 255.
        
        if len(self.obs.shape) == 1:
            state = torch.FloatTensor(self.obs)
        elif len(self.obs.shape) == 2:
            state = torch.FloatTensor(self.obs).unsqueeze(0)
        else:
            state = torch.FloatTensor(self.obs)           
            
        # next step in environment - model takes history into account
        action = self.next_action(self.__encode_frame(self.obs), random=random)
        
        # display screen if render is active
        if render:
            screen.render()
        
        # get new screen from environment
        self.next_obs, reward, done = screen.input(action[0,0])
        
        # increase steps in case we really played
        if save and not random:
            self.steps += 1
                
        # accumulated reward
        self.rew += reward
        self.dur += 1
        
        # store negative reward when sequence ends
        if done:
            reward = -1
            
        if save:          
            idx = self.memory.store_frame(self.__encode_frame(self.obs))
            self.memory.store_effect(idx, action, torch.FloatTensor([reward]), 
                                     self.__encode_frame(self.next_obs))
        
        self.obs = self.next_obs
        
        # reset observation when done
        if done:
            self.obs = None
        
        return self.obs, reward, action, done
    
    def initialize(self, screen, num_frames=10000):
        """Randomly initialize the replay buffer.

        Args:
            num_replays (int) : the number of replays.
        """
        initialized = False
        for e in range(num_frames):
            # run sequences (dont need epoch and max_epoch as random is True)
            self.step(screen, save=True, random=True)               

    def play(self, screen, max_duration=10000, save=False):
        """
        Play a sequence
        
        Args:
            screen : wrapper for the environment
            epoch (int) : current epoch (for epsilon calculation)
            max_epoch (int) : last epoch (for epsilon calculation)
            max_duration (int) : number of frames to be played 
            save (bool) : Write to replay memory
        """
        
        # while game not lost/terminated
        done = False        
        while not done and self.dur < max_duration:            
            _, _, _, done = self.step(screen, save)
            
        return self.rew, self.dur
