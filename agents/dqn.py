"""Agents based on deep Q-networks as proposed by Deepmind for atari games."""

from agents.base import AgentBase
from models.dqn import DQN
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import SmoothL1Loss
import torch.nn.functional as F
from utils.replay_buffer import ReplayBuffer

class DQNAgent(AgentBase):
    """A DQN agent implementation according to the DeepMind paper."""
    def __init__(self, screen, history_len=10, gamma=0.8, loss=SmoothL1Loss(), memory=None, model=None):
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
        self.done = True # start new sequence
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
        else:
            batch_state, batch_action, batch_reward, batch_next_state = data
            batch_state = Variable(batch_state.float())
            batch_action = Variable(batch_action.long().unsqueeze(1))
            batch_reward = Variable(batch_reward.float())
            batch_next_state = batch_next_state.float()
        
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch_next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch_next_state if s is not None], dim=0), volatile=True)
        next_state_values = Variable(torch.zeros(batchsize).float())

        if self.model.is_cuda:
            batch_state, batch_action, batch_reward = batch_state.cuda(), batch_action.cuda(), batch_reward.cuda()
            non_final_next_states, next_state_values = non_final_next_states.cuda(), next_state_values.cuda()
            non_final_mask = non_final_mask.cuda()
        
        try:
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            state_action_values = self.model(batch_state).gather(1, batch_action)

            # Compute V(s_{t+1}) for all next states.   
            if len(non_final_next_states.shape) == 3:
                non_final_next_states = non_final_next_states.unsqueeze(1)
            next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
            next_state_values.volatile = False
            
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + batch_reward

            # loss is measured from error between current and newly expected Q values
            loss = self.loss(state_action_values, expected_state_action_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
    
            return loss.data.cpu().numpy()
        except RuntimeError:
            return 0
    
    def __encode_frame(self, frame):
        # if this is an image, make sure it is float 
        if np.issubdtype(frame.dtype, np.integer):
            frame = frame.astype(np.float32) / 255.
    
        if len(frame.shape) == 1:
            state = torch.FloatTensor(frame)
        elif len(frame.shape) == 2:
            state = torch.FloatTensor(frame).unsqueeze(0)
        else:        
            if frame.shape[0] > frame.shape[2]:
                frame = frame.transpose(2, 0, 1)
            state = torch.FloatTensor(frame)     

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
        if self.done:
            self.current_frame = self.__encode_frame(self.screen.reset())
            self.last_frame = self.current_frame            
            self.state = self.current_frame - self.last_frame
            
            self.rew = 0 # accumulated
            self.dur = 0 # accumulated
            
        # next step in environment - model takes history into account
        action = self.next_action(self.state, random=random)
        self.last_frame = self.current_frame
        
        # display screen if render is active
        if render:
            screen.render()
        
        # get new screen from environment
        self.current_frame, reward, self.done = screen.input(action[0,0])
        self.current_frame = self.__encode_frame(self.current_frame)
        
        # increase steps in case we really played
        if save and not random:
            self.steps += 1
                
        # accumulated reward
        self.rew += reward
        self.dur += 1
        
        # store negative reward when sequence ends
        if not self.done:
            self.next_state = self.current_frame - self.last_frame
        else:
            reward = -1.
            self.next_state = None
            
        if save:          
            self.memory.push(self.state, action, torch.FloatTensor([reward]), self.next_state)
        
        self.state = self.next_state
        
        return self.state, reward, action, self.done
    
    def initialize(self, screen, num_frames=10000):
        """Randomly initialize the replay buffer.

        Args:
            num_replays (int) : the number of replays.
        """
        initialized = False
        for e in range(num_frames):
            # run sequences (dont need epoch and max_epoch as random is True)
            self.step(screen, save=True, random=True)               

    def play(self, screen, max_duration=10000, save=False, render=False):
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
            _, _, _, done = self.step(screen, save, render)
            
        return self.rew, self.dur