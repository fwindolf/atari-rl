"""Agents based on deep Q-networks as proposed by Deepmind for atari games."""

from agents.base import AgentBase
from models.dqn import DQN
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from utils.replay_buffer import ReplayBuffer


class DQNAgent(AgentBase):
    """A DQN agent implementation according to the DeepMind paper."""

    def __init__(self, screen, mem_size=100000, history_len=10, gamma=0.999,
                 loss=CrossEntropyLoss()):
        """Initialize agent."""
        super().__init__(screen)
        self.history_len = history_len
        self.memory = ReplayBuffer(mem_size, history_len, screen)
        self.model = DQN(history_len, screen.get_actions())
        
        if torch.cuda.is_available():
            self.model.cuda()

        self.gamma = gamma
        self.loss = loss
        
        # for stepping
        self.obs = None
        self.state_buffer = None
        self.dur = 0 


    def initialize(self, num_replays=10000):
        """Randomly initialize the replay buffer.

        Args:
            num_replays (int) : the number of replays.
        """
        self.memory.initialize_random(num_replays)

    def __encode_model_input(self, observation):
        """Encode the observation in a way that the model can use it.

        Args :
            observation : Any number of frames in screen output format
            (greyscale)
        """
        num_channels = 1      
        if len(observation.shape) == 4:
            num_channels = observation.shape[1]  # Batchsize, Channels, H, W
        elif len(observation.shape) == 3:
            num_channels = observation.shape[0]  # Channels, H, W
        
        assert (num_channels <= self.history_len)

        if observation.dtype.name == 'uint8':
            observation = observation.astype('float') / 255.

        if num_channels < self.history_len:
            num_missing = self.history_len - num_channels
            observation = np.concatenate((
                np.zeros((*observation.shape[0:2], num_missing)),
                np.expand_dims(observation, 2)), axis=2)
        
        observation = torch.FloatTensor(observation)
        
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        
        assert(len(observation.shape) == 4)       
        
        return observation


    def next_action(self, observation, epoch, max_epochs, epsilon=0.9):
        """
        Choose the next action to take based on an observation
        
        Interpolates in a way that at epoch 0 the initial epsilon is used,
        and 80% of epochs passed, epsilon is 0.
        
        observation : sequence of frames in screen output format        
        """        
        # decay epsilon
        cur_eps = max(0, epsilon * (1 - (1.0/0.8 * min(1, epoch/max_epochs))))  
        
        if np.random.rand() < cur_eps:            
            action = self.screen.sample_action() # random action from action space
        else:            
            observation = Variable(self.__encode_model_input(observation), volatile=True)
            action = self.model.predict(observation).data.cpu().numpy().squeeze()
            
        return action
  
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
            data = self.memory.sample(batchsize)
            
        obs, action, reward, done, next_obs = data
        
        # convert to variables (not from dataloader, so manually wrap in torch tensor)
        obs = Variable(obs.float())
        action = Variable(action.long()) 
        reward = Variable(reward.float())
        
        # mask of non final next_obs 
        non_final_mask = done.int().eq(0).nonzero().squeeze(1) # nonzero() adds dimension

        # Observations that dont end the sequence
        non_final_obs = Variable(next_obs[non_final_mask, :])
        
        # future predicted rewards
        next_obs_values = Variable(torch.zeros(reward.data.shape))
        
        if self.model.is_cuda:
            obs = obs.cuda()
            action = action.cuda()
            reward = reward.cuda()
            non_final_mask = non_final_mask.cuda()
            non_final_obs = non_final_obs.cuda()
            next_obs_values = next_obs_values.cuda()
        
        # q values predicted by model for observation and select the columns of actions taken
        obs_action_values = self.model(obs).gather(1, action.unsqueeze(1))

        # future rewards predicted by model (when sequences dont end)
        next_obs_values[non_final_mask] = self.model(non_final_obs).max(1)[0]
        next_obs_values = Variable(next_obs_values.data, volatile=False)
        
        # To not mess up the loss
        next_obs_values.volatile = False
        
        # Expected q values from observation
        expected_obs_action_values = (next_obs_values * self.gamma) + reward

        # Loss is the difference between the calculated and the predicted q values
        loss = self.loss(obs_action_values, expected_obs_action_values)
        
        # Apply to model
        optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)  # clamp gradient to stay stable
        optimizer.step()

        return loss.data.cpu().numpy()
    
    def step(self, screen, epoch, max_epochs, save=False):
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
            self.obs = screen.reset()
            self.last_obs = self.obs
            self.rew = 0
            self.dur = 0
        
        # Buffer for last history_len frames
        if self.state_buffer is None:
            self.state_buffer = np.zeros([self.memory.get_history_len(), *self.obs.shape])
            
        action = self.next_action(self.state_buffer, epoch, max_epochs)
        self.obs, reward, done = screen.input(action)
        
        self.rew += reward
        self.dur += 1
        
        # accumulate states in state_buffer
        state = self.obs - self.last_obs
        self.state_buffer = np.roll(self.state_buffer, -1, axis=0) # shift to make room for new
        self.state_buffer[-1] = state
        self.last_obs = self.obs
        
        if save:
            idx = self.memory.store_frame(self.obs)
            self.memory.store_effect(idx, action, reward, done)
        
        if done:
            self.obs = None
        
        return state, reward, action, done

    def play(self, screen, epoch, max_epoch, max_duration=10000, save=False):
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
            _, _, _, done = self.step(screen, epoch, max_epoch, save)
            
        return self.rew, self.dur
    
class HumanMemoryDQNAgent(DQNAgent):
    """
    The DQN Agent that does not sample the initial replay memories from
    random actions in the environment but from the AtariGC dataset
    """
    def __init__(self, screen, mem_size=100000, history_len=10):
        super().__init__(screen, mem_size, history_len)
        
    def initialize(self, dataset, num_replays=100000):
        """
        Initialize the replay buffer from a dataset
        """
        self.memory.initialize_dataset(num_replays, dataset)
        

class HumanTrainedDQNAgent(DQNAgent):
    """
    The DQN Agent that first trains the model from the dataset and then
    initializes the replay buffer by playing
    """
    def __init__(self, screen, mem_size=100000, history_len=10):
        super().__init__(screen, mem_size, history_len)
    
    def initialize(self, solver, dataset, num_epochs=100, num_replays=10000):
        """
        Initialize by training the model, then initialize the replay memory 
        by playing
        """
        # Train the model from some dataset
        data_train, data_valid, data_test = dataset.split(0.7, 0.2)
        train_loader = DataLoader(data_train, batchsize=10, num_workers=4)
        val_loader = DataLoader(data_val, batchsize=10, num_workers=4)
        test_loader = DataLoader(data_test, batchsize=10, num_workers=4)
        
        solver.train_offline(self, train_loader, val_loader, num_epochs)
        
        solver.eval(self, test_loader)
                
        self.memory.initialize_playing(num_replays, agent)
