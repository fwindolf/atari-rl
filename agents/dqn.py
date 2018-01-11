from agents.base import AgentBase
from utils.replay_buffer import ReplayBuffer
from models.dqn import DQN

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F


import numpy as np

class DQNAgent(AgentBase):
    """
    The DQN agent implementation according to the DeepMind paper
    """
    def __init__(self, screen, mem_size=100000, history_len=10, gamma=0.999, loss=F.smooth_l1_loss):
        super().__init__(screen)
        self.memory = ReplayBuffer(mem_size, history_len, screen)
        self.model = DQN(history_len, screen.get_actions())
        
        self.gamma = gamma
        self.loss = loss
        
    def initialize(self, num_replays=10000):
        """
        Randomly initialize the replay buffer
        """
        self.memory.initialize_random(num_replays)
        
    def __encode_model_input(self, observation):
        """
        Encode the observation in a way that the model can use it
        
        observation : Any number of frames in screen output format (greyscale)
        """
        num_channels = 1      
        history_len = self.memory.get_history_len()
        if len(observation.shape) > 2:
            num_channels = observation.shape[2]
            
        assert (num_channels < history_len)
        
        if observation.dtype.name == 'uint8':
            observation = observation.astype('float') / 255.
            
        if num_channels < history_len: 
            num_missing = history_len - num_channels
            observation = np.concatenate((
                np.zeros((*observation.shape[0:2], num_missing)), 
                np.expand_dims(observation, 2)), axis=2)

        return torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0) # make N,c,h,w
        
    def next_action(self, observation):
        """
        Choose the next action to take based on an observation
        
        observation : 0-<history_len> number of frames in screen output format        
        """
        
        # TODO implement strategy for epsilon
        if np.random.rand() < 0.0:
            return self.screen.sample_action() # random action from action space
        else:            
            observation = self.__encode_model_input(observation)            
            prediction = self.model.predict(Variable(observation, volatile=True)).data.cpu().numpy()
            return int(prediction)
    
    def optimize(self, optimizer, screen, batchsize, num_epochs, log_nth):
        """
        Everything that relates to state, action, reward, next_state comes from replay memory
        
        All we do is 
        1. Predict the Q values for state (sequence of frames) from the model
        2. Predict the Q values fro next_state (sequence of frames) from the model
        3. Calculate the targetQ values via bellman equation (target = reward + discount * max Q(next_state))
        4. Do backpropagation with the Q(state) and the targetQ values
        """     
        for epoch in range(num_epochs):
            obs, action, reward, done, next_obs = self.memory.sample(batchsize)

            # convert to variables (not from dataloader, so manually wrap in torch tensor)
            obs = Variable(torch.FloatTensor(obs)) 
            action = Variable(torch.LongTensor(action)) 
            reward = Variable(torch.FloatTensor(reward)) 
            non_final_mask = Variable(torch.ByteTensor((done==0).astype(np.int))) # invert done

            # Observations that dont end the sequence
            next_obs = torch.FloatTensor([o for o in next_obs if o is not None])
            non_final_obs = Variable(next_obs)

            # Q(s_t, a) -> Q(s_t) from model and select the columns of actions taken
            # .gather() chooses the confidence values that the model predicted at the index of the action
            # that was originally taken
            obs_action_values = self.model(obs).gather(1, action.unsqueeze(1))

            # V(s_t+1) for all next observations
            next_obs_values = Variable(torch.zeros(batchsize).type(torch.Tensor))
            next_obs_values[non_final_mask] = self.model(non_final_obs).max(1)[0] # future rewards predicted by model
            next_obs_values = Variable(next_obs_values.data, volatile = False)

            expected_obs_action_values = (next_obs_values * self.gamma) + reward

            loss = self.loss(obs_action_values, expected_obs_action_values)

            optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1) # clamp gradient to stay stable
            optimizer.step() 
            
            
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
        