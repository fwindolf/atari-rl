from agents.base import AgentBase
from utils.replay_buffer import ReplayBuffer
from models.dueling_dqn import Dueling_DQN

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F


import numpy as np

# Adaptation from https://github.com/dxyang/DQN_pytorch

class Dueling_DQNAgent(AgentBase):
    """
    The Dueling DQN agent implementation
    """
    def __init__(self, screen, mem_size=100000, history_len=10, gamma=0.999, loss=F.smooth_l1_loss):
        super().__init__(screen)

        self.memory = ReplayBuffer(mem_size, history_len, screen)
        self.model = Dueling_DQN(history_len, screen.get_actions())
        self.target_Q_model = Dueling_DQN(history_len, screen.get_actions())
        
        self.history_len = history_len
        self.gamma = gamma
        self.loss = loss
        
        # for stepping
        self.obs = None
        self.state_buffer = None
        self.dur = 0 
        
        # for updating target network
        self.num_param_updates = 0
        
    def initialize(self, num_replays=10000):
        """
        Randomly initialize the replay buffer
        """
        initialized = False
        while(not initialized):
            # run sequences
            self.screen.reset()
            done = False
            while(not done):                
                action = self.screen.sample_action() # random action
                obs, reward, done = self.screen.input(action)                
                idx = self.memory.store_frame(obs)
                self.memory.store_effect(idx, action, reward, done)
                    
                if self.memory.num_transitions >= num_replays:        
                    initialized = True
                    break  
        
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
            
        return observation
    
        
    def __epsilon(self, epsilon, epsilon_end=0.05, epoch=1, max_epochs=1):
        """
        Linearily decay epsilon from epsilon to epsilon_end
        """
        eps_step = (epsilon - epsilon_end) / max_epochs
        return epsilon - epoch * eps_step
    

    def next_action(self, observation, epoch, max_epochs, epsilon=0.9):
        """
        Choose the next action to take based on an observation
        
        Interpolates in a way that at epoch 0 the initial epsilon is used,
        and 80% of epochs passed, epsilon is 0.
        
        observation : sequence of frames in screen output format        
        """        
        # decay epsilon
        cur_eps = self.__epsilon(epsilon, epoch=epoch, max_epochs=max_epochs)  
        
        if np.random.rand() < cur_eps:            
            action = self.screen.sample_action() # random action from action space
        else:            
            observation = Variable(self.__encode_model_input(observation), volatile=True)
            action = self.model.predict(observation).data.cpu().numpy().squeeze()
            
        return action

    
    def optimize(self, optimizer, screen, batchsize, data=None):
        """
        Everything that relates to state, action, reward, next_state comes from replay memory
        
        All we do is 
        1. Predict the Q values for state (sequence of frames) from the model
        2. Predict the Q value of best action for next_state (sequence of frames) from the model
        3. Predict target Q values of best action of next state from target model 
        4. Calculate bellman equation (target = reward + discount * target Q value)
        5. Do backpropagation with the Q(state) and the targetQ values
        """
        target_update_freq = 100        # <- test this value (suggested 10.000)

        # sample transition batch from replay memory, done=1 if next state is end of episode
        obs_t, act_t, rew_t, done, obs_tp1 = self.memory.sample(batchsize)

        FloatTensor = torch.cuda.FloatTensor if self.model.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.model.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if self.model.is_cuda else torch.ByteTensor

        # convert to variables (not from dataloader, so manually wrap in torch tensor)
        obs_t = Variable(obs_t.float())
        act_t = Variable(act_t.long())
        rew_t = Variable(rew_t.float())
        non_final_mask = Variable(done==0) # invert done

        # Observations that don't end the sequence
        #obs_tp1 = torch.FloatTensor([o for o in obs_tp1 if o is not None])
        obs_tp1_non_final = Variable(obs_tp1[done.eq(0).nonzero().squeeze()])

        # input batches to network
        # get Q values for current observation -> Q(s, a, theta_i)
        q_s_a = self.model(obs_t).gather(1, act_t.unsqueeze(1)).squeeze()

        # get Q values of best action for next observation -> max Q(s', a', theta_i)
        q_tp1_values = self.model(obs_tp1_non_final).detach()
        _, a_prime = q_tp1_values.max(1)

        # initialize with zeros to cater for final states (value = 0 if terminal state)
        q_target_s_a_prime = Variable(
            torch.zeros(batchsize).type(torch.Tensor)
        )

        # get target Q values from frozen network for next state and choosen action
        # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
        q_target_tp1_value = self.target_Q_model(obs_tp1_non_final ).detach()
        q_target_s_a_prime[non_final_mask] = q_target_tp1_value.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()

        expected_obs_action_values = (q_target_s_a_prime * self.gamma) + rew_t
        loss = self.loss(q_s_a, expected_obs_action_values)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1) # clamp gradient to stay stable

        # update
        optimizer.step()
        self.num_param_updates += 1

        # update target model weights with model weights
        if self.num_param_updates % target_update_freq == 0:
            self.target_Q_model.load_state_dict(self.model.state_dict())
            
        return loss.data.numpy()


        # Implementation Check list - Please keep until fuly resolved!
        # For one episode
        # DONE 1. check stopping criterion
        # DONE 2. get current observation
        # DONE 3. action, Q_values = model(observation) (use action selection)
        # REPLAY MEMORY 4. next_observation, reward, done = environment(action)
        # REPLAY MEMORY 5. store tuple in replay buffer < observation, action, reward, next_observation >
        # - 6. if done, reset initial observation
        # - 7. next_observation is now current observation
        # DONE 8. Perform experience replay and train network, if replay buffer is big enough
        # DONE 9. sample transition batch from replay memory < obs, act, rew, done, n_obs >
        # DONE 10. Q_values = model(obs) -> using action get relevant rel_Q_values
        # [DDQN]
        # DONE 11. n_Q_values = model(n_obs) -> max(n_Q_values) -> get respective action (n_action)
        # DONE 12. target_Q_values = target_model(n_obs) -> target_Q_value(n_action) = a_target_Q
        # DONE13. take care of terminated episode -> predicted Q_value is not of any sense
        # DONE 14. compute bellman error: error = rew + gamma * a_target_Q - Q_values
        # DONE 15. backward pass
        # DONE 16. optimize model
        # DONE 17. after certain model optimizations, reset target_model to weights of model
        # DONE 18. logging

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