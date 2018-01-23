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

    def select_action(self, Qvalues,selection_type='e_greedy', epsilon=0.3):

        # adapted from
        # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf

        # 1 random approach
        if selection_type == 'random':
            action = self.screen.sample_action()

        # 2 greedy approach
        elif selection_type == 'greedy':
            action = np.argmax(Qvalues)

        # 4 boltzmann approach
        elif selection_type == 'boltzmann':
            action_value = np.random.choice(Qvalues, p=Qvalues)
            action = np.argmax(Q_values == action_value)

        # 5 bayesian approach (multiple predictions with dropout) - advanced
        # different network architecture neccessary

        # 3 epsilon greedy approach - DEFAULT
        else:
            if np.random.rand(1) < epsilon:
                action = self.screen.sample_action()
            else:
                action = np.argmax(Qvalues)

        return int(action)


    def next_action(self, observation):
        """
        Choose the next action to take based on an observation
        
        observation : 0-<history_len> number of frames in screen output format        
        """

        observation = self.__encode_model_input(observation)
        Qvalues = self.model(Variable(observation, volatile=True)).data.cpu().numpy()
        prediction = self.select_action(Qvalues,selection_type='e_greedy', epsilon=0.3)

        return int(prediction)

    
    def optimize(self, optimizer, screen, batchsize, num_epochs, logger, log_nth):
        """
        Everything that relates to state, action, reward, next_state comes from replay memory
        
        All we do is 
        1. Predict the Q values for state (sequence of frames) from the model
        2. Predict the Q value of best action for next_state (sequence of frames) from the model
        3. Predict target Q values of best action of next state from target model 
        4. Calculate bellman equation (target = reward + discount * target Q value)
        5. Do backpropagation with the Q(state) and the targetQ values
        """

        num_param_updates = 0
        target_update_freq = 100        # <- test this value (suggested 10.000)

        for epoch in range(num_epochs):

            # sample transition batch from replay memory, done=1 if next state is end of episode
            obs_t, act_t, rew_t, done, obs_tp1 = self.memory.sample(batchsize)

            FloatTensor = torch.cuda.FloatTensor if self.model.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if self.model.is_cuda else torch.LongTensor
            ByteTensor = torch.cuda.ByteTensor if self.model.is_cuda else torch.ByteTensor

            # convert to variables (not from dataloader, so manually wrap in torch tensor)
            obs_t = Variable(torch.FloatTensor(obs_t))
            act_t = Variable(torch.LongTensor(act_t))
            rew_t = Variable(torch.FloatTensor(rew_t))
            non_final_mask = Variable(torch.ByteTensor((done==0).astype(np.int))) # invert done

            # Observations that don't end the sequence
            obs_tp1 = torch.FloatTensor([o for o in obs_tp1 if o is not None])
            obs_tp1_non_final = Variable(obs_tp1)

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

            logger.debug('Loss: %f' % float(loss.data.cpu().numpy()))

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1) # clamp gradient to stay stable

            # update
            optimizer.step()
            num_param_updates += 1

            # update target model weights with model weights
            if num_param_updates % target_update_freq == 0:
                self.target_Q_model.load_state_dict(self.model.state_dict())


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

    def play(self, screen, max_duration=100000, save = True):
        """
        Let agent play on openAI gym environment for benchmarking
        """

        # set to initial
        obs = screen.reset()        
        
        running_reward = 0
        duration = 0
        
        done = False
        # while game not lost/terminated
        while not done and duration < max_duration:
            action = self.next_action(obs)              # predict next action
            obs, reward, done = screen.input(action)    # apply action to environment

            if save:
                idx = self.memory.store_frame(obs)
                self.memory.store_effect(idx, action, reward, done)
                        
            running_reward += reward
            duration += 1

        return running_reward, duration