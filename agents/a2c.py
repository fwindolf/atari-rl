from agents.base import AgentBase
from utils.replay_buffer import ReplayBuffer
from models.a2c import A2C

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

# adapted from https://github.com/lnpalmer/A2C
# and adapted from https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

class A2CAgent(AgentBase):
    """
    The A2C agent implementation
    """
    def __init__(self, screen, mem_size=100000, history_len=10, gamma=0.999, loss=F.smooth_l1_loss):
        super().__init__(screen)

        self.memory = ReplayBuffer(mem_size, history_len, screen)
        self.model = A2C(history_len, screen.get_actions())

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
        policies, values = self.model(Variable(observation, volatile=True))

        policy_probs = F.softmax(policies.squeeze(), dim=0)
        # as usual
        #prediction = self.select_action(policy_probs,selection_type='e_greedy', epsilon=0.3)

        # or using multinomial
        prediction = torch.multinomial(policy_probs,1).data.cpu().numpy()

        return int(prediction)


    def process_rollout(self, rollout_data, lambd=1.0):

        FloatTensor = torch.cuda.FloatTensor if self.model.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.model.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if self.model.is_cuda else torch.ByteTensor

        # extract final value
        _, _, _, _, last_value = rollout_data[-1]
        returns = last_value.data

        # pipeline variable to cuda
        advantages = torch.zeros(1,1)
        advantages = FloatTensor(advantages)

        # pre-construct output container
        out = [None] * (len(rollout_data) - 1)

        # run Generalized Advantage Estimation -> calculate returns and advantages
        for t in reversed(range(len(rollout_data) -1)):

            rewards, masks, actions, policies, values = rollout_data[t]
            _, _, _, _, next_values = rollout_data[t + 1]

            returns = rewards + returns * self.gamma * masks

            deltas = rewards + next_values.data * self.gamma * masks - values.data
            advantages = advantages * self.gamma * lambd * masks + deltas

            out[t] = actions, policies, values, returns, advantages

        # return data as bacthed Tensor, Variables
        return map(lambda x: torch.cat(x, 0), zip(*out))

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

        FloatTensor = torch.cuda.FloatTensor if self.model.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.model.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if self.model.is_cuda else torch.ByteTensor
        cuda = self.model.is_cuda
        print('Real Cuda: %s' % str(cuda))

        # Hyperparameters (default from link)
        rollout_steps = 20
        rollout_data = []
        value_coeff = 0.5
        entropy_coeff = 0.01
        lambd = 1.0
        grad_norm_limit = 40

        losses = 0

        # get initial state from environment
        obs = screen.reset()
        obs = self.__encode_model_input(obs)
        obs = FloatTensor(obs)

        # step through every epoch
        for epoch in range(num_epochs):

            for _ in range(rollout_steps):

                # get policy and values from model
                policies, values = \
                    self.model(Variable(obs, volatile=False))

                # calculate actions from policy probabilities
                policy_probs = F.softmax(policies.squeeze(), dim=0)
                act = policy_probs.multinomial().data

                # step environment
                obs, rew, dones = screen.input(act.cpu().numpy())
                obs = self.__encode_model_input(obs)

                # reset the LSTM state for done envs
                masks = (1. - torch.from_numpy(np.array([0 if dones == False else 1], dtype=np.float32))).unsqueeze(1)
                rew = torch.from_numpy(np.array([rew], dtype=np.float32)).float().unsqueeze(1)

                # save for rollout
                rollout_data.append((rew, masks, act, policies, values))


            # compute values of last state and append to rollout
            test, final_values = \
                self.model(Variable(obs, volatile=False))

            rollout_data.append((None, None, None, None, final_values))

            # extract rollout
            acts, pols, vals, rets, advs = self.process_rollout(rollout_data=rollout_data, lambd=lambd)

            # calculate action probabilities
            pol_probs = F.softmax(pols, dim=1)
            log_pol_probs = F.log_softmax(pols, dim=1)
            log_act_probs = log_pol_probs.gather(1, Variable(acts.unsqueeze(1)))

            # calculate respective losses
            policy_loss = (-log_act_probs * Variable(advs)).sum()
            value_loss = (0.5 * (vals - Variable(rets)) ** 2.).sum()
            entropy_loss = (log_pol_probs * pol_probs).sum()

            # construct final loss
            loss = policy_loss + value_loss * value_coeff + entropy_loss * entropy_coeff
            loss.backward()

            logger.debug('Loss: %f' % float(loss.data.cpu().numpy()))
            losses += float(loss.data.cpu().numpy())

            nn.utils.clip_grad_norm(self.model.parameters(), grad_norm_limit)
            optimizer.step()
            optimizer.zero_grad()

            # clear roll out
            rollout_data = []

        meanLoss = losses/num_epochs
        logger.debug('Mean Loss: %f' % float(meanLoss))

        #raise NotImplemented()

        # Implementation Check list - Please keep until fuly resolved!
        # For one episode
        # 0. for rollout
        # 1. predict policiy & values for current observation
        #   2. apply softmax to policy for probabilities
        #   3. compute action probabilities with weighted policy probability
        #   4. perform action on environment
        #   5. compute terminal mask
        #   6. weired plotting/logging
        #   7. append rollout for LSTM
        # 8. calculate log-/probabilities for policy
        # 9. calculate log action probabilities
        # 10. compute policy loss: sum(-log_act_probs * advantages)
        # 11. compute value loss: L2norm values - returns
        # 12. compute entropy loss: sum(log probs * probs)
        # 13. loss = policy + value + entropy
        # 14. backpropagate
        # 15. optiize


        '''
        num_param_updates = 0
        target_update_freq = 100        # <- test this value (suggested 10.000)

        for epoch in range(num_epochs):

            # sample transition batch from replay memory, done=1 if next state is end of episode
            obs_t, act_t, rew_t, done, obs_tp1 = self.memory.sample(batchsize)

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
        '''

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