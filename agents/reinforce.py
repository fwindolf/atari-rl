"""Implementation of REINFORCE algorithm for policy gradient model."""
import math
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim

from agents.base import AgentBase
from models.policy import DiscretePolicy, ContinuousPolicy

pi = Variable(torch.FloatTensor([math.pi]))


def normal(x, mu, sigma_sq):
    """Generate a normal distribution."""
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


class REINFORCE(AgentBase):
    """Agent for REINFORCE policy gradient algorithm."""

    def __init__(self, screen, history_len, hidden_size, gamma, continuous):
        """Set up agent, build model, set it up."""  
        super().__init__(screen)
        num_inputs = int(history_len * screen.get_dim())
        num_outputs = screen.get_actions()
        
        if continuous:
            self.model = ContinuousPolicy(hidden_size, num_inputs, num_outputs)
        else:
            self.model = DiscretePolicy(hidden_size, num_inputs, num_outputs)
            
        self.continuous = continuous
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.gamma = gamma

        # sets model in training mode.
        self.model.train()
        
        # for stepping
        self.done = True
        self.obs = None
        self.dur = 0
        
    def next_action(self, observation, random=False):
        if random:
            action, log_prob, entropy = torch.LongTensor(
                [self.screen.sample_action()]), 0, 0
        if self.continuous:
            action, log_prob, entropy = self.__select_continuous_action(
                observation)
        else:
            action, log_prob, entropy = self.__select_discrete_action(
                observation)
            
        return action.unsqueeze(1), log_prob, entropy
    
    
    def __select_discrete_action(self, state):
        state = Variable(state)
        if self.model.is_cuda:
            state = state.cuda()
            
        probs = self.model(state)
        action = probs.multinomial().data
        prob = probs[:, action[0, 0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()
        return action[0], log_prob, entropy

    def __select_continuous_action(self, state):
        state = Variable(state)
        if self.model.is_cuda:
            state = state.cuda()
            
        mu, sigma_sq = self.model()
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        eps = Variable(eps)
        if self.model.is_cuda:
            eps = eps.cuda()
            
        action = (mu + sigma_sq.sqrt() * eps).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        log_prob = prob.log()
        return action, log_prob, entropy
        
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
        if self.done is True:            
            self.obs = screen.reset() # new sequence
            self.rew = 0 # accumulated
            self.dur = 0 # accumulated               
            # for updating params later
            self.entropies = []
            self.log_probs = []
            self.rewards = []
            
        # next step in environment - model takes history into account
        action, log_prob, entropy  = self.next_action(
            self.__encode_frame(self.obs),random=random)
        
        # display screen if render is active
        if render:
            screen.render()
        
        # get new screen from environment
        self.next_obs, reward, self.done = screen.input(action[0,0])
        
        # store entropies, log_probs
        if save == True:
            self.entropies.append(entropy)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
                        
        # accumulated reward
        self.rew += reward
        self.dur += 1
        
        # store negative reward when sequence ends
        if self.done:
            reward = -1        
        
        self.obs = self.next_obs
        
        return self.obs, reward, action, self.done
    
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
            _, _, _, done = self.step(screen, save=save, render=render)
            
        return self.rew, self.dur

    def optimize(self, optimizer, screen, batchsize, data=None):
        R = torch.zeros(1, 1)
        loss = 0
        
        if data is None:
            # only optimize after a full episode
            if self.done == False:
                return 0
            
            rewards = self.rewards
            entropies = self.entropies
            log_probs = self.log_probs
            
        else:
            # TODO:  generate entropies and log_probs from dataset
            raise NotImplemented()
                        
        # discount rewards
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            Rprime = Variable(R).expand_as(log_probs[i])
            entropy = 0.0001*entropies[i]
            if self.model.is_cuda:
                Rprime, entropy = Rprime.cuda(), entropy.cuda()
                
            loss = loss - (log_probs[i] * Rprime).sum() - entropy.sum()
        
        # mean of rewards
        loss = loss / len(rewards)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        optimizer.step()

        return loss.data.cpu().numpy()


