"""Implementation of REINFORCE algorithm for policy gradient model."""
import math
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim

from models.policy import DiscretePolicy, ContinuousPolicy

pi = Variable(torch.FloatTensor([math.pi])).cuda()


def normal(x, mu, sigma_sq):
    """Generate a normal distribution."""
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

class REINFORCE:
    """Agent for REINFORCE policy gradient algorithm."""

    def __init__(
        self,
        hidden_size,
        num_inputs,
        action_space,
        gamma,
        ckpt_freq,
        continuous
    ):
        """Set up agent, build model, set it up."""
        self.action_space = action_space
        if continuous:
            self.model = ContinuousPolicy(hidden_size,
                                          num_inputs,
                                          action_space)
        else:
            self.model = DiscretePolicy(hidden_size, num_inputs, action_space)
        self.continuous = continuous
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = gamma
        self.ckpt_freq = ckpt_freq

        # sets model in training mode.
        self.model.train()

    def select_discrete_action(self, state):
        probs = self.model(Variable(state).cuda())
        action = probs.multinomial().data
        prob = probs[:, action[0, 0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()
        return action[0], log_prob, entropy

    def select_continuous_action(self, state):
        mu, sigma_sq = self.model(Variable(state).cuda())
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt()*Variable(eps).cuda()).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss \
                - (
                    log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()
                  ).sum() \
                - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

    def train(self, env, num_episodes, max_episode_length):
        """Trains the model."""
        ckpt_dir = 'checkpoints'
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        rewards_cum = list()

        for i_episode in range(num_episodes):
            state = torch.Tensor([env.reset()])
            entropies = []
            log_probs = []
            rewards = []
            for t in range(max_episode_length):
                if self.continuous:
                    action, log_prob, entropy = \
                        self.select_continuous_action(state)
                else:
                    action, log_prob, entropy = \
                        self.select_discrete_action(state)
                action = action.cpu()

                next_state, reward, done, _ = env.step(action.numpy()[0])

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = torch.Tensor([next_state])

                if done:
                    break

            self.update_parameters(rewards, log_probs, entropies, self.gamma)

            rewards_cum.append(np.sum(rewards))
            if i_episode % self.ckpt_freq == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(ckpt_dir, 'reinforce-'+str(i_episode)+'.pkl'))
                print("Episode: {}, reward: {}".format(
                    i_episode, np.sum(rewards)
                ))

        return rewards_cum
