"""Solver that trains models."""
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
import logging

class Solver():
    """The solver.

    The models can be trained online (with an agent playing inside an
    environment) or offline (using data generated from games).
    """

    def __init__(self, optimizer, batchsize=100, log_level='WARNING'):
        """Create a new Solver class and start logging.

        Args:
            optimizer (torch.optim) : Optimizer for trainig the model
            loss (torch.nn)         : Loss function
            batchsize (int)         : Size of a minibatch of observations
        """
        self.optimizer = optimizer
        self.batchsize = batchsize

        self.log_level = log_level.upper()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        
        self.online_scores = []
        self.online_durations = []
        self.online_losses = []
        
        self.data_loss_history = []
        
        
    def __decay_learning_rate_half(self, learning_rate, epoch):
        """ Drop the learning rate by half every 10 epochs"""
        return learning_rate * np.power(0.5, np.floor((1 + epoch) / 100)) 
    
    def __decay_learning_rate(self, learning_rate, epoch):
        """ Drop the learning rate by half every 10 epochs"""
        return np.power(0.999, epoch) * learning_rate 
        
    def train_dataset(self, agent, screen, data_loader, num_epochs=10, learning_rate=0.01, decay=False):
        """
        Let the agent train pseudo online, without sampling from/to replay buffer
        Args:
            agent (agent) : The agent that should be trained
            data_loader : torch DataLoader with training data
            num_epochs (int) : Total number of training epochs
            learning_rate (float) : The learning rate with which the model is udpated
        """
        self.logger.info('Training from Dataset started')
        
        self.data_loss_history = []
        
        optim = self.optimizer(agent.model.parameters(), learning_rate)
        
        for epoch in range(num_epochs):                        
            for i, data in enumerate(data_loader, 1):
                loss = agent.optimize(optim, screen, self.batchsize, data=data)
                self.data_loss_history.append(loss)
                
            self.logger.info('Epoch %d/%d: Mean loss %f' % 
                             (epoch, num_epochs, np.mean(self.data_loss_history[-len(data_loader):])))
            
            if epoch % 10 == 9: # benchmark every 10 epochs
                best, mean, dur = self.play(agent, screen, num_sequences=10)
                self.logger.info('Epoch %d/%d: Mean score %d with %d frames' % 
                             (epoch, num_epochs, mean, dur))
            
            # advance trajectory
            data_loader.dataset.next()
            
            # decay learning rate
            if decay:
                optim.learning_rate = self.__decay_learning_rate(learning_rate, epoch)
            
            del data
                
        self.logger.info('Training from Dataset finished')
        
    def train_online(self, agent, screen, num_epochs=200, learning_rate=0.01, decay=False):
        """
        Let the agent play in the environment to optimize the strategy
        Args:
            agent (agent) : The agent that should be trained
            screen (screen) : Screen wrapper around the environment
            num_epochs (int) : Total number of training epochs
            learning_rate (float) : The learning rate with which the model is udpated
        """
        self.logger.info('Online Training started')
        
        optimizer = self.optimizer(agent.model.parameters(), learning_rate)
        
        for epoch in range(num_epochs):
            steps = 0
            score = 0
            done = False
            losses = []
            while not done:        
                _, reward, _, done = agent.step(screen, save=True, render=False)

                loss = agent.optimize(optimizer, screen, self.batchsize)
                losses.append(loss)
                steps += 1
                score += reward
            
            self.logger.info('Epoch %d/%d - Score %d with Duration %d (Loss %f)' %
                                 (epoch, num_epochs, score, steps, np.mean(losses)))
            
            self.online_losses.append(np.mean(losses))
            self.online_scores.append(score)
            self.online_durations.append(steps)
            
            # decay learning rate
            if decay:
                optim.learning_rate = self.__decay_learning_rate(learning_rate, epoch)
        
        self.logger.info('Online Training finished')
        
    def play(self, agent, screen, num_sequences=1, save=False, render=False):
        """
        Let the agent play to benchmark
        agent (agent)   : The agent that should be trained
        screen (screen) : Screen wrapper around the environment
        """

        self.logger.debug('Wanna play a game? - Game started!')

        best_score = 0
        scores = []
        durations = []

        for i in range(num_sequences):
            score, duration = agent.play(screen, save=save, render=render)
            scores.append(score)
            durations.append(duration)

            self.logger.debug('Score %d after %d frames' % (score, duration))

            if score > best_score:
                self.logger.debug('New Highscore %d' % (score))
                best_score = score

        self.logger.debug('Game ended')

        return best_score, np.mean(scores), np.mean(durations)