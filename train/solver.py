import torch
from torch.autograd import Variable
import numpy as np
import logging
import sys

class Solver:
    def __init__(self, optimizer, loss, stop_epoch, batchsize=100, logfile_path='logfile.log', log_level='WARNING'):
        """
        Create a new solver class
        
        optimizer (torch.optim) : Optimizer for trainig the model
        loss (torch.nn)         : Loss function
        batchsize (int)         : Size of a minibatch of observations
        """
        
        self.optimizer = optimizer
        self.loss = loss
        self.batchsize = batchsize

        self.log_level = log_level.upper()
        self.logfile_path = logfile_path
        self.logger = None
        self.init_logging()
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def init_logging(self):


        log_level_dict = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30,
                          'ERROR': 40, 'CRITICAL': 50}

        if self.log_level not in log_level_dict:
            self.log_level = log_level_dict['WARNING']
        else:
            self.log_level = log_level_dict[self.log_level]

        console_logFormatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        file_logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logging.basicConfig(level=logging.DEBUG, format=console_logFormatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        # file handler always set to INFO - file not capturing DEBUG level
        fh = logging.FileHandler(self.logfile_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(file_logFormatter)
        self.logger.addHandler(fh)

        self.logger.debug('Test Debug')
        self.logger.info('Test Info')
        self.logger.warning('Test Warning')
        self.logger.error('Test Error')
        self.logger.critical('Test Critical')

    def train_offline(self, agent, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train the agents model only with the provided data from a data loader.
        This involves no RL approach as it only optimizes the agent to predict
        the correct action for an observation.

        Inputs:
        agent (agent)   : The agent with the model that should be trained
        train_loader    : train data in torch.utils.data.DataLoader
        val_loader      : val data in torch.utils.data.DataLoader
        num_epochs (int): total number of training epochs
        log_nth (int)   : log training accuracy and loss every nth iteration
        """    
        # TODO: Empty the histories
        
        # Initialize optimizer with the models parameters
        optim = self.optimizer(agent.model.parameters())
        len_trainloader = len(train_loader)
        self.logger.info('Offline Training started')


        for epoch in range(num_epochs):
            # Adapted from dl4cv exercise 3 solver
            for i, (obs, action) in enumerate(train_loader, 1):
                
                inputs, targets = Variable(obs.type(torch.FloatTensor)), Variable(action.type(torch.LongTensor))                
                if agent.model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(async=True)

                optim.zero_grad()                
                outputs = agent.model(inputs)   
                
                loss = self.loss(outputs, targets)
                self.train_loss_history.append(loss.data.cpu().numpy())
                
                t_loss = loss.data.cpu().numpy()[0] # fetch from gpu
                
                loss.backward()                
                optim.step()

                # TODO: Some sort of logging
                self.logger.debug('Epoch %d/%d\t Iter %d/%d\t Loss %f' %
                                  (epoch, num_epochs, i, len_trainloader, t_loss))

                #REMOVE AFTER USE
                if i == 10:
                    break
                
            _, preds = torch.max(outputs, 1)
            train_acc = np.mean((preds == targets).data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            
            # TODO: Logging
            self.logger.info('Train Acc %s' % str(train_acc))

        self.logger.info('Offline Training ended')
            
    def train_online(self, agent, screen, num_epochs=10, log_nth=0):
        """
        Let the agent play in the environment to optimize the strategy
        
        agent (agent)   : The agent that should be trained
        screen (screen) : Screen wrapper around the environment
        num_epochs (int): total number of training epochs
        log_nth (int)   : log training accuracy and loss every nth iteration
        """
        self.logger.info('Online Training started')
        optim = self.optimizer(agent.model.parameters())
        agent.optimize(optim, screen, self.batchsize, num_epochs, self.logger, log_nth)
        self.logger.info('Online Training finished')
       
            
    def play(self, agent, screen):
        """
        Let the agent play to benchmark
        agent (agent)   : The agent that should be trained
        screen (screen) : Screen wrapper around the environment        
        """

        self.logger.info('Wanna play a game? - Game started!')

        agent.play(screen, 2, self.logger)

        self.logger.info('Game ended')
        
            
    
    
    