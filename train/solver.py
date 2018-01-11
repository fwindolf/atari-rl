import torch
from torch.autograd import Variable
import numpy as np

class Solver:
    def __init__(self, optimizer, loss, stop_epoch, batchsize=100):
        """
        Create a new solver class
        
        optimizer (torch.optim) : Optimizer for trainig the model
        loss (torch.nn)         : Loss function
        batchsize (int)         : Size of a minibatch of observations
        """
        
        self.optimizer = optimizer
        self.loss = loss
        self.batchsize = batchsize
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
    
    def train_offline(self, agent, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train only the agents model with the provided data from a data loader.

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
                
            _, preds = torch.max(outputs, 1)
            train_acc = np.mean((preds == targets).data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            
            # TODO: Logging
            
    def train_online(self, agent, screen, num_epochs=10, log_nth=0):
        """
        Let the agent play in the environment to optimize the strategy
        
        agent (agent)   : The agent that should be trained
        screen (screen) : Screen wrapper around the environment
        num_epochs (int): total number of training epochs
        log_nth (int)   : log training accuracy and loss every nth iteration
        """       
        optim = self.optimizer(agent.model.parameters())
        agent.optimize(optim, screen, self.batchsize, num_epochs, log_nth)
       
            
    def play(self, agent, screen):
        """
        Let the agent play to benchmark
        agent (agent)   : The agent that should be trained
        screen (screen) : Screen wrapper around the environment        
        """
        agent.play(screen)
        
        
            
    
    
    