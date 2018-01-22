"""Solver that trains models."""
import torch
from torch.autograd import Variable
import numpy as np
import logging


class Solver:
    """The solver.

    The models can be trained online (with an agent playing inside an
    environment) or offline (using data generated from games).
    """

    def __init__(self,
                 optimizer,
                 loss,
                 batchsize=100,
                 logfile_path='logfile.log',
                 log_level='WARNING'):
        """Create a new Solver class and start logging.

        Args:
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
        
        self.online_scores = []
        self.online_durations = []


    def init_logging(self):
        """Configure logging options."""
        log_level_dict = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}

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

    def train_offline(self, agent, train_loader, val_loader, num_epochs=10):
        """Train the agents model only with the provided data from a data loader.

        This involves no RL approach as it only optimizes the agent to predict
        the correct action for an observation.

        Args:
            agent (agent)   : The agent with the model that should be trained
            train_loader    : train data in torch.utils.data.DataLoader
            val_loader      : val data in torch.utils.data.DataLoader
            num_epochs (int): total number of training epochs
            log_nth (int)   : log training accuracy and loss every nth iterat°
        """
        # delete history
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        
        self.data_loss_history = []

        # Initialize optimizer with the models parameters
        optim = self.optimizer(agent.model.parameters())
        self.logger.info('Offline Training started with %d iterations in %d epochs' % 
                         (len(train_loader), num_epochs))

        for epoch in range(num_epochs):
            correct_train = 0
            correct_val = 0
            
            for i, data in enumerate(train_loader, 1):
                obs, action, _, _, _ = data

                inputs = Variable(obs.type(torch.FloatTensor))
                targets = Variable(action.type(torch.LongTensor))

                if agent.model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(async=True)

                optim.zero_grad()
                outputs = agent.model(inputs)

                loss = self.loss(outputs, targets)
                self.train_loss_history.append(loss.data.cpu().numpy())

                t_loss = loss.data.cpu().numpy()[0]  # fetch from gpu

                loss.backward()
                optim.step()

                _, pred = torch.max(outputs, 1)     # compute prediction

                if np.sum((pred == targets).data.cpu().numpy()):
                    correct_train += 1

                # log to console: epoch, iteration, loss, prediction, target
                self.logger.debug('Epoch %d/%d\t Iter %d/%d\t Loss %f' % 
                                  (epoch, num_epochs, i, len(train_loader), t_loss))

                self.logger.debug('\t\tPrediction: %s \t Target: %s' %
                                  (str(pred.data.cpu().numpy()),
                                   str(targets.data.cpu().numpy())))
            
            for i, data in enumerate(train_loader, 1):
                obs, action, _, _, _ = data
                
                inputs = Variable(obs.type(torch.FloatTensor), volatile=True)
                targets = Variable(action.type(torch.LongTensor), volatile=True)

                if agent.model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(async=True)

                outputs = agent.model(inputs)
                
                loss = self.loss(outputs, targets)
                self.val_loss_history.append(loss.data.cpu().numpy())
                
                _, pred = torch.max(outputs, 1)     # compute prediction
                
                if np.sum((pred == targets).data.cpu().numpy()):
                    correct_val += 1
                    
            train_acc = correct_train / len(train_loader)
            val_acc = correct_val / len(val_loader)
            
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(train_acc)
                
            self.logger.info('Epoch %d \t Train/Val Acc %d/%d %%' % 
                             (epoch, train_acc * 100, val_acc * 100))

        self.logger.info('Offline Training ended')
        
        
    def train_dataset(self, agent, screen, data_loader, num_epochs=10, learning_rate=0.001):
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
                loss = agent.optimize(optim, screen, self.batchsize, data=data).cpu().numpy()[0]
                self.data_loss_history.append(loss)
                
            self.logger.info('Epoch %d/%d: Mean loss %f' % 
                             (epoch, num_epochs, np.mean(self.data_loss_history[-len(data_loader):])))
                
            # benchmark once per epoch
            best, mean, dur = self.play(agent, screen, num_sequences=4)
            self.logger.info('Epoch %d/%d: Mean score %d with %d frames' % 
                             (epoch, num_epochs, mean, dur))
                
        self.logger.info('Training from Dataset finished')                

    def train_online(self, agent, screen, num_epochs=1000, learning_rate=1e-4):
        """
        Let the agent play in the environment to optimize the strategy
        Args:
            agent (agent) : The agent that should be trained
            screen (screen) : Screen wrapper around the environment
            num_epochs (int) : Total number of training epochs
            learning_rate (float) : The learning rate with which the model is udpated
        """
        # reset online histories
        self.online_scores = []
        self.online_durations = []

        optim = self.optimizer(agent.model.parameters(), learning_rate)

        # create a temporary nework to increase stability
        tmp_model = deepcopy(agent.model)
        tmp_model.load_state_dict(agent.model.state_dict())
        update_count = 0

        # move to gpu if applicable
        if agent.model.is_cuda:
            tmp_model.cuda()

        self.logger.info('Online Training started')

        for i in range(num_epochs):
            duration = 0
            score = 0
            done = False
            losses = []
            while not done:
                # Make a step in the environment
                _, reward, _, done = agent.step(screen, i, num_epochs, save=True)
                score += reward
                duration += 1

                self.logger.debug("Epoch %d/%d - Iteration %d - Score: %d" % (i, num_epochs, duration, score))

                # optimize model by sampling from replay memory
                
                loss = agent.optimize(optim, screen, self.batchsize, tmp_model)
                losses.append(loss)

                self.logger.debug("Epoch %d/%d - Iteration %d - Loss: %f" % (i, num_epochs, duration, loss))

                update_count += 1

                if update_count % update_step == 0:
                    agent.model.load_state_dict(tmp_model.state_dict())
                    update_count = 0

            self.logger.info('Epoch %d/%d - Score %d with Duration %d (Loss %f)' %
                                 (i, num_epochs, score, duration, np.mean(losses)))

        self.logger.info('Online Training finished')


    def play(self, agent, screen, num_sequences=1, epoch=1, max_epoch=1, save=False):
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
            score, duration = agent.play(screen, epoch, max_epoch, save=save)
            scores.append(score)
            durations.append(duration)

            self.logger.debug('Score %d after %d frames' % (score, duration))

            if score > best_score:
                self.logger.debug('New Highscore %d' % (score))
                best_score = score

        self.logger.debug('Game ended')

        return best_score, np.mean(scores), np.mean(durations)
