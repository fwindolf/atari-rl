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

    def init_logging(self):
        """Configure logging options."""
        log_level_dict = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30,
                          'ERROR': 40, 'CRITICAL': 50}

        if self.log_level not in log_level_dict:
            self.log_level = log_level_dict['WARNING']
        else:
            self.log_level = log_level_dict[self.log_level]

        console_logFormatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
        file_logFormatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        logging.basicConfig(level=logging.DEBUG, format=console_logFormatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        # file handler always set to INFO - file not capturing DEBUG level
        fh = logging.FileHandler(self.logfile_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(file_logFormatter)
        self.logger.addHandler(fh)

    def train_offline(self,
                      agent,
                      train_loader,
                      val_loader,
                      num_epochs=10,
                      log_nth=0):
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

        # Initialize optimizer with the models parameters
        optim = self.optimizer(agent.model.parameters())
        self.logger.info(
            'Offline Training started with %d iterations in %d epochs' %
            (len(train_loader), num_epochs)
        )

        for epoch in range(num_epochs):
            # Adapted from dl4cv exercise 3 solver
            no_correct_preds = 0
            for i, (obs, action) in enumerate(train_loader, 1):

                inputs, targets = Variable(obs.type(torch.FloatTensor)), \
                    Variable(action.type(torch.LongTensor))

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
                    no_correct_preds += 1

                # log to console: epoch, iteration, loss, prediction, target
                self.logger.debug(
                    'Epoch %d/%d\t Iter %d/%d\t Loss %f' %
                    (epoch, num_epochs, i, len(train_loader), t_loss)
                )

                self.logger.debug(
                    '\t\tPrediction: %s \t Target: %s' %
                    (str(pred.data.cpu().numpy()),
                     str(targets.data.cpu().numpy()))
                )

            train_acc = no_correct_preds / len(train_loader)
            self.train_acc_history.append(train_acc)

            self.logger.info(
                'Epoch %d \t Train Acc %s %%' %
                (epoch, str(train_acc*100))
            )

        self.logger.info('Offline Training ended')

    def train_online(self,
                     agent,
                     screen,
                     num_epochs=10,
                     learning_rate=0.01,
                     log_nth=0,
                     weight_decay=0):
        """Let the agent play in the environment to optimize the strategy.

        Args:
            agent (agent)   : The agent that should be trained
            screen (screen) : Screen wrapper around the environment
            num_epochs (int): total number of training epochs
            learning_rate (float) : the learning rate
            log_nth (int)   : log training accuracy and loss every nth iterat°
        """
        self.logger.info('Online Training started')
        self.logger.info('Memory.next_idx : '+str(agent.memory.next_idx))

        optim = self.optimizer(
            agent.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.train_loss_history = agent.optimize(
            optim,
            screen,
            self.batchsize,
            num_epochs,
            self.logger,
            log_nth
        )

        self.logger.info('Online Training finished')

    def play(self, agent, screen, num_sequences=100, save=False):
        """Let the agent play for benchmarking.

        Args:
            agent (agent)   : The agent that should be trained
            screen (screen) : Screen wrapper around the environment
        """
        self.logger.info('Wanna play a game? - Game started!')

        best_score = 0
        scores = []
        durations = []

        for i in range(num_sequences):
            score, duration = agent.play(screen, save=save)
            scores.append(score)
            durations.append(duration)
            self.logger.debug('Score %d after %d frames' % (score, duration))

            if score > best_score:
                self.logger.debug('New Highscore %d' % (score))
                best_score = score

            if i % int(num_sequences / 10) == 0:
                self.logger.info(
                    '%d/%d Highscore is %d' %
                    (i, num_sequences, best_score)
                )

        self.logger.info('Mean score %f' % (np.mean(scores)))
        self.logger.info('Mean duration %f' % (np.mean(duration)))

        self.logger.info(
            'Highscore %d out of %d sequences' % (best_score, num_sequences)
        )
        self.logger.info('Game ended')

        return best_score, np.mean(scores), np.mean(duration)
