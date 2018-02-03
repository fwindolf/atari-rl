import argparse
import logging
import torch
import sys
import gym

from torch import optim
from torch.nn import CrossEntropyLoss, SmoothL1Loss, MSELoss
from torch.utils.data import DataLoader

from torchvision import transforms, utils

from datetime import datetime


from utils.screen import SpaceInvaderScreen, CartPoleScreen, CartPoleBasic, PinballScreen, PacmanScreen
from utils.agc_data_loader import AGCDataSet
from utils.replay_buffer import SimpleReplayBuffer, ReplayBuffer

from agents.dqn import DQNAgent
from agents.reinforce import REINFORCE

from models.dqn import DQN, DQNLinear, DQNCapsNet

from train.solver import Solver


# Choices
games = ["spaceinvaders", "pinball", "mspacman", "cartpole", "cartpole-basic"]
agents = ["dqn", "reinforce", "a3c", "pg"]
models = ["cnn", "caps", "linear", "continuous", "discrete", "dueling", "a2c"]
optimizers = ["rmsprop", "sgd", "adam"]
losses = ["huber", "l2", "crossentropy"]
log_levels = ["DEBUG", "INFO", "WARN", "ERROR"]

def train(args):  
        
    # Create Logging
    logger = logging.getLogger("Main")
    logger.setLevel(args.log_level.upper())
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        fh.setLevel(args.log_level.upper())
        logger.addHandler(fh)
    
    time = datetime.now()
    
    logger.debug("Time  : %s" % str(time))
    logger.debug("Params: %s" % str(args))

    
    # Create Game Environment
    screen = None
    logger.info("Creating game environment in a screen")
    if args.game == "spaceinvaders":
        screen = SpaceInvaderScreen()
        logger.info("SpaceInvaderScreen created")
    elif args.game == "pinball":
        screen = PinballScreen()
        logger.info("PinballScreen created")
    elif args.game == "mspacman":
        screen = PacmanScreen()
        logger.info("PacmanScreen created")
    elif args.game == "cartpole":
        logger.warn("CartPoleScreen needs an active/configured Display to work")
        screen = CartPoleScreen()
        logger.info("CartPoleScreen created")
    elif args.game == "cartpole-basic":        
        logger.warn("CartPoleBasic does not provide any images!")
        screen = CartPoleBasic()
        logger.info("CartPoleBasic created")
    else:
        logger.error("Screen value not supported!")
        exit(-1)
        
    # Create Dataset
    dataset = None
    dataloader = None
    if args.from_dataset:
        logger.info("Opening dataset")
        dataset = AGCDataSet(args.dataset_dir, args.game, history_len=args.agent_hist,
                             screen=screen, gamma=args.agent_gamma) 
        logger.info("Dataset created")
        logger.info("Creating dataloader")
        dataloader = DataLoader(dataset, batch_size=args.train_batch, num_workers=4)
        logger.info("Dataloader created")
        
    # Create Loss    
    loss = None
    if args.agent_loss == "crossentropy":
        loss = CrossEntropyLoss()
    elif args.agent_loss == "l2":
        loss = MSELoss()
    elif args.agent_loss == "huber":
        loss = SmoothL1Loss()
    else:
        logger.error("Loss not supported!")
        exit(-2)
    
    # Create Agent
    agent = None
    logger.info("Create agent")
    
    if args.agent == "dqn":
        # Replay Buffer
        replay_mem = None
        if args.agent_simple:
            if args.agent_hist > 1:
                logger.error("Simple Replay Buffer can only be used with single frame observations")
                exit(-3)
                
            replay_mem = SimpleReplayBuffer(args.agent_mem)
            logger.info("SimpleReplayBuffer created")
        else:
            replay_mem = ReplayBuffer(args.agent_mem, args.agent_hist, screen)
            logger.info("ReplayBuffer created")
        
        # Model
        model = None
        if args.agent_model == "cnn":
            model = DQN(args.agent_hist * screen.get_shape()[0], screen.get_actions())
            logger.info("DQN CNN model created")
        elif args.agent_model == "caps":
            model = DQNCapsNet(args.agent_hist * screen.get_shape()[0], screen.get_actions())
            logger.info("DQN CapsNet model created")
        elif args.agent_model == "linear":
            model = DQNLinear(screen.get_dim(), screen.get_actions(), args.agent_network_hidden)
            logger.info("DQN Linear model created")
        else:
            logger.error("Agent model type not supported!")
            exit(-3)
            
        agent = DQNAgent(screen, history_len=args.agent_hist, loss=loss,
                         memory=replay_mem, model=model, gamma=args.agent_gamma)      
        logger.info("Agent created (Gamma=%f, HistoryLen=%d)" % (args.agent_gamma, args.agent_hist))
        
    elif args.agent == "reinforce":
        if args.agent_model == "discrete":
            if type(screen.env.unwrapped.action_space) != gym.spaces.discrete.Discrete:
                logger.warn("Agent model type is discrete but environment seems to be not")
            
            agent = REINFORCE(screen, args.agent_hist, args.agent_network_hidden, args.agent_gamma, False)            
            logger.info("Reinforce Discrete agent created")
            
        elif args.agent_model == "continuous":
            if type(screen.env.unwrapped.action_space) == gym.spaces.discrete.Discrete:
                logger.warn("Agent model type is continuous but environment seems to be discrete")
            
            agent = REINFORCE(screen, args.agent_hist, args.agent_network_hidden, args.agent_gamma, True)            
            logger.info("Reinforce Continuous agent created")
            
        else:
            logger.error("Agent model type not supported!")
            exit(-3)
        
        logger.info("Agent created (Gamma=%f, HistoryLen=%d)" % (args.agent_gamma, args.agent_hist))
        
    elif args.agent == "a3c":
        raise NotImplemented()
    elif args.agent == "pg":
        raise NotImplemented()
    else:
        logger.error("Agent type not supported!")
        exit(-3)
        
    # Create Optimizer
    optimizer = None
    if args.train_optim == "rmsprop":
        optimizer = optim.RMSprop
    elif args.train_optim == "sgd":
        optimizer = optim.SGD
    elif args.train_optim == "adam":
        optimizer = optim.Adam
    else:
        logger.error("Agents optimizer not supported!")
        exit(-4) 
    
    # Create Solver
    logger.info("Creating solver")
    solver = Solver(optimizer, batchsize=args.train_batch, log_level=args.log_level)
        
    ### Benchmarking 
    logger.info("Creating Baseline scores")
    intial_best, initial_mean, intial_dur = solver.play(agent, screen, args.train_playtime)
    logger.info("Baseline with best score %d (Mean %d in %d frames)" % (intial_best, initial_mean, intial_dur))
    
    ### Training
    if args.from_dataset:
        logger.info("Training from dataset")      
        solver.train_dataset(agent, screen, dataloader, num_epochs=args.train_epochs, 
                             learning_rate=args.train_lr)
        logger.info("Training finished")
        
    else:
        logger.info("Initializing Memory")
        agent.initialize(screen, args.agent_init)
        logger.info("Training online")        
        solver.train_online(agent, screen, num_epochs=args.train_epochs, learning_rate=args.train_lr)
        logger.info("Training finished")
        
    ### Benchmarking 
    logger.info("Creating Benchmark scores")
    final_best, final_mean, final_dur = solver.play(agent, screen, args.train_playtime, render=True)
    logger.info("Benchmark with best score %d (Mean %d in %d frames)" % (final_best, final_mean, final_dur))
    
    # Save model
    agent.model.save("output/%s-%s-%s-%s-%s" % (str(time), args.agent, args.agent_model, 
                                                str(args.train_lr), str(args.agent_gamma)))

    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # Game
    parser.add_argument("-g", "--game", type=str, choices = games,
                        default="spaceinvaders", help="The game to train on")

    # Dataset 
    parser.add_argument("-d", "--from-dataset",
                        action="store_true", help="Train from dataset instead of online training")
    parser.add_argument("-dd", "--dataset-dir", type=str, 
                        required="-d" in sys.argv or "--from-dataset" in sys.argv,
                        help="The base directory of the dataset ")

    # Agent
    parser.add_argument("-a", "--agent", type=str, choices = agents,
                        default="dqn", help="The type used of the agent")
    parser.add_argument("-am", "--agent-model", type=str, choices = models,
                        default="cnn", help="The model type used for the agent's model")
    parser.add_argument("-ar", "--agent-mem", type=int, 
                        default=0, help="The agent's replay buffer size")
    parser.add_argument("-ai", "--agent-init", type=int,
                        default=0, help="How many frames the replay buffer is initialized with")
    parser.add_argument("-ah", "--agent-hist", type=int,
                        default=4, help="The number of frames in an observation")
    parser.add_argument("-al", "--agent-loss", type=str, choices = losses,
                        default="huber", help="The loss used for optimizing the agents model")
    parser.add_argument("-as", "--agent-simple", 
                        action="store_true", help="Use a simple replay memory")
    
    # Optional
    parser.add_argument("-ag", "--agent-gamma", type=float, 
                        default=0.99, help="The gamma parameter for DQN")
    parser.add_argument("-an", "--agent-network-hidden", type=int, 
                        default=256, help="The number of hidden units for a model")


    # Training
    parser.add_argument("-tb", "--train-batch", type=int,
                        default=32, help="Batchsize for training/sampling")
    parser.add_argument("-te", "--train-epochs", type=int, 
                        default=1000, help="Numbers of epochs used for training")
    parser.add_argument("-tp", "--train-playtime", type=int,
                        default=10, help="Number of sequences to play to benchmark, ..")
    parser.add_argument("-to", "--train-optim", type=str, choices = optimizers,
                        default="adam", help="The optimizer used during training")
    parser.add_argument("-tl", "--train-lr", type=float,
                        default=0.00025, help="The learning rate for training")

    # Misc
    parser.add_argument("-l", "--log-level", type=str, choices = log_levels,
                        default="INFO", help="The level used for logging")
    parser.add_argument("-lf", "--log-file", type=str,
                        help="The file used for logging")
    
    
    # Hyperparameter Search
    parser.add_argument("-hs", "--hyperparameter-search",
                        action="store_true", help="Search among different hyper parameters")
    parser.add_argument("-hlr", "--hyperparameter-lr", type=float, nargs='+',  
                        required="-hs" in sys.argv or "--hyperparameter-search" in sys.argv,
                        help="Search different parameters for learning rate")
    parser.add_argument("-hg", "--hyperparameter-gamma", type=float, nargs='+',  
                        required="-hs" in sys.argv or "--hyperparameter-search" in sys.argv,
                        help="Search different parameters for gamma")
    
    
    # Parse the arguments and provide for main
    args = parser.parse_args()    
    
    if args.hyperparameter_search:
        for lr in args.hyperparameter_lr:
            for gamma in args.hyperparameter_gamma:
                args.train_lr = lr
                args.agent_gamma = gamma
                train(args)                
    else:         
        train(args)












