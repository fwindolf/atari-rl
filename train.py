import argparse
import logging
import torch
import sys

from torch import optim
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.utils.data import DataLoader

from torchvision import transforms, utils

from datetime import datetime

from agents.dqn import DQNAgent
from utils.screen import SpaceInvaderScreen, CartPoleScreen
from utils.agc_data_loader import AGCDataSet
from train.solver import Solver


def main(args):
    
    time = datetime.now()
    
    # Create Logging
    logger = logging.getLogger("Main")
    logger.setLevel(args.log_level.upper())
    
    # Create Game Environment
    screen = None
    logger.info("Creating game environment in a screen")
    if args.game == "spaceinvaders":
        screen = SpaceInvaderScreen()
    elif args.game == "cartpole":
        logger.warn("CartPoleScreen needs an active/configured Display to work")
        screen = CartPoleScreen()
    else:
        logger.error("Screen value not supported!")
        exit(-1)
        
    # Create Dataset
    dataset = None
    dataloader = None
    if args.from_dataset:
        logger.info("Opening dataset")
        dataset = AGCDataSet(args.dataset_dir, args.game, history_len=args.agent_hist,
                             screen=screen) 
        logger.info("Creating dataloader")
        dataloader = DataLoader(dataset, batch_size=args.train_batch, num_workers=4)
        
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
        # TODO different models for dqn
        if args.agent_model == "cnn":
            agent = DQNAgent(screen, history_len=args.agent_hist, mem_size=args.agent_mem, loss=loss)
        elif args.agent_model == "caps":
            raise NotImplemented()
        elif args.agent_model == "linear":
            raise NotImplemented()
        else:
            logger.error("Agent model type not supported!")
            exit(-3)
          
    elif args.aget == "reinforce":
        raise NotImplemented()
    elif args.aget == "a3c":
        raise NotImplemented()
    elif args.aget == "pg":
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
    solver = Solver(optimizer, loss, batchsize=args.train_batch, playtime=args.train_playtime,
                    log_level=args.log_level)
        
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
        agent.initialize(args.agent_init)
        logger.info("Training online")        
        solver.train_online(agent, screen, num_epochs=args.train_epochs, learning_rate=args.train_lr)
        logger.info("Training finished")
        
    ### Benchmarking 
    logger.info("Creating Baseline scores")
    final_best, final_mean, final_dur = solver.play(agent, screen, args.train_playtime)
    logger.info("Benchmark with best score %d (Mean %d in %d frames)" % (final_best, final_mean, final_dur))
    
    # Save model
    agent.model.save("output/%s-%s-%s-%s" % (str(time), args.agent, args.agent_model, str(args.train_lr)))
    
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # Game
    parser.add_argument("-g", "--game", type=str, choices=["spaceinvaders", "cartpole"],
                        default="spaceinvaders", help="The game to train on")

    # Dataset 
    parser.add_argument("-d", "--from-dataset",
                        action="store_true", help="Train from dataset instead of online training")
    parser.add_argument("-dd", "--dataset-dir", type=str, 
                        required="-d" in sys.argv or "--from-dataset" in sys.argv,
                        help="The base directory of the dataset ")

    # Agent
    parser.add_argument("-a", "--agent", type=str, choices=["dqn", "reinforce", "a3c", "pg"],
                        default="dqn", help="The type used of the agent")
    parser.add_argument("-am", "--agent-model", type=str, choices=["cnn", "caps", "linear"],
                        default="cnn", help="The model type used for the agent's model")
    parser.add_argument("-ar", "--agent-mem", type=int, 
                        default=0, help="The agent's replay buffer size")
    parser.add_argument("-ai", "--agent-init", type=int,
                        default=0, help="How many frames the replay buffer is initialized with")
    parser.add_argument("-ah", "--agent-hist", type=int,
                        default=4, help="The number of frames in an observation")
    parser.add_argument("-al", "--agent-loss", type=str, choices=["huber", "l2", "crossentropy"],
                        default="crossentropy", help="The loss used for optimizing the agents model")


    # Training
    parser.add_argument("-tb", "--train-batch", type=int,
                        default=32, help="Batchsize for training/sampling")
    parser.add_argument("-te", "--train-epochs", type=int, 
                        default=1000, help="Numbers of epochs used for training")
    parser.add_argument("-tp", "--train-playtime", type=int,
                        default=10, help="Number of sequences to play to benchmark, ..")
    parser.add_argument("-to", "--train-optim", type=str, choices=["rmsprop", "sgd", "adam"],
                        default="adam", help="The optimizer used during training")
    parser.add_argument("-tl", "--train-lr", type=float,
                        default=0.00025, help="The learning rate for training")

    # Misc
    parser.add_argument("-l", "--log-level", type=str, choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        default="INFO", help="The level used for logging")
    
    # Parse the arguments and provide for main
    args = parser.parse_args()    
    main(args)












