This repository is for the class project of DL4CV (Winter 2017)

# About

The OpenAI gym environment is installed as a submodule in gym. 

Environment interaction is wrapped in screen, which simplifies the generation of new frames (of the right shape and ROI). The caller can input actions and will get ouputs (frame, reward, done_flag) and is able to retrieve the current and last frame.

Different agents `/agents` (with different strategies) can aggregate different models `/models` without having to redefine basic functionality. Agent memory (eg. replaybuffers) can be found in `/utils`, where also the screen wrapper lie.

# Installation

Tested on Ubuntu Bash (Windows 10) with conda (python3.6) already installed.

```
git clone ...

git submodule update --init

# Follow the instructions for installing OpenAI gym

cd gym

sudo apt-get install cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libsdl2-dev
pip install -e '.[atari]'

# Pytorch (with Cuda Support)
conda install -c soumith cuda80
conda install -c peterjc123 pytorch

# Jupyter
conda install jupyter

```

In order to be able to see screens, vcXsrv has to be installed on Windows 10 and the default output has to be suppressed `export DISPLAY=:0`.

## Usage

Run `train.py` with the following options:

```
usage: train.py [-h]
                [-g {spaceinvaders,pinball,mspacman,cartpole,cartpole-basic}]
                [-d] [-dd DATASET_DIR] [-a {dqn,reinforce,a3c,pg}]
                [-am {cnn,caps,linear,continuous,discrete,dueling,a2c}]
                [-ar AGENT_MEM] [-ai AGENT_INIT] [-ah AGENT_HIST]
                [-al {huber,l2,crossentropy}] [-as] [-ag AGENT_GAMMA]
                [-an AGENT_NETWORK_HIDDEN] [-tb TRAIN_BATCH]
                [-te TRAIN_EPOCHS] [-tp TRAIN_PLAYTIME]
                [-to {rmsprop,sgd,adam}] [-tl TRAIN_LR]
                [-l {DEBUG,INFO,WARN,ERROR}] [-lf LOG_FILE] [-hs]
                [-hlr HYPERPARAMETER_LR [HYPERPARAMETER_LR ...]]
                [-hg HYPERPARAMETER_GAMMA [HYPERPARAMETER_GAMMA ...]]

optional arguments:
  -h, --help            show this help message and exit
  -g {spaceinvaders,pinball,mspacman,cartpole,cartpole-basic}, --game {spaceinvaders,pinball,mspacman,cartpole,cartpole-basic}
                        The game to train on
  -d, --from-dataset    Train from dataset instead of online training
  -dd DATASET_DIR, --dataset-dir DATASET_DIR
                        The base directory of the dataset
  -a {dqn,reinforce,a3c,pg}, --agent {dqn,reinforce,a3c,pg}
                        The type used of the agent
  -am {cnn,caps,linear,continuous,discrete,dueling,a2c}, --agent-model {cnn,caps,linear,continuous,discrete,dueling,a2c}
                        The model type used for the agent's model
  -ar AGENT_MEM, --agent-mem AGENT_MEM
                        The agent's replay buffer size
  -ai AGENT_INIT, --agent-init AGENT_INIT
                        How many frames the replay buffer is initialized with
  -ah AGENT_HIST, --agent-hist AGENT_HIST
                        The number of frames in an observation
  -al {huber,l2,crossentropy}, --agent-loss {huber,l2,crossentropy}
                        The loss used for optimizing the agents model
  -as, --agent-simple   Use a simple replay memory (This is default, currently the faster buffer has a bug!)
  -ag AGENT_GAMMA, --agent-gamma AGENT_GAMMA
                        The gamma parameter for DQN
  -an AGENT_NETWORK_HIDDEN, --agent-network-hidden AGENT_NETWORK_HIDDEN
                        The number of hidden units for a model
  -tb TRAIN_BATCH, --train-batch TRAIN_BATCH
                        Batchsize for training/sampling
  -te TRAIN_EPOCHS, --train-epochs TRAIN_EPOCHS
                        Numbers of epochs used for training
  -tp TRAIN_PLAYTIME, --train-playtime TRAIN_PLAYTIME
                        Number of sequences to play to benchmark, ..
  -to {rmsprop,sgd,adam}, --train-optim {rmsprop,sgd,adam}
                        The optimizer used during training
  -tl TRAIN_LR, --train-lr TRAIN_LR
                        The learning rate for training
  -l {DEBUG,INFO,WARN,ERROR}, --log-level {DEBUG,INFO,WARN,ERROR}
                        The level used for logging
  -lf LOG_FILE, --log-file LOG_FILE
                        The file used for logging
  -hs, --hyperparameter-search
                        Search among different hyper parameters
  -hlr HYPERPARAMETER_LR [HYPERPARAMETER_LR ...], --hyperparameter-lr HYPERPARAMETER_LR [HYPERPARAMETER_LR ...]
                        Search different parameters for learning rate
  -hg HYPERPARAMETER_GAMMA [HYPERPARAMETER_GAMMA ...], --hyperparameter-gamma HYPERPARAMETER_GAMMA [HYPERPARAMETER_GAMMA ...]
                        Search different parameters for gamma
```

Some models will only work with certain options. For example, running Cartpole-Basic with a CNN will obviously not work. CNN models generally work for Screens that produce images with dimensions of (80, 80).

Not all options might be implemented/working!