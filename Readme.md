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

## Running

Different Jupyter Notebooks can be created for different agents.