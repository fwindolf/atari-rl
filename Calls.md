# DQN

## Cartpole-Basic

Linear model (1024 hiddens) with DQN algorithm (0.8 gamma) over 400 epochs with 10000 frames in replay memory.
```bash
python train.py -g cartpole-basic -a dqn -am linear -ar 10000 -ai 5000 -ah 1 -as -al huber -ag 0.8 -an 1024 -te 200 -tb 32 -tl 0.001 -to adam  
```

## Cartpole

CNN model with DQN algorithm (0.99 gamma) over 400 epochs with 100000 frames in replay memory.

```bash
python train.py -g cartpole-basic -a dqn -am cnn -ar 100000 -ai 50000 -ah 1 -as -al huber -te 400 -to adam -tb 64 -tl 0.00025 -ag 0.99
```

The same works for a Capsnet model 

```bash
python train.py -g cartpole-basic -a dqn -am cnn -ar 100000 -ai 50000 -ah 1 -as -al huber -te 400 -to adam -tb 64 -tl 0.00025 -ag 0.99
```



# REINFORCE

## Cartpole-Basic

```bash
python train.py -g cartpole-basic -a reinforce -am discrete -ah 1 -as -al huber -te 400 -to adam -tb 100 -tl 0.001 -ag 0.99
```