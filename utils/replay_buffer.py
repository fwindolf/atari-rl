import numpy as np
import torch
from utils.transition import Transition

class ReplayBuffer():
    """
    ReplayMemory implementation
    adapted from: https://github.com/transedward/pytorch-dqn/blob/master/utils/replay_buffer.py    
    """    
    def __init__(self, size, history_len, screen):
        """
        Replay Buffer implementation
        Stores the frames as int8. This keeps the memory footprint low.
        
        size (int):        maximum numbers of transitions stored. 
        history_len (int): number of frames for each observation        
        screen           : The reference to the screen that generates experiences
        """
        
        self.size = size
        self.history_len = history_len
        self.screen = screen
        
        self.next_idx = 0
        self.num_transitions = 0
       
        self.frames     = None # dont know frame shape yet
        self.action  = np.empty([self.size], dtype=np.int32)
        self.reward  = np.empty([self.size], dtype=np.float32)
        self.done    = np.empty([self.size], dtype=np.bool)
        
    def get_history_len(self):
        """
        Return the numer of frames for each observation
        """
        return self.history_len
    
    def initialize_dataset(self, num_replays, dataset):
        """
        Initilize the replay memory with <num_replays> experiences
        
        Uses random samples from the datatset to generate experiences
        
        num_replays (int): How many experiences should be generated
        dataset          : The dataset from which sequences are taken
        return           : The latest observation
        """
        
        initialized = False
        while(not initialized):
            # sample the a random frame (= start of our sequence)
            d_idx = np.random.randint(len(dataset))
            
            for t in range(self.history_len):
                transition = dataset.raw(d_idx)
                
                obs = self.screen.output(transition.observation)
                reward = transition.reward
                done = transition.done
                action = transition.action
            
                assert(obs.dtype.name == 'uint8') # uses less memory
                
                idx = self.store_frame(obs) 
                self.store_effect(idx, action, reward, done)
                
                d_idx += 1 # next frame in that sequence
        
                if done or transition.next_observation is None:
                    break # start new sequence
                    
                if self.num_transitions >= num_replays:
                    initialized = True
                    break
                    
        return obs
    
    def initialize_playing(self, num_replays, agent):
        """
        Initialize the replay memory with <num_replays> experiences
        
        Let the agent play to generate experiences
        
        num_replays (int): How many experiences should be generated
        agent            : The agent that plays in the environment
        return           : The latest observation
        """
        
        initialized = False
        while(not initialized):
            # run sequences
            obs = self.screen.reset()
            idx = self.store_frame(obs)
            
            for t in range(self.history_len):                
                action = agent.next_action(obs)
                obs, reward, done, _ = self.screen.input(action)
                
                assert(obs.dtype.name == 'uint8') # uses less memory
                
                idx = self.store_frame(obs)
                self.store_effect(idx, action, reward, done)  
                
                if done:
                    break # start new sequence
                    
                if self.num_transitions >= num_replays:
                    initialized = True
                    break
                
        return obs          
        
    def __encode_observation(self, idx):
        """
        Aggregates frames to observations of history_len (padded with 0-frames if necessary)                
        idx (int) : The index of the observation that should be encoded.
        return    : The observation consisting out of history_len frames with float datatype
        """
        
        assert len(self.frames.shape) == 3 # (size, h, w)        
        observation = np.empty([self.history_len] + list(self.frames.shape[1:]))
        
        end_idx = idx + 1
        start_idx = end_idx - self.history_len
        
        # are there enough frames in buffer and buffer not full?
        if start_idx < 0 and self.num_transitions != self.size:
            start_idx = 0
            
        # only use one episode
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        
        # number of frames that have to be padded with 0s
        missing_frames = self.history_len - (end_idx - start_idx)
        
        # do we need to generate empty frames ?
        for idx in range(0, missing_frames):
            observation[idx] = self.screen.output_float(np.zeros(self.frames.shape[1:]))
            
        for idx in range(missing_frames, self.history_len):
            observation[idx] = self.screen.output_float(self.frames[(start_idx + idx) % self.size])    
            
        return observation
        
    def can_sample(self, batchsize):
        """
        Is it possible to sample <batchsize> different transitions from the buffer
        batchsize (int): The number of transitions to sample 
        return         : True if there are enough transitions available
        """        
        return (batchsize < self.num_transitions)
        
        
    def sample(self, batchsize):
        """
        Sample <batchsize> different transitions.        
        
        batchsize(int): The number of transitions to sample
        return        : Tuple containing observations, actions, rewards and next_observations with done_mask indicating which actions have ended the episode
        """
        assert self.can_sample(batchsize)
        
        # generate indices 
        idxs = np.random.choice(range(self.num_transitions), batchsize, replace=False)
        
        # generate samples with history_len size that encode until done state
        obs = np.concatenate([self.__encode_observation(idx)[np.newaxis, :] for idx in idxs], 0) #(batchsize, history_len, h, w)
        act = np.asarray(self.action[idxs], dtype=np.int32)
        rew = np.asarray(self.reward[idxs], dtype=np.float32)
        next_obs = np.concatenate([self.__encode_observation(idx + 1)[np.newaxis, :] for idx in idxs], 0) 
        done = np.array([1 if self.done[idx] else 0 for idx in idxs], dtype=np.uint8)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs)
        act = torch.from_numpy(act)
        rew = torch.from_numpy(rew)
        next_obs = torch.FloatTensor(next_obs)
        done = torch.ByteTensor(done)
        
        return (obs, act, rew, done, next_obs)
    
    def store_frame(self, frame):
        """
        Store a new frame in the buffer, overwriting old frames if the buffer is full
        
        frame (np.array): Uint8 array of (h,w,c) 
        return          : Index of the stored frame
        """
        assert(frame.dtype == np.uint8)
        
        # check if we need to initialize frames (we dont know frame shape during _init_)
        if self.frames is None: 
            self.frames = np.empty([self.size] + list(frame.shape), dtype='uint8')

        self.frames[self.next_idx] = frame
        at_idx = self.next_idx
        
        self.next_idx = (self.next_idx + 1) % self.size # circular buffer
        self.num_transitions = min(self.size, self.num_transitions + 1)
        
        return at_idx
    
    def store_effect(self, idx, action, reward, done):
        """
        Store a new effect in the buffer (action, reward, done), overwriting old effects if the buffer is full.
        store_frame and store_effect seperated to be able to call encode_recent_observation in between.
        
        idx (int)      : Index of the corresponding frame that was already stored
        action (int)   : Action that was performed to observe this frame
        reward (float) : Reward of the taken action
        done (bool)    : True if the episode was finished after performing that action
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
        
        
class SimpleReplayBuffer():
    def __init__(self, size):
        self.size = size
        self.memory = np.empty(size, dtype=tuple)
        self.next_idx = 0
        self.num_transitions = 0
        
    def store_frame(self, frame):
        self.memory[self.next_idx] = (frame.cpu())
        return self.next_idx
        
    def store_effect(self, idx, action, reward, next_frame):
        frame = self.memory[idx]
        self.memory[idx] = (frame, action.cpu(), reward.cpu(), next_frame.cpu())
        self.num_transitions = min(self.num_transitions + 1, self.size)
        self.next_idx += 1
        if self.next_idx >= self.size:
            self.next_idx = 0
        
    def can_sample(self, batchsize):
        return self.num_transitions >= batchsize
    
    def sample(self, batchsize):
        transitions = np.random.choice(self.memory[:self.num_transitions], batchsize, replace=False)        
        frames, actions, rewards, next_frames = zip(*transitions)   
        
        return (frames, actions, rewards, next_frames)
    
    def get_history_len(self):
        return 1
    
    def __len__(self):
        return self.num_transitions
        