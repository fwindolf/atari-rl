import numpy as np

class ReplayBuffer():
    """
    ReplayMemory implementation
    adapted from: https://github.com/transedward/pytorch-dqn/blob/master/utils/replay_buffer.py    
    """    
    def __init__(self, size, history_len):
        """
        Replay Buffer implementation
        Stores the frames as int8. This keeps the memory footprint low.
        
        size (int):        maximum numbers of transitions stored. 
        history_len (int): number of memories retried for each observation
        """
        
        self.size = size
        self.history_len = history_len
        
        self.next_idx = 0
        self.num_transitions = 0
        
        self.obs      = np.empty([self.size], dtype=np.uint8)
        self.action   = np.empty([self.size], dtype=np.int32)
        self.reward   = np.empty([self.size], dtype=np.float32)
        self.done     = np.empty([self.size], dtype=np.bool)
        
    def initialize(self, number_of_replays, env):
        """
        Initilize the replay memory with <number_of_replays> experiences
        
        Internally resets the environment to generate a fresh observation
        
        number_of_replays (int): How many experiences should be generated
        env (gym.env)          : The reference to the environment that the experiences should be generated
        return                 : The latest observation
        """
        raise NotImplemented()
        
        obs = env.reset()
        
        for i in range(number_of_replays):
            # Do an action and save to this buffer
            action = env.action_space.sample()
            new_obs, reward, done, _ = env.step(action)
            
            # TODO: How to incorporate the preprocessing strategy 
            obs_image = None
            new_obs_image = None 
                        
            idx = self.store_frame(obs_image)
            self.encode_recent_observation()
            self.store_effect(action, reward, done)
            
            new_obs = obs
            
        return new_obs 
        
        
    def __encode_observation(self, idx):
        """
        Encode observation in a defined way.
        Pad with 0-frames to achieve history_len.
        Use a maximum of one episode and encode the frames in (h, w, c)
        
        idx (int) : The index of the observation that should be encoded.
        """
        end_idx = idx + 1
        start_idx = end_idx - self.history_len
        
        # are we using the right format of observations?
        if len(self.obs.shape) == 2:
            return self.obs[end_idx - 1]
        
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
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0])] * missing_frames
            frames += [self.obs[idx % self.size] for idx in range(start_idx, end_idx)]
            return np.concatenate(frames, 0)
        else:
            h, w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, h, w)
        
        
    def can_sample(self, batchsize):
        """
        Is it possible to sample <batchsize> different transitions from the buffer
        batchsize (int): The number of transitions to sample 
        return         : True if there are enough transitions available
        """        
        return (batchsize + 1 <= self.num_transitions)
        
        
    def sample(self, batchsize):
        """
        Sample <batchsize> different transitions.        
        
        batchsize(int): The number of transitions to sample
        return        : Tuple containing observations, actions, rewards and next_observations with done_mask indicating which actions have ended the episode
        """
        assert self.num_tranisitions > 0
        
        # generate indices 
        idxs = np.random.choice(range(self.num_transitions), batchsize, replace=False)
                
        # put together the observations, actions, rewards, ...
        obs = np.concatenate([self.__encode_observation(idx)[np.newaxis, :] for idx in idxs], 0)
        act = self.action[idxs]
        rew = self.reward[idxs]
        next_obs = np.concatenate([self.__encode_observation(idx + 1)[np.newaxis, :] for idx in idxs], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxs], dtype=np.float32)
        
        return (obs, act, rew, next_obs, done_mask)
    
    def encode_recent_observations(self):
        """
        Return the <history_len> most recent frames
        """        
        assert self.num_transitions > 0
        return self.__encode_observation((self.next_idx - 1) % self.size)
    
    def store_frame(self, frame):
        """
        Store a new frame in the buffer, overwriting old frames if the buffer is full
        
        frame (np.array): Uint8 array of (h,w,c) 
        return          : Index of the stored frame
        """
        if len(frame.shape) > 1:
            frame.transpose(2, 0, 1)
        
        self.obs[self.next_idx] = frame
        at_idx = self.next_idx
        
        self.next_idx = (self.next_idx + 1) % self.size # circular buffer
        self.num_transitions = min(self.size, self.num_tranisitions + 1)
    
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
        
        