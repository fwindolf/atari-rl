from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

class ScreenBase:
    def __init__(self, env, dim=None):
        """
        Create a new screen wrapper around the environment
        
        env (gym)   : OpenAI gym environment that should be wrapped
        dim (shape) : H, W, C of the output frames
        """
        self.env = env
        if dim is None:
            dim = env.observation_space.shape
            
        self.h, self.w, self.c = dim
        self.crop = None
        
        self.last_frame = None
        self.cur_frame = None
    
    def init(self):
        if self.cur_frame is None:
            self.cur_frame = self.__output(self.env.reset())
            
    def get_dim(self):
        return dim[0] * dim[1] * dim[2]
            
    def input(self, action):
        """
        Input an action to the screen
        action (int)   : Number inside env.action_space
        return (tuple) : Frame, reward and done flag as a result of the action
        """
        self.last_frame = self.cur_frame
        
        # Do the next step in the environment
        obs, reward, done, _ = env.step(action)
        
        self.cur_frame = self.__output(obs)
        
        return self.cur_frame, reward, done
    
    def __output(self, frame):
        """        
        Get the important screen area
        """        
        # Omit screen that doesn't change
        if self.crop is not None:
            frame = frame[self.crop[0]:-self.crop[1], self.crop[2]:-self.crop[3]]
        
        # Make RGB if only 1 channel
        if self.c is 1:
            frame = rgb2gray(frame)
               
        # Resize to output dimensions
        frame = resize(frame, (self.h, self.w))
        frame = np.reshape(frame, [self.h, self.w, self.c])               
        return frame   
    
    def get_current_frame(self):
        """
        Get the current frame of this screen
        """
        return self.cur_frame
    
    def get_last_frame(self):
        """
        Get the last frame of this screen
        """
        return self.last_frame
    
class SpaceInvaderScreen(ScreenBase):
    def __init__(self, env, dim = (80, 80, 3)):
        super().__init__(env, dim)        
        self.crop = (8, 12, 4, 16)
        
            
        