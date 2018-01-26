import gym
from gym.spaces.box import Box
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

class OutputWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
     
    def _observation(self, frame):
        return frame
    
    def output(self, frame):
        return self._observation(frame)
        
class GreyscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space.shape
        self.observation_space = Box(0, 255, [shape[0], shape[1], 1])
        
    def _observation(self, frame):
        return rgb2gray(frame)
    
    def output(self, frame):
        frame = self.env.output(frame) # recursion level towards env
        return self._observation(frame)
            

class RescaleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, dim=(80, 80)):
        """
        Create a new Wrapper that rescales the observations
        env (gym) : The environment to be wrapped
        dim (ints): W, H, C of the observation
        """        
        super().__init__(env)
        self.w, self.h = dim
        if len(self.env.observation_space.shape) > 1:
            self.c = self.env.observation_space.shape[2]
        else:
            self.c = 1
        
        self.observation_space = Box(0.0, 1.0, [self.c, self.w, self.h]) # C, W, H
        
    def _observation(self, frame):
        frame = resize(frame, (self.w, self.h), mode='constant')
        return frame
    
    def output(self, frame):
        frame = self.env.output(frame) # recursion level towards env
        return self._observation(frame)
        
class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, crop=(0, 0, 0, 0)):
        """
        Create a new Wrapper that crops the image to fixed proportions
        env (gym)  : The environment to be wrapped
        crop (ints): Cropping values (ht, hb, wl, wr)
        """        
        super().__init__(env)        
        self.crop = crop            
        
    def _observation(self, frame):
        if self.crop[0] > 0:
            frame = frame[self.crop[0]:, :]
        if self.crop[1] > 0:
            frame = frame[:-self.crop[1], :]
        if self.crop[2] > 0:
            frame = frame[:, self.crop[2]:]
        if self.crop[3] > 0:
            frame = frame[:, :-self.crop[3]]
        return frame
    
    def output(self, frame):
        frame = self.env.output(frame) # recursion level towards env
        return self._observation(frame)
    
class Uint8Wrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
    
    def _observation(self, frame):
        if np.issubdtype(frame.dtype, np.integer):
            pass
        elif np.issubdtype(frame.dtype, np.floating):
            frame *= 255  # convert first
        
        return frame.astype('uint8')
    
    def output(self, frame):
        frame = self.env.output(frame) # recursion level towards env
        return self._observation(frame)
    
class CartPoleScreenWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.screen_width=600
        
    def __get_cart_location(self):
        env = self.env.unwrapped
        world_width = env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(env.state[0] * scale + self.screen_width / 2.0) # MIDDLE OF CART
    
    def __get_screen(self):
        # needs a display!
        screen = self.env.render(mode='rgb_array')
        screen = screen[:, 160:320]

        view_width = 320
        cart_location = self.__get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)

        return screen
    
    def _observation(self, frame):
        if frame is None or len(frame) == 4:
            return self.__get_screen()
        else:
            return frame
    
    def output(self, frame):
        frame = self.env.output(frame) # recursion level towards env
        return self._observation(frame)   
    
    def _reset(self):        
        self.env.reset()
        return self._observation(None)
        

class ScreenBase:
    def __init__(self, env_name):
        """
        Create a new screen wrapper around the environment        
        env (gym)   : OpenAI gym environment that should be wrapped
        """
        self.env = gym.make(env_name).env # get rid of the TimeLimit  
        self.env = OutputWrapper(self.env) # make 'output' has a final recursion (kinda hacky..)
        
    def reset(self):
        return self.env.reset()
            
    def get_dim(self):
        """
        Get the dimensions of the screen
        """        
        return np.prod(self.env.observation_space.shape)
    
    def get_shape(self):
        """
        Get the shape of the screen
        """
        return self.env.observation_space.shape
    
    def get_actions(self):
        """
        Get the possible actions of the gym env
        """
        return self.env.action_space.n
    
    def get_action_code(self, action=None):
        action_codes = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 
                        'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
                        'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
                        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
        
        if action is not None:
            return action_codes[int(action)]
        else:
            return action_codes
        
    def get_action_meaning(self, action=None):
        """
        Get the meaning of an action as string
        action (int) : The code for the action
        """
        if action is not None:
            assert(action < self.get_actions())
            return self.env.unwrapped.get_action_meanings()[action]
        else:
            return self.env.unwrapped.get_action_meanings()
    def sample_action(self):
        """
        Get a random action from the gyms action space
        """
        return self.env.action_space.sample()
        
    def input(self, action):
        """
        Input an action to the screen
        action (int)   : Number inside env.action_space
        return (tuple) : Frame, reward and done flag as a result of the action
        """
        # Do the next step in the environment
        obs, reward, done, _ = self.env.step(action) 
        return obs, reward, done
    
    def output_float(self, frame):
        """
        Transform a frame the same way a gym observation would be transformed and output as float
        """
        return self.output(frame).astype('float') / 255.0
    
    def output(self, frame):
        """
        Transform a frame the same way a gym observation would be transformed
        """
        return self.env.output(frame)
    
class SpaceInvaderScreen(ScreenBase):
    def __init__(self, dim=(80,80), crop=(20, 10, 4, 16), greyscale=True):
        super().__init__('SpaceInvaders-v0')  
        
        if greyscale:
            self.env = GreyscaleWrapper(self.env)
        if crop:
            self.env = CropWrapper(self.env, crop) # Crop first so we don't potentially lose resolution        
        if dim:
            self.env = RescaleWrapper(self.env, dim)
            
        self.env = Uint8Wrapper(self.env)
        
        
class CartPoleScreen(ScreenBase):
    def __init__(self, dim=(80,80), crop=None, greyscale=True):
        super().__init__('CartPole-v0')
        
        self.env = CartPoleScreenWrapper(self.env)
        
        if dim:
            self.env = RescaleWrapper(self.env, dim)
        if greyscale:
            self.env = GreyscaleWrapper(self.env)
        
        self.env = Uint8Wrapper(self.env)
        
        
            
        