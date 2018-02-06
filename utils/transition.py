class Transition():
    """
    Transition class
    Stores observation and associated action, rewards, and done mask

    """
    def __init__(self, observation, action, reward, done, next_observation=None):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
        self.next_observation = next_observation
        
    def __getitem__(self, attr):
        return self.get(attr)
            