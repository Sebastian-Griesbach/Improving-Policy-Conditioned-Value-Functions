import gym
from gym import spaces
import numpy as np


class DummyEnvironment(gym.Env):
    def __init__(self, observation_space_size: int, action_space_size: int, action_limit: float):
        super(DummyEnvironment, self).__init__()

        self.action_space = spaces.Box(low=-action_limit, high=action_limit, shape=(action_space_size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32)

    def reset(self):
        pass

    def step(self, action):
        pass

    def close(self):
        pass