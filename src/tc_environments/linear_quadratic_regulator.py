import gym
from gym import spaces
import numpy as np


class LinearQuadraticRegulator(gym.Env):
    def __init__(self, start_position: float, limit: float, max_time_steps: int):
        super(LinearQuadraticRegulator, self).__init__()

        self.start_position = start_position
        self.limit = limit
        self.max_time_steps = max_time_steps

        self.action_space = spaces.Box(low=-2*self.limit, high=2*self.limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.limit, high=self.limit, shape=(1,), dtype=np.float32)

        self.time_steps = None
        self.position = None

    def reset(self):
        self.position = self.start_position
        self.time_steps = 0
        return self.position

    def step(self, action):
        next_position = np.clip(self.position + action, -self.limit, self.limit)
        reward = -self.position**2 - action**2
        self.time_steps += 1
        done = self.time_steps >= self.max_time_steps

        self.position = next_position

        return self.position, reward, done, None

    def render(self, mode='console'):
        print(self.position)

    def close(self):
        pass