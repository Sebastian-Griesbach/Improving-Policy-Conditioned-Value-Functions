import gym
import numpy as np

from tc_environments.dummy_environment import DummyEnvironment

class EnvironmentMaker():
    def __init__(self, name: str) -> None:
        if(not name == "RandomNets"):
            self.environment = gym.make(name)
        else:
            self.environment = DummyEnvironment(observation_space_size=10, action_space_size=5, action_limit=1.)

        self.state_size = np.prod(self.environment.observation_space.shape)
        self.action_size = np.prod(self.environment.action_space.shape)


        