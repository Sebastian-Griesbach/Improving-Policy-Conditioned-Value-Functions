import gym
import numpy as np

class GoalEnvObservationSpaceWrapper(gym.Wrapper):
    def __init__(self, env, L2_loss_to_goal=False):
        super(GoalEnvObservationSpaceWrapper, self).__init__(env)

        if L2_loss_to_goal:
            self._get_reward = lambda obs, reward: -np.linalg.norm(obs["achieved_goal"]-obs["desired_goal"])
        else:
            self._get_reward = lambda obs, reward: reward

        if(type(self.observation_space) == gym.spaces.dict.Dict):
                self.observation_space = gym.spaces.Box(low=np.hstack([env.observation_space["observation"].low, env.observation_space["desired_goal"].low]),
                                                        high=np.hstack([env.observation_space["observation"].high, env.observation_space["desired_goal"].high]))
        else:
            raise TypeError("This Wrapper wraps goal environments.")
    
    def reset(self):
        obs = self.env.reset()
        return self._combine_observations(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._combine_observations(obs), self._get_reward(obs, reward), done, info

    def _combine_observations(self, observation_dict):
        return np.hstack([observation_dict["observation"], observation_dict["desired_goal"]])
