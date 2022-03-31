import gym
import numpy as np
import stable_baselines3
import torch
from torch import nn
from tqdm import tqdm
import copy
import os
from typing import Dict
import stable_baselines3

class DatasetBuilder():
    def __init__(self, data_folder:str ,env: gym.Env, stable_baseline_algo: stable_baselines3.common.base_class.BaseAlgorithm, model_parameters: Dict) -> None:
        self.data_folder = data_folder
        self.env = env
        self.algo = stable_baseline_algo
        self.model_parameters = model_parameters

        self.file_base_name = "policy_dataset"
        self.file_prefix = os.path.join(self.data_folder,self.file_base_name)

    def build_dataset(self,
                    total_train_runs,
                  train_partition_size,
                  num_train_partitions,
                  rollout_timesteps,
                  start_idx=0,
                  disable_tqdm=False):
    
        idx = start_idx
        for train_run in tqdm(range(total_train_runs),disable=disable_tqdm):
            model = self.algo(**self.model_parameters)
            
            for partition in range(num_train_partitions):
                model.learn(total_timesteps=train_partition_size)
            
                states, actions, dones = self.record_rollout(model, rollout_timesteps)
                policy = copy.deepcopy(model.policy.actor).cpu()
                
                sample = {
                    "state_dict": policy.state_dict(),
                    "states": states,
                    "actions": actions,
                    "dones": dones
                }

                torch.save(sample, f"{self.file_prefix}_{idx}.pt")
                idx += 1

    def record_rollout(self, model, rollout_timesteps):
        states, actions , dones= [], [], []
        obs = self.env.reset()
        done = False
        for i in range(rollout_timesteps):
            action, _states = model.predict(obs)
            states.append(obs)
            actions.append(action)
            obs, rewards, done, info = self.env.step(action)
            dones.append(done)
            if done:
                obs = self.env.reset()
                done = False
        
        return states, actions, dones

    def train_policy(self, parameters, total_timesteps):
        model = self.algo(**parameters)
        model.learn(total_timesteps=total_timesteps)
        return model