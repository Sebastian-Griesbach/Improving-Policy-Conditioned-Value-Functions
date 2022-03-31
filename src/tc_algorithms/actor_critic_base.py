from abc import ABC, abstractmethod, abstractproperty
from typing import Dict
import numpy as np
import torch
import os

from tc_utils import constants


class ActorCriticBase(ABC):

    def __init__(self, 
                critic_optimizer_parameters: Dict,
                actor_optimizer_parameters: Dict,
                critic_optimizer: torch.nn.Module = torch.optim.Adam,
                actor_optimizer: torch.nn.Module = torch.optim.Adam,
                **kwargs) -> None:

        super().__init__()

        self.critic_optimizer_parameters = critic_optimizer_parameters
        self.actor_optimizer_parameters = actor_optimizer_parameters

        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

    def save_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        torch.save(actor_update_arguments[constants.ARGS_ACTOR].state_dict(), os.path.join(save_dir, prefix+constants.FILE_ACTOR_STATE_DICT+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_CRITIC].state_dict(), os.path.join(save_dir, prefix+constants.FILE_CRITIC_STATE_DICT+".pt"))
        torch.save(actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER].state_dict(), os.path.join(save_dir, prefix+constants.FILE_ACTOR_OPTIMIZER_STATE_DICT+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_CRITIC_OPTIMIZER].state_dict(), os.path.join(save_dir, prefix+constants.FILE_CRITIC_OPTIMIZER_STATE_DICT+".pt"))

    def load_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        actor_update_arguments[constants.ARGS_ACTOR].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_ACTOR_STATE_DICT+".pt")))
        critic_update_arguments[constants.ARGS_CRITIC].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_CRITIC_STATE_DICT+".pt")))
        actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_ACTOR_OPTIMIZER_STATE_DICT+".pt")))
        critic_update_arguments[constants.ARGS_CRITIC_OPTIMIZER].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_CRITIC_OPTIMIZER_STATE_DICT+".pt")))

    def update_critic(self, critic, critic_optimizer, data_sampler, num_updates, **kwargs):
        losses = []
        for _ in range(num_updates):
            batch_dict = data_sampler()
            loss = self.calculate_critic_loss(critic=critic, **batch_dict, **kwargs)

            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

            losses.append(loss.item())

        return {constants.LOG_TAG_CRITIC_LOSS: np.mean(losses)}

    def update_actor(self, actor, critic, actor_optimizer, data_sampler, num_updates, **kwargs):
        losses = []
        for _ in range(num_updates):
            batch_dict = data_sampler()
            loss = self.calculate_actor_loss(actor=actor, critic=critic, **batch_dict, **kwargs)

            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()

            losses.append(loss.item())
        return {constants.LOG_TAG_ACTOR_LOSS: np.mean(losses)}

    def update_critic_update_arguments(self, critic_update_arguments, actor_update_arguments):
        pass

    def update_actor_update_arguments(self, critic_update_arguments, actor_update_arguments):
        pass

    def create_update_arguments(self, **kwargs):
        critic_update_arguments = {}
        critic_update_arguments[constants.ARGS_CRITIC] = kwargs[constants.ARGS_CRITIC]
        critic_update_arguments[constants.ARGS_CRITIC_OPTIMIZER] = self.critic_optimizer(params=kwargs[constants.ARGS_CRITIC].parameters(), **self.critic_optimizer_parameters)

        actor_update_arguments = {}
        actor_update_arguments[constants.ARGS_ACTOR] = kwargs[constants.ARGS_ACTOR]
        actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER] = self.actor_optimizer(params=kwargs[constants.ARGS_ACTOR].parameters(), **self.actor_optimizer_parameters)
        actor_update_arguments[constants.ARGS_CRITIC] = kwargs[constants.ARGS_CRITIC]

        return critic_update_arguments, actor_update_arguments

    def get_policy_replay_data(self, policy_model):
        return {}, {}

    @abstractmethod
    def calculate_critic_loss(self, critic, **kwargs):
        ...

    @abstractmethod
    def calculate_actor_loss(self, actor, critic, **kwargs):
        ...

    @property
    def replay_buffer_info(self):
        return {**self.critic_data_info, **self.actor_data_info}

    @abstractproperty
    def critic_data_info(self):
        ...

    @abstractproperty
    def actor_data_info(self):
        ...