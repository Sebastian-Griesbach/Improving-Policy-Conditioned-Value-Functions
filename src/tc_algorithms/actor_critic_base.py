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
        """Interface for actor critic architecture Reinforcement Learning algorithms.

        Args:
            critic_optimizer_parameters (Dict): Dictionary of critic optimizer hyperparameters.
            actor_optimizer_parameters (Dict): Dictionary of actor optimizer hyperparameters.
            critic_optimizer (torch.nn.Module, optional): Optimizer used for cirtic. Defaults to torch.optim.Adam.
            actor_optimizer (torch.nn.Module, optional): Optimizer used for actor. Defaults to torch.optim.Adam.
        """

        super().__init__()

        self.critic_optimizer_parameters = critic_optimizer_parameters
        self.actor_optimizer_parameters = actor_optimizer_parameters

        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

    def save_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        """Saves checkpoint of all stateful parts.

        Args:
            save_dir (str): Directory to save to.
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        torch.save(actor_update_arguments[constants.ARGS_ACTOR].state_dict(), os.path.join(save_dir, prefix+constants.FILE_ACTOR_STATE_DICT+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_CRITIC].state_dict(), os.path.join(save_dir, prefix+constants.FILE_CRITIC_STATE_DICT+".pt"))
        torch.save(actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER].state_dict(), os.path.join(save_dir, prefix+constants.FILE_ACTOR_OPTIMIZER_STATE_DICT+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_CRITIC_OPTIMIZER].state_dict(), os.path.join(save_dir, prefix+constants.FILE_CRITIC_OPTIMIZER_STATE_DICT+".pt"))

    def load_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        """Loads Checkpoint.

        Args:
            save_dir (str): Directory to load checkpoint from
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        actor_update_arguments[constants.ARGS_ACTOR].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_ACTOR_STATE_DICT+".pt")))
        critic_update_arguments[constants.ARGS_CRITIC].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_CRITIC_STATE_DICT+".pt")))
        actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_ACTOR_OPTIMIZER_STATE_DICT+".pt")))
        critic_update_arguments[constants.ARGS_CRITIC_OPTIMIZER].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_CRITIC_OPTIMIZER_STATE_DICT+".pt")))

    def update_critic(self, critic, critic_optimizer, data_sampler, num_updates, **kwargs):
        """Update critic according to loss calculated by subclass.

        Args:
            critic (torch.nn.module): Critic to update
            critic_optimizer (torch.nn.module): Optimizer used for critic
            data_sampler (Callable): Function that provides data needed for update.
            num_updates (int): Number of Update steps performed.

        Returns:
            Dict: Logging data.
        """
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
        """Update actor according to loss calculated by subclass.

        Args:
            actor (torch.nn.module): Actor to update.
            critic (torch.nn.module): Critic used to calculate actor loss.
            actor_optimizer (torch.nn.module): Optimizer used for critic.
            data_sampler (Callable): Function that provides data needed for update.
            num_updates (int): Number of Update steps performed.

        Returns:
            Dict: Logging data.
        """
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
        """This function might be useful for algorithms that that have hyperparameters that dynamically change over time during training. This can be implemented here.

        Args:
            critic_update_arguments (Dict): Arguments used for the critic update.
            actor_update_arguments (Dict): Arguments used for the actor update.
        """
        pass

    def update_actor_update_arguments(self, critic_update_arguments, actor_update_arguments):
        """This function might be useful for algorithms that that have hyperparameters that dynamically change over time during training. This can be implemented here.

        Args:
            critic_update_arguments (Dict): Arguments used for the critic update.
            actor_update_arguments (Dict): Arguments used for the actor update.
        """
        pass

    def create_update_arguments(self, **kwargs):
        """Creates dictionary of argument to pass to critic and actor update function. This as everything else might needs to be overwritten for specific algorithms.

        Returns:
            critic_update_arguments (Dict): Arguments used for the critic update.
            actor_update_arguments (Dict): Arguments used for the actor update.
        """
        critic_update_arguments = {}
        critic_update_arguments[constants.ARGS_CRITIC] = kwargs[constants.ARGS_CRITIC]
        critic_update_arguments[constants.ARGS_CRITIC_OPTIMIZER] = self.critic_optimizer(params=kwargs[constants.ARGS_CRITIC].parameters(), **self.critic_optimizer_parameters)

        actor_update_arguments = {}
        actor_update_arguments[constants.ARGS_ACTOR] = kwargs[constants.ARGS_ACTOR]
        actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER] = self.actor_optimizer(params=kwargs[constants.ARGS_ACTOR].parameters(), **self.actor_optimizer_parameters)
        actor_update_arguments[constants.ARGS_CRITIC] = kwargs[constants.ARGS_CRITIC]

        return critic_update_arguments, actor_update_arguments

    def get_policy_replay_data(self, policy_model):
        """Extract data from the policy that might be needed for the replay buffer. This matters for Policy-Conditioned Value Functions algorithms.

        Args:
            policy_model (torch.nn.module): Module to extract data from

        Returns:
            Dict, Dict: Replay buffer data and logging data.
        """
        return {}, {}

    @abstractmethod
    def calculate_critic_loss(self, critic, **kwargs):
        """How the critic loss is calculated this has to be implemented in subclasses as a specific algorithm.

        Args:
            critic (torch.nn.loss): Critic to calculate loss of.
        """
        ...

    @abstractmethod
    def calculate_actor_loss(self, actor, critic, **kwargs):
        """How the actor loss is calculated this has to be implemented in subclasses as a specific algorithm.

        Args:
            actor (torch.nn.loss): Actor to calculate loss of.
            critic (_type_): Critic used for loss calculation.
        """
        ...

    @property
    def replay_buffer_info(self):
        """Description of all data a specific algorithm needs.

        Returns:
            Dict: Description of all data a specific algorithms needs during training.
        """
        return {**self.critic_data_info, **self.actor_data_info}

    @abstractproperty
    def critic_data_info(self):
        """Description of data needed for a critic update step.
        """
        ...

    @abstractproperty
    def actor_data_info(self):
        """Description of data needed for a actor update step.
        """
        ...