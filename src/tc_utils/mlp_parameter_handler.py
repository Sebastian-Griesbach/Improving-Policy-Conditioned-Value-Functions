from abc import ABC, abstractproperty, abstractmethod
from enum import Enum
import torch
import numpy as np

from tc_utils.util import tensor_to_numpy
from tc_reinforcement_learning.rollout_handler import RolloutHandler
from tc_utils import constants

class MLPParamHandler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_policy_replay_data(self, model: torch.nn.Module):
        ...

    @abstractmethod
    def get_policy_critic_data(self, model: torch.nn.Module):
        ...

    @abstractmethod
    def format_replay_buffer_data(self, **kwargs):
        ...

    @abstractproperty
    def replay_data_keys(self):
        ...

    @abstractproperty
    def replay_data_info(self):
        ...

class FlatParamHandler(MLPParamHandler):

    def __init__(self, example_policy: torch.nn.Module) -> None:
        super().__init__()
        self._replay_data_keys = [constants.DATA_PARAMETERS]
        self._replay_data_info = {self._replay_data_keys[0]: {"shape": self.get_policy_critic_data(example_policy).shape}}

    def get_policy_replay_data(self, model: torch.nn.Module):
        return {self._replay_data_keys[0]: tensor_to_numpy(torch.nn.utils.parameters_to_vector(model.parameters())).reshape(1,-1)}

    def get_policy_critic_data(self, model: torch.nn.Module):
        return torch.nn.utils.parameters_to_vector(model.parameters()).reshape(1,-1)

    def format_replay_buffer_data(self, **kwargs):
        return kwargs[self._replay_data_keys[0]]

    @property
    def replay_data_keys(self):
        return self._replay_data_keys

    @property
    def replay_data_info(self):
        return self._replay_data_info

class NamedParamHandler(MLPParamHandler):
    def __init__(self, example_policy: torch.nn.Module) -> None:
        super().__init__()
        actor_parameter_dict = self.get_policy_critic_data(example_policy)
        self._replay_data_keys = actor_parameter_dict.keys()
        self._replay_data_info = {key: {"shape": actor_parameter_dict[key].shape[1:]} for key in self._replay_data_keys}

    def get_policy_replay_data(self, model: torch.nn.Module):
        batched_param_dict = self.get_policy_critic_data(model)
        return {key:tensor_to_numpy(value) for key, value in batched_param_dict.items()}

    def get_policy_critic_data(self, model: torch.nn.Module):
        param_dict = dict(model.named_parameters())
        return {key:torch.unsqueeze(tensor ,dim=0) for key, tensor in param_dict.items()}

    def format_replay_buffer_data(self, **kwargs):
        return {key:kwargs[key] for key in self._replay_data_keys}

    @property
    def replay_data_info(self):
        return self._replay_data_info

    @property
    def replay_data_keys(self):
        return self._replay_data_keys

class StateActionHandler(MLPParamHandler):

    def __init__(self,
                num_state_action_pairs: int,
                episode_length: int,
                rollout_handler: RolloutHandler) -> None:
        super().__init__()
        self.rollout_handler = rollout_handler
        self.num_state_action_pairs = num_state_action_pairs
        self.episode_length = episode_length
        self._replay_data_keys = [constants.DATA_OBSERVATIONS, constants.DATA_ACTIONS]
        self._replay_data_info = {constants.DATA_OBSERVATIONS: {"shape": (episode_length, *rollout_handler.environment_handler.exploration_environment.observation_space.shape)},
                                    constants.DATA_ACTIONS: {"shape": (episode_length, *rollout_handler.environment_handler.exploration_environment.action_space.shape)}}

    def get_policy_replay_data(self, model: torch.nn.Module):
        return {}

    def get_policy_critic_data(self, model: torch.nn.Module):
        rollout_data = self.rollout_handler.update_rollout(policy=model, extraction_keys=[constants.DATA_OBSERVATIONS, constants.DATA_ACTIONS])
        sampled_states, sampeled_actions = self.format_replay_buffer_data(**rollout_data)

        return sampled_states, sampeled_actions

    def format_replay_buffer_data(self, **kwargs):
        states = kwargs[constants.DATA_OBSERVATIONS]
        actions = kwargs[constants.DATA_ACTIONS]

        sampled_states, sampeled_actions = self._sample_state_action_paris(states, actions)

        return sampled_states, sampeled_actions

    def _sample_state_action_paris(self, states, actions):
        sample_id = np.random.choice(range(self.episode_length), size=self.num_state_action_pairs, replace=False)

        sampled_states = states[:,sample_id]
        sampeled_actions = actions[:,sample_id]

        return sampled_states, sampeled_actions

    @property
    def replay_data_info(self):
        return self._replay_data_info

    @property
    def replay_data_keys(self):
        return self._replay_data_keys

class ParameterFormat(Enum):
    FlatParameters = FlatParamHandler
    NamedParameters = NamedParamHandler
    StateAction = StateActionHandler
