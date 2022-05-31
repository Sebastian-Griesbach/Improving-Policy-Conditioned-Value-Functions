from abc import ABC, abstractproperty, abstractmethod
from enum import Enum
import torch
import numpy as np

from tc_utils.util import tensor_to_numpy
from tc_reinforcement_learning.rollout_handler import RolloutHandler
from tc_utils import constants

class MLPParamHandler(ABC):
    def __init__(self) -> None:
        """Interface for parameter handler. For Algorithms that require data on past model parameters this module handels this data such that is it is in the right format according to the specific algorithm.
        """
        super().__init__()

    @abstractmethod
    def get_policy_replay_data(self, model: torch.nn.Module):
        """Function to extract replay data from a module in a format that it can be saved to the replay buffer.

        Args:
            model (torch.nn.Module): Module to extract data from.
        """
        ...

    @abstractmethod
    def get_policy_critic_data(self, model: torch.nn.Module):
        """Function to extract data from a policy that the critic requires to evaluate it.

        Args:
            model (torch.nn.Module): Module to extract data from.
        """
        ...

    @abstractmethod
    def format_replay_buffer_data(self, **kwargs):
        """Function to format data from the replay buffer such that it can be used as input to the critic afterwards.
        """
        ...

    @abstractproperty
    def replay_data_keys(self):
        """Keys used to save data to the replay buffer.
        """
        ...

    @abstractproperty
    def replay_data_info(self):
        """Description of the data needed to initialize the replay buffer.
        """
        ...

class FlatParamHandler(MLPParamHandler):

    def __init__(self, example_policy: torch.nn.Module) -> None:
        """Parameter handler that simply takes all parameters flattens them and saves them in one vector.

        Args:
            example_policy (torch.nn.Module): Example policy network to acquire shape of data.
        """
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
        """Parameter handler that saves parameters in a dictionary shape such that the parameters are saved in a similar format of how they are used in the actual module. Useful if the Parameters are later reused similarly to how the are used within the module they are extracted from.

        Args:
            example_policy (torch.nn.Module): Example policy network to acquire structure of module and according dictionary.
        """
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
        """Parameter handler that does not actually use parameters but rather state action pairs as representation of policies.

        Args:
            num_state_action_pairs (int): Number of state action pairs used as a representation for a policy.
            episode_length (int): Maximal time steps of a episode used for representation purposes.
            rollout_handler (RolloutHandler): Rollout handler used to execute rollouts when needed.
        """
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
        """To make sure the number of state actions paris is always the same, this function sub samples the desired amount from a state action batch. This also acts as a kind of data augmentation as the representation of a single policy will consist of different state action pairs if called multiple times.

        Args:
            states (np.ndarray): Batch of states to sub sample from.
            actions (np.ndarray): Batch of actions (according to states) to sum sample from.

        Returns:
            sampled_states (np.ndarray): Sub sampled states
            sampeled_actions (np.ndarray): Sub sampled actions
        """
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
