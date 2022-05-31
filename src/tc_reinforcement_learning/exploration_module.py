from abc import ABC, abstractmethod
import torch
import copy
import numpy as np
from gym import Space

from tc_utils.util import numpy_wrap_module

class ExplorationModule(ABC):
    def __init__(self) -> None:
        """Interface for exploration handling
        """
        super().__init__()

    def save_checkpoint(self, save_path, prefix=""):
        pass

    def load_checkpoint(self, save_path, prefix=""):
        pass

    @abstractmethod
    def random(self):
        """Generates random explorations.
        """
        ...

    @abstractmethod
    def explorativ(self, base):
        """Generates noise in relation to some base. Usually adding noise to that base

        Args:
            base: base to add exploration too.
        """
        ...

class ExplorationPolicyModule(ExplorationModule):
    def __init__(self, example_policy: torch.nn.Module, device:str = "cpu") -> None:
        """Interface for exploration handling of entire policies. 

        Args:
            example_policy (torch.nn.Module): An example policy to acquire the structure of the network.
            device (str, optional): On which device the noise is added. Defaults to "cpu".
        """
        super().__init__()
        self.device = device
        self.example_policy = copy.deepcopy(example_policy.to(self.device))

class EpsilonNormalExplorationPolicyModule(ExplorationPolicyModule):
    def __init__(self, example_policy: torch.nn.Module, noise_std: float, epsilon: float, random_prob: float= 0., device="cpu") -> None:
        """Exploration module that does exploration in the parameter space of the policy by adding noise to the parameters or replacing them with random parameters.

        Args:
            example_policy (torch.nn.Module): An example policy to acquire the structure of the network.
            noise_std (float): Standard deviation of added noise.
            epsilon (float): Probability to not add noise.
            random_prob (float, optional): Probability to use random parameters. Defaults to 0.
            device (str, optional): On which device the noise is added. Defaults to "cpu".
        """
        super().__init__(example_policy = example_policy, device = device)
        self.noise_std = noise_std
        self.epsilon = epsilon
        self.random_prob = random_prob
        self.not_noise_prob = epsilon + random_prob

    def random(self):
        random_policy = copy.deepcopy(self.example_policy)
        for parameters in random_policy.parameters():
            if(parameters.requires_grad):
                parameters.data = torch.randn_like(parameters.data, device=self.device) * torch.tensor(self.noise_std, requires_grad=False, device=self.device)

        return random_policy

    def explorativ(self, base):
        exploration_policy = base
        random = np.random.random()
        if(self.not_noise_prob <= random):
            self._add_noise(exploration_policy)
            return exploration_policy
        if(self.epsilon >= random):
            return exploration_policy
        else:
            return self.random()
            

    def _add_noise(self, policy):
        for parameters in policy.parameters():
            if(parameters.requires_grad):
                parameters.data += torch.randn_like(parameters.data, device=self.device) * torch.tensor(self.noise_std, requires_grad=False, device=self.device)

class AdaptivNormalExplorationPolicyModule(EpsilonNormalExplorationPolicyModule):
    def __init__(self, example_policy: torch.nn.Module,  epsilon: float, min_noise_std: float = 0.01, noise_scaling: float = 1., random_prob: float= 0., device="cpu") -> None:
        """Exploration module similar to EpsilonNormalExplorationPolicyModule. But here noise is adapted per layer by the standard deviation of the existing parameters of that layer.

        Args:
            See EpsilonNormalExplorationPolicyModule
            min_noise_std (float, optional): Minimal noise standard deviation used if the standard deviation of a layer is lower. Defaults to 0.01.
            noise_scaling (float, optional): Scaling factor for the standard deviations of the layers to use a noise distribution. Defaults to 1.
        """
        super().__init__(example_policy=example_policy,
                        noise_std=min_noise_std,
                        epsilon=epsilon,
                        random_prob=random_prob,
                         device=device)
        self.min_noise_std = min_noise_std
        self.noise_scaling = noise_scaling

    def _add_noise(self, policy):
        for parameters in policy.parameters():
            if(parameters.requires_grad):
                layer_std = torch.std(input=parameters.data.flatten(), dim=0, unbiased=False)
                parameters.data += torch.randn_like(parameters.data, device=self.device) * torch.max(torch.tensor([layer_std * self.noise_scaling, self.min_noise_std], requires_grad=False, device=self.device))

class ExplorationActionModule(ExplorationModule):
    def __init__(self, noise_std: float, action_space: Space) -> None:
        """Exploration module that adds noise to the actions of a module.

        Args:
            noise_std (float): Standard deviation of normal noise distribution.
            action_space (Space): Relevant action space of the policy.
        """
        super().__init__()
        self.noise_std = noise_std
        self.action_space = action_space
        self.action_space_high = action_space.high
        self.action_space_low = action_space.low

    def random(self):
        return lambda observation: self.action_space.sample()

    def explorativ(self, base):
        return lambda observation: self._add_noise(numpy_wrap_module(base, device="cpu")(observation))

    def _add_noise(self, action):
        return np.minimum(np.maximum(np.random.randn(*action.shape)*self.noise_std + action, self.action_space_low), self.action_space_high)

    