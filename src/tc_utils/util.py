from typing import List
import torch
import numpy as np
from torch import nn
import copy

def numpy_to_tensor(np_array, device="cpu"):
    return torch.tensor(np_array, dtype=torch.float32, device=device, requires_grad=False)

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def numpy_wrap_module(module, device):
    return lambda _input: tensor_to_numpy(module(numpy_to_tensor(_input, device=device)))

def init_weights(module):
    if (type(module) == torch.nn.Linear):
        module.reset_parameters()

def extract_data(data_batch, keys, device):
    return {key:numpy_to_tensor(data_batch[key], device=device) for key in keys}

def find_sequential_module(module):
    sequential = None
    if(isinstance(module, torch.nn.modules.container.Sequential)):
        sequential = module
    else:
        for child in list(module.children()):
            if(isinstance(child, torch.nn.modules.container.Sequential)):
                sequential = child
                break
    if(sequential == None):
        raise Exception(f"Could not locate sequential module in module: {module}.")
    return sequential

def calculate_discounted_future_rewards(discount, rewards):
    discounted_future_rewards = [rewards[-1]]
    current_discounted_reward = rewards[-1]
    for reward in np.flip(rewards[:-1]):
        current_discounted_reward = reward + (discount * current_discounted_reward)
        discounted_future_rewards.append(current_discounted_reward)

    return np.flip(discounted_future_rewards)

def update_target_net(source_net, target_net, tau):
    for target_params, source_params in zip(target_net.parameters(), source_net.parameters()):
        target_params.data.copy_(target_params.data * (1.0 - tau) + source_params.data * tau)

class ModuleDuplicator():
    def __init__(self, reinitalizes: List[bool], **kwargs) -> None:
        self.module_dict = {}
        for i, name, module in enumerate(zip(kwargs.items)):
            module = copy.deepcopy(module)
            if(reinitalizes[i]):
                module.apply(init_weights)
            self.module_dict[name] = module
        