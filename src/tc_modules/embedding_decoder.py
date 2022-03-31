import torch
from torch import nn
from math import prod
from gym.spaces.box import Box
from typing import List

from tc_modules.fully_connected import NonLinearFullyConnectedNet
from torch.utils import data

class PolicyDecoderNet(nn.Module):
    def __init__(self, embedding_size: int, state_space: Box, action_space: Box, hidden_dims: List[int]):
        super(PolicyDecoderNet, self).__init__()
        self.embedding_size = embedding_size
        self.state_size = prod(state_space.shape)
        self.action_size = prod(action_space.shape)
        self.input_size = self.embedding_size + self.state_size

        action_scaling = (action_space.high - action_space.low) / 2.
        action_offset =  action_scaling + action_space.low
        self.action_scaling = nn.parameter.Parameter(data = torch.tensor(action_scaling, dtype=torch.float32), requires_grad=False)
        self.action_offset = nn.parameter.Parameter(data = torch.tensor(action_offset, dtype=torch.float32), requires_grad=False)

        self.decoder_net = NonLinearFullyConnectedNet(layer_dims=[self.input_size, *hidden_dims, self.action_size],
                                                    final_activation=nn.Tanh())
        
    def forward(self, embeddings, states):
        combined_input = torch.hstack([embeddings, states])
        return self.decoder_net(combined_input) * self.action_scaling + self.action_offset