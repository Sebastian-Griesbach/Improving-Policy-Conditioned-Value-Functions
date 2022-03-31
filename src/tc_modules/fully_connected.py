import torch
import torch.nn as nn
from typing import List

class NonLinearFullyConnectedNet(nn.Module):
    LAYER_PREFIX = "fully_connected"
    ACTIVATION_PREFIX = "activation"
    def __init__(self, 
                layer_dims: List[int], 
                activation: torch.nn.modules.activation = nn.ReLU(), 
                final_activation: torch.nn.modules.activation = None):
        super(NonLinearFullyConnectedNet, self).__init__()
        self.activation = activation
        self.layers = torch.nn.Sequential()

        self.layers.add_module(f"{self.LAYER_PREFIX}_1", nn.Linear(layer_dims[0],layer_dims[1]))
        for layer in range(1, len(layer_dims)-1):
            self.layers.add_module(f"{self.ACTIVATION_PREFIX}_{layer}", self.activation)
            self.layers.add_module(f"{self.LAYER_PREFIX}_{layer+1}", nn.Linear(layer_dims[layer],layer_dims[layer+1]))

        if(final_activation != None):
            self.layers.add_module("final_activation", final_activation)
        
    def forward(self, x):
        return self.layers(x)

class LinearFullyConnectedNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_weights: torch.Tensor = None, init_bias: torch.Tensor = None, final_activation: nn.Module = None):
        super(LinearFullyConnectedNet, self).__init__()
        self.layer = torch.nn.Sequential()
        self.layer.add_module("linear", nn.Linear(in_dim, out_dim))
        if (init_weights != None):
            self.layer[0].weight.data = init_weights
        if (init_bias != None): 
            self.layer[0].bias.data = init_bias
        if(final_activation != None):
            self.layer.add_module("activation", final_activation)

    def forward(self, x):
        return self.layer(x)