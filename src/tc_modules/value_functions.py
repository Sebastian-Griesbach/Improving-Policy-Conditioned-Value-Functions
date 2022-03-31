from typing import Tuple, List
import math
import torch

from tc_modules.fully_connected import NonLinearFullyConnectedNet

class QValueFunction(torch.nn.Module):
    def __init__(self, 
                environment_state_shape: Tuple[int],
                environment_action_shape: Tuple[int],
                evaluation_net_hidden_dims: List[int]) -> None:
        super(QValueFunction, self).__init__()
        
        self.state_size = math.prod(environment_state_shape)
        self.action_size = math.prod(environment_action_shape)

        self.evaluation_net = NonLinearFullyConnectedNet(layer_dims=[self.state_size + self.action_size, *evaluation_net_hidden_dims, 1])

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_size)
        actions = actions.reshape(-1, self.action_size)

        state_actions = torch.hstack([states, actions])

        return self.evaluation_net(state_actions)

    