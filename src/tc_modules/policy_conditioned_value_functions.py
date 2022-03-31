import torch
from torch import nn
from typing import List, Tuple, Union
from abc import abstractmethod
import numpy as np

from tc_modules.mlp_embedding import MLPEmbeddingNetwork
from tc_modules.fully_connected import NonLinearFullyConnectedNet

class PolicyConditionedValueFunction(nn.Module):
    def __init__(self,
                embedding_net: MLPEmbeddingNetwork):
        super(PolicyConditionedValueFunction, self).__init__()

        self.embedding_net = embedding_net

    def embed(self, to_embed):
        return self.embedding_net(to_embed).flatten(start_dim=1)

    @abstractmethod
    def evaluate(self, **kwargs):
        ...

class StartStatePolicyConditionedValueFunction(PolicyConditionedValueFunction):
    def __init__(self, embedding_net: nn.Module,
                evaluation_net_hidden_dims: List[int]):
        super(StartStatePolicyConditionedValueFunction, self).__init__(embedding_net)

        self.evaluation_net = NonLinearFullyConnectedNet([embedding_net.embedding_size,*evaluation_net_hidden_dims,1])

    def evaluate(self, policy_embeddings):
        return self.evaluation_net(policy_embeddings)

    def forward(self, mlp_parameters):
        embeddings = self.embed(mlp_parameters)
        return self.evaluate(embeddings)

class StatePolicyConditionedValueFunction(PolicyConditionedValueFunction):
    def __init__(self, embedding_net: nn.Module,
                evaluation_net_hidden_dims: List[int],
                state_size: Union[int, Tuple[int]]):
        super(StatePolicyConditionedValueFunction, self).__init__(embedding_net)

        if type(state_size) == tuple:
            state_size = np.prod(state_size)

        self.evaluation_net = NonLinearFullyConnectedNet([embedding_net.embedding_size + state_size,*evaluation_net_hidden_dims,1])

    def evaluate(self, policy_embeddings, states):
        concatinated_inputs = torch.hstack([policy_embeddings, states])
        return self.evaluation_net(concatinated_inputs)

    def forward(self, mlp_parameters, states):
        embeddings = self.embed(mlp_parameters)
        return self.evaluate(embeddings, states)

class StateActionPolicyConditionedValueFunction(PolicyConditionedValueFunction):
    def __init__(self, embedding_net: nn.Module,
                evaluation_net_hidden_dims: List[int],
                state_size: Union[int, Tuple[int]],
                action_size: Union[int, Tuple[int]]):
        super(StateActionPolicyConditionedValueFunction, self).__init__(embedding_net)

        if type(state_size) == tuple:
            state_size = np.prod(state_size)
        if type(action_size) == tuple:
            action_size = np.prod(action_size)

        self.evaluation_net = NonLinearFullyConnectedNet([embedding_net.embedding_size + state_size + action_size,*evaluation_net_hidden_dims,1])

    def evaluate(self, policy_embeddings, states, actions):
        concatinated_inputs = torch.hstack([policy_embeddings, states, actions])
        return self.evaluation_net(concatinated_inputs)

    def forward(self, mlp_parameters, states, actions):
        embeddings = self.embed(mlp_parameters)
        return self.evaluate(embeddings, states, actions)

class ParamlessStateActionPolicyConditionedValueFunction(PolicyConditionedValueFunction):
    def __init__(self, embedding_net: MLPEmbeddingNetwork, evaluation_net_hidden_dims: List[int]):
        super().__init__(embedding_net)
        self.evaluation_net = NonLinearFullyConnectedNet([embedding_net.embedding_size, *evaluation_net_hidden_dims, 1])

    def embed(self, **to_embed):
        return self.embedding_net(**to_embed).flatten(start_dim=1)

    def evaluate(self, policy_embeddings):
        return self.evaluation_net(policy_embeddings)

    def forward(self, states, actions):
        embeddings = self.embed(states=states, actions=actions)
        return self.evaluate(embeddings)

class ComparingStartStatePolicyConditionedValueFunction(PolicyConditionedValueFunction):
    def __init__(self, embedding_net: MLPEmbeddingNetwork, evaluation_net_hidden_dims: List[int]):
        super().__init__(embedding_net)
        self.evaluation_net = NonLinearFullyConnectedNet([embedding_net.embedding_size*2, *evaluation_net_hidden_dims, 1], final_activation=None)
        #No sigmoid activation because of BCEWithLogitsLoss

    def evaluate(self, policy_embeddings_1, policy_embeddings_2):
        concatinated_embeddings = torch.hstack([policy_embeddings_1, policy_embeddings_2])
        return self.evaluation_net(concatinated_embeddings)

    def forward(self, mlp_parameters_1, mlp_parameters_2):
        embeddings_1 = self.embed(mlp_parameters_1)
        embeddings_2 = self.embed(mlp_parameters_2)

        return self.evaluate(embeddings_1, embeddings_2)

class ComparingStatePolicyConditionedValueFunction(PolicyConditionedValueFunction):
    def __init__(self, embedding_net: MLPEmbeddingNetwork, evaluation_net_hidden_dims: List[int], state_size: Union[int, Tuple[int]]):
        super().__init__(embedding_net)
        if type(state_size) == tuple:
            state_size = np.prod(state_size)
        self.evaluation_net = NonLinearFullyConnectedNet([(embedding_net.embedding_size+state_size)*2, *evaluation_net_hidden_dims, 1], final_activation=None)
        #No sigmoid activation because of BCEWithLogitsLoss

    def evaluate(self, states_1, states_2, policy_embeddings_1, policy_embeddings_2):
        concatinated_embeddings = torch.hstack([states_1, states_2, policy_embeddings_1, policy_embeddings_2])
        return self.evaluation_net(concatinated_embeddings)

    def forward(self, states_1, states_2, mlp_parameters_1, mlp_parameters_2):
        embeddings_1 = self.embed(mlp_parameters_1)
        embeddings_2 = self.embed(mlp_parameters_2)

        return self.evaluate(states_1, states_2, embeddings_1, embeddings_2)