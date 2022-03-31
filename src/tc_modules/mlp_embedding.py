import torch
from torch import nn
from typing import List, Union
from abc import abstractproperty
import gym
import math

from tc_modules.fully_connected import NonLinearFullyConnectedNet
from tc_utils.mlp_parameter_handler import ParameterFormat
from tc_utils.util import find_sequential_module

class MLPEmbeddingNetwork(nn.Module):
    def __init__(self):
        super(MLPEmbeddingNetwork, self).__init__()

    @abstractproperty
    def embedding_size(self) -> int:
        ...

    @abstractproperty
    def input_type(self) -> ParameterFormat:
        ...

class EmbeddingPlusHead(MLPEmbeddingNetwork):
    def __init__(self, 
                embedding_net: MLPEmbeddingNetwork,
                hidden_dims: List[int],
                ):
        super().__init__()
        self.embedding_net = embedding_net
        self.head = NonLinearFullyConnectedNet(layer_dims=[self.embedding_net.embedding_size,*hidden_dims])
        self._embedding_size = hidden_dims[-1]

    def forward(self, to_embed):
        embed = self.embedding_net(to_embed)
        output = self.head(embed)
        return output

    @property
    def input_type(self):
        return self.embedding_net.input_type

    @property
    def embedding_size(self) -> int:
        return self._embedding_size
    
class NeuronEmbeddingNetwork(MLPEmbeddingNetwork):
    def __init__(self, 
    embedding_sizes: List[int], 
    policy_network_example: nn.Module,  
    non_linearity_hidden_dims: List[Union[List[int], None]],
    variant:bool = False,
    learn_inital_embed: bool = True, 
    initialize_normal: bool = False,
    learn_bias_embed: bool = True,
    ):
        super(NeuronEmbeddingNetwork, self).__init__()

        self.learn_inital_embed = learn_inital_embed
        self.learn_bias_embed = learn_bias_embed
        self.initialize_normal = initialize_normal

        named_parameters = dict(policy_network_example.named_parameters())
        param_keys = list(named_parameters.keys())

        self.weight_keys = [key for key in param_keys if ".weight" in key]
        self.bias_keys = [key for key in param_keys if ".bias" in key]

        num_output_neurons = named_parameters[self.bias_keys[-1]].shape[0]
        self._embedding_size = num_output_neurons * embedding_sizes[-1]

        self.num_input_neurons = named_parameters[self.weight_keys[0]].shape[1]

        policy_net = find_sequential_module(policy_network_example)
        self.activations = [module for module in list(policy_net) if isinstance(module, (nn.modules.activation.ReLU, nn.modules.activation.Tanh, nn.modules.activation.Sigmoid))]

        self.neuron_embedding_layers = nn.ModuleList()

        if(self.learn_inital_embed):
            self.inital_embedding = nn.Parameter(data = torch.Tensor(self.num_input_neurons, embedding_sizes[0]).unsqueeze(0), requires_grad=True)
            if(self.initialize_normal):
                nn.init.normal_(self.inital_embedding)
            else:
                nn.init.xavier_uniform_(self.inital_embedding, gain=1.0)
        else:
            self.inital_embedding = nn.Parameter(data = torch.eye(embedding_sizes[0], requires_grad=False).unsqueeze(0), requires_grad=False)

        num_layers = len(embedding_sizes)-1
        if(num_layers > len(self.activations)):
            self.activations.append(nn.Identity())

        for i in range(num_layers):
            if(non_linearity_hidden_dims[i] != None):
                if(variant):
                    self.neuron_embedding_layers.append(NeuronEmbeddingLayer(input_embedding_size=embedding_sizes[i],
                                                                        output_embedding_size=embedding_sizes[i+1],
                                                                        hidden_dims=non_linearity_hidden_dims[i],
                                                                        mlp_imitation_activation=self.activations[i],
                                                                        start_with_activation= True,
                                                                        end_with_activation= False,
                                                                        learn_bias_embed=self.learn_bias_embed))
                else:
                    self.neuron_embedding_layers.append(NeuronEmbeddingLayer(input_embedding_size=embedding_sizes[i],
                                                                        output_embedding_size=embedding_sizes[i+1],
                                                                        hidden_dims=non_linearity_hidden_dims[i],
                                                                        #mlp_imitation_activation=self.activations[i],
                                                                        start_with_activation= i>0,
                                                                        end_with_activation= i<num_layers-1,
                                                                        learn_bias_embed=self.learn_bias_embed))
            else:
                self.neuron_embedding_layers.append(ImitationNeuronEmbeddingLayer(input_embedding_size=embedding_sizes[i],
                                                                                activation=self.activations[i],
                                                                                learn_bias_embed=self.learn_bias_embed))
        
    def forward(self, named_parameters):
        neuron_embedding = self.inital_embedding.repeat(named_parameters[self.weight_keys[0]].shape[0],1,1)

        for i in range(len(self.weight_keys)):
            neuron_embedding = self.neuron_embedding_layers[i](neuron_embedding, named_parameters[self.weight_keys[i]], named_parameters[self.bias_keys[i]])
        
        return neuron_embedding

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def input_type(self) -> ParameterFormat:
        return ParameterFormat.NamedParameters
        

class NeuronEmbeddingLayer(nn.Module):
    def __init__(self, input_embedding_size: int,
                output_embedding_size: int, 
                hidden_dims: List[int], 
                mlp_imitation_activation: nn.modules.activation = nn.ReLU(),
                non_linear_net_activation: nn.modules.activation = nn.ReLU(),
                start_with_activation: bool = True, 
                end_with_activation: bool = True,
                learn_bias_embed: bool = True,
                **kwargs):

        super(NeuronEmbeddingLayer, self).__init__()

        self.input_embedding_size = input_embedding_size
        self.output_embedding_size = output_embedding_size
        self.combined_embedding_size = self.input_embedding_size
        self.mlp_imitation_activation = mlp_imitation_activation
        self.non_linear_net_activation = non_linear_net_activation

        if(learn_bias_embed):
            self.bias_embed = nn.Parameter(data=torch.Tensor(1,self.input_embedding_size), requires_grad=True)
            nn.init.xavier_uniform_(self.bias_embed, gain=1.0)
        else:
            self.bias_embed = nn.Parameter(torch.ones(1,self.input_embedding_size, requires_grad=False))
        
        modules = []
        if start_with_activation:
            modules.append(self.mlp_imitation_activation)

        modules.append(NonLinearFullyConnectedNet([self.combined_embedding_size, *hidden_dims, self.output_embedding_size], self.non_linear_net_activation))
        
        if end_with_activation:
            modules.append(self.non_linear_net_activation)
        
        self.non_linearity = nn.Sequential(*modules)

    def forward(self, batched_input_embedding, batched_weights, batched_bias):
        repeated_bias_embed = self.bias_embed.repeat(batched_input_embedding.shape[0],1,1)
        batched_input_embedding_with_bias = torch.cat([batched_input_embedding, repeated_bias_embed], dim=1)
        batched_weights_with_bias = torch.cat([batched_weights, batched_bias.unsqueeze(2)], dim=2)
        batched_linear_transformed_embed = batched_weights_with_bias @ batched_input_embedding_with_bias
        batchless_linear_transformed_embed = batched_linear_transformed_embed.reshape(-1, self.input_embedding_size)
        batchless_non_linear_embed = self.non_linearity(batchless_linear_transformed_embed)
        batched_non_linear_embed = batchless_non_linear_embed.reshape(-1, batched_weights.shape[1], self.output_embedding_size)
        
        return batched_non_linear_embed

class ImitationNeuronEmbeddingLayer(nn.Module):
    def __init__(self, input_embedding_size: int,
                activation: nn.modules.activation,
                learn_bias_embed: bool = True,
                **kwargs):

        super(ImitationNeuronEmbeddingLayer, self).__init__()

        self.input_embedding_size = input_embedding_size
        self.activation = activation

        if(learn_bias_embed):
            self.bias_embed = nn.Parameter(data=torch.Tensor(1,self.input_embedding_size), requires_grad=True)
            nn.init.xavier_uniform_(self.bias_embed, gain=1.0)
        else:
            self.bias_embed = nn.Parameter(torch.ones(1,self.input_embedding_size, requires_grad=False))

    def forward(self, batched_input_embedding, batched_weights, batched_bias):
        repeated_bias_embed = self.bias_embed.repeat(batched_input_embedding.shape[0],1,1)
        batched_input_embedding_with_bias = torch.cat([batched_input_embedding, repeated_bias_embed], dim=1)
        batched_weights_with_bias = torch.cat([batched_weights, batched_bias.unsqueeze(2)], dim=2)
        batched_linear_transformed_embed = batched_weights_with_bias @ batched_input_embedding_with_bias
        batched_non_linear_embed = self.activation(batched_linear_transformed_embed)
        
        return batched_non_linear_embed

class FlatMPLEmbedding(MLPEmbeddingNetwork):
    def __init__(self,
                policy_network_example: nn.Module):
        super(FlatMPLEmbedding, self).__init__()

        self.indentity = nn.Identity()
        policy_num_parameters = sum(p.numel() for p in policy_network_example.parameters() if p.requires_grad)
        self._embedding_size = policy_num_parameters

    def forward(self, flat_params):
        return self.indentity(flat_params)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def input_type(self) -> ParameterFormat:
        return ParameterFormat.FlatParameters

class LinearTransformMPLEmbedding(MLPEmbeddingNetwork):
    def __init__(self,
                policy_network_example: nn.Module,
                embedding_size):
        super(LinearTransformMPLEmbedding, self).__init__()

        self._embedding_size = embedding_size

        policy_num_parameters = sum(p.numel() for p in policy_network_example.parameters() if p.requires_grad)
        self.linear_transform = nn.Linear(in_features=policy_num_parameters, out_features=self.embedding_size)

    def forward(self, flat_params):
        return self.linear_transform(flat_params)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def input_type(self) -> ParameterFormat:
        return ParameterFormat.FlatParameters


class FingerprintEmbedding(MLPEmbeddingNetwork):
    def __init__(self, 
    policy_network_example: nn.Module,
    num_in_states: int,
    observation_space: gym.Space,
    action_space: gym.Space,
    normalized_input: bool = True
    ):
        super(FingerprintEmbedding, self).__init__()

        named_parameters = dict(policy_network_example.named_parameters())
        param_keys = list(named_parameters.keys())
        self.normalized_input = normalized_input

        self.weight_keys = [key for key in param_keys if ".weight" in key]
        self.bias_keys = [key for key in param_keys if ".bias" in key]
        
        policy_net = find_sequential_module(policy_network_example)

        self.activations = [module for module in list(policy_net) if isinstance(module, (nn.modules.activation.ReLU, nn.modules.activation.Tanh, nn.modules.activation.Sigmoid))]
        self.num_activations = len(self.activations)

        action_scaling = (action_space.high - action_space.low) / 2.
        action_offset =  action_scaling + action_space.low
        self.scaling = nn.Parameter(data=torch.tensor(action_scaling, dtype=torch.float32), requires_grad=False)
        self.off_set = nn.Parameter(data=torch.tensor(action_offset, dtype=torch.float32), requires_grad=False)
        num_output_neurons = named_parameters[self.bias_keys[-1]].shape[0]
        self._embedding_size = num_output_neurons * num_in_states

        self.num_input_neurons = named_parameters[self.weight_keys[0]].shape[1]

        self.neuron_embedding_layers = nn.ModuleList()

        if(normalized_input):
            in_states = torch.randn([num_in_states, *observation_space.shape])
        else:
            in_states = torch.tensor([observation_space.sample() for _ in range(num_in_states)], dtype=torch.float32).reshape(-1, *observation_space.shape)
        self.in_states = nn.Parameter(data=in_states, requires_grad=True)
        
    def forward(self, named_parameters):
        batch_size = named_parameters[self.weight_keys[0]].shape[0]
        #TODO repeat eventuell ersetzten mit unsqueeze und expand
        forward_values = self.in_states.repeat(batch_size,1,1)

        for i in range(len(self.weight_keys)):
            linear_transformed = forward_values @ torch.transpose(named_parameters[self.weight_keys[i]], dim0=1, dim1=2)
            forward_values = linear_transformed + named_parameters[self.bias_keys[i]].unsqueeze(1)
            if(self.num_activations > i):
                forward_values = self.activations[i](forward_values)

        return (forward_values * self.scaling + self.off_set).reshape(batch_size, -1)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def input_type(self) -> ParameterFormat:
        return ParameterFormat.NamedParameters

class StateActionEmbedding(MLPEmbeddingNetwork):

    def __init__(self,
        num_state_action_pairs: int,
        observation_space: gym.Space,
        action_space: gym.Space
        ):
        super(StateActionEmbedding, self).__init__()
        self.observation_shape = observation_space.shape
        self.action_shape = action_space.shape

        self.num_state_action_pairs = num_state_action_pairs
        self._embedding_size = num_state_action_pairs* (math.prod(self.observation_shape) + math.prod(self.action_shape))
        
    def forward(self, states, actions):
        concatinated_batch = torch.cat([states, actions], dim=2)
        return concatinated_batch

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def input_type(self) -> ParameterFormat:
        return ParameterFormat.StateAction