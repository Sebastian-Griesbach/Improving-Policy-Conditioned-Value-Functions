from typing import Tuple
import torch
import copy
import numpy as np
import os

from tc_algorithms.actor_critic_base import ActorCriticBase
from tc_utils.util import tensor_to_numpy, update_target_net
from tc_utils.mlp_parameter_handler import StateActionHandler, MLPParamHandler
from tc_utils import constants

#Policy Contioned Actor Critic
class PCAC(ActorCriticBase):
    def __init__(self,
                actor_param_handler: MLPParamHandler,
                log_parameters: bool = False,
                actor_update_step_penalty: float = None,
                **kwargs) -> None:

        super().__init__( **kwargs)

        self.log_parameters = log_parameters
        self.actor_param_handler = actor_param_handler
        self.actor_update_step_penalty = actor_update_step_penalty
        if(actor_update_step_penalty != None):
            self.update_step_penalty_loss = torch.nn.MSELoss()
            self.update_actor = self._update_actor_step_penalty

        self.policy_replay_keys = list(self.actor_param_handler.replay_data_info.keys())

    def get_policy_replay_data(self, policy_model):
        replay_data = self.actor_param_handler.get_policy_replay_data(policy_model)
        if(not self.log_parameters):
            logging_data = {}
        else:
            logging_data = tensor_to_numpy(self._get_flat_params_from_model(policy_model))

        return replay_data, logging_data

    def _get_flat_params_from_model(self, model: torch.nn.Module) -> torch.tensor:
        return torch.nn.utils.parameters_to_vector(model.parameters())

    def _update_actor_step_penalty(self, actor, critic, actor_optimizer, data_sampler, num_updates, **kwargs):
        losses = []
        additional_losses = []
        with torch.no_grad():
            original_actor_params = self.actor_param_handler.get_policy_critic_data(actor)
            original_embedding = critic.embed(original_actor_params).detach()

        for _ in range(num_updates):
            batch_dict = data_sampler()
            loss = self.calculate_actor_loss(actor=actor, critic=critic, **batch_dict, **kwargs)

            updated_actor_params = self.actor_param_handler.get_policy_critic_data(actor)
            updated_embedding = critic.embed(updated_actor_params)
            additional_loss = self.actor_update_step_penalty * self.update_step_penalty_loss(updated_embedding, original_embedding)
            total_loss = additional_loss + loss

            actor_optimizer.zero_grad()
            total_loss.backward()
            actor_optimizer.step()

            losses.append(loss.item())
            additional_losses.append(additional_loss.item())
        return {constants.LOG_TAG_ACTOR_LOSS: np.mean(losses),
                constants.LOG_ACTOR_STEP_PENALTY_LOSS: np.mean(additional_losses)}


class StartStatePCAC(PCAC):

    def __init__(self,
                actor_param_handler: MLPParamHandler,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_RETURNS: {"shape": torch.Size([1])}}

        self._actor_data_info = {}

        self.loss = torch.nn.MSELoss()

    def calculate_critic_loss(self, critic, **kwargs) -> None:

        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        values = kwargs[constants.DATA_RETURNS]

        predicted_values = critic(parameters)
        critic_loss = self.loss(predicted_values, values)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs) -> None:

        actor_params = self.actor_param_handler.get_policy_critic_data(actor)
        
        actor_loss = -critic(actor_params)

        return actor_loss

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class StatePCAC(PCAC):

    def __init__(self,
                environment_state_shape: Tuple[int],
                actor_param_handler: MLPParamHandler,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)


        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_RETURNS: {"shape": (1,)},
                            constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

        self._actor_data_info = {constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

        self.loss = torch.nn.MSELoss()

    def calculate_critic_loss(self, critic, **kwargs):
        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        values = kwargs[constants.DATA_RETURNS]
        states = kwargs[constants.DATA_OBSERVATIONS]

        predicted_values = critic(parameters, states)
        critic_loss = self.loss(predicted_values, values)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs):

        states = kwargs[constants.DATA_OBSERVATIONS]
        actor_params = self.actor_param_handler.get_policy_critic_data(actor)
        actor_embedding = critic.embed(actor_params)
        staked_actor_embedding = actor_embedding.repeat(states.shape[0],1)
        actor_loss = -critic.evaluate(staked_actor_embedding, states).mean()

        return actor_loss

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class PSVF(PCAC):

    def __init__(self,
                environment_state_shape: Tuple[int],
                actor_param_handler: MLPParamHandler,
                discount: float,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self.discount = discount

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_REWARDS: {"shape": (1,)},
                            constants.DATA_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_TERMINALS: {"shape": (1,)},
                            constants.DATA_LAST_OBSERVATIONS: {"shape": environment_state_shape}}

        self._actor_data_info = {constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

        self.loss = torch.nn.MSELoss()

    def calculate_critic_loss(self, critic, **kwargs):
        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        rewards = kwargs[constants.DATA_REWARDS]
        states = kwargs[constants.DATA_OBSERVATIONS]
        next_states = kwargs[constants.DATA_LAST_OBSERVATIONS]
        not_dones = 1. - kwargs[constants.DATA_TERMINALS]

        with torch.no_grad():
            targets = rewards + not_dones * self.discount * critic(parameters, next_states).detach()
        
        predicted_values = critic(parameters, states)
        
        critic_loss = self.loss(predicted_values, targets)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs):

        states = kwargs[constants.DATA_OBSERVATIONS]
        actor_params = self.actor_param_handler.get_policy_critic_data(actor)
        actor_embedding = critic.embed(actor_params)
        staked_actor_embedding = actor_embedding.repeat(states.shape[0],1)
        actor_loss = -critic.evaluate(staked_actor_embedding, states).mean()

        return actor_loss

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class PAVF(PCAC):

    def __init__(self,
                environment_state_shape: Tuple[int],
                environment_action_shape: Tuple[int],
                actor_param_handler: MLPParamHandler,
                discount: float,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self.discount = discount

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_REWARDS: {"shape": (1,)},
                            constants.DATA_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_ACTIONS: {"shape": environment_action_shape},
                            constants.DATA_LAST_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_TERMINALS: {"shape": (1,)}}

        self._actor_data_info = {constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

        self.loss = torch.nn.MSELoss()

    def calculate_critic_loss(self, critic, **kwargs):
        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        rewards = kwargs[constants.DATA_REWARDS]
        states = kwargs[constants.DATA_OBSERVATIONS]
        next_states = kwargs[constants.DATA_LAST_OBSERVATIONS]
        actions = kwargs[constants.DATA_ACTIONS]
        not_dones = 1. - kwargs[constants.DATA_TERMINALS]
        actor = kwargs[constants.ARGS_ACTOR]

        with torch.no_grad():
            next_actions = actor(next_states)
            targets = rewards + not_dones * self.discount * critic(parameters, next_states, next_actions).detach()

        predicted_values = critic(parameters, states, actions)
        
        critic_loss = self.loss(predicted_values, targets)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs):

        states = kwargs[constants.DATA_OBSERVATIONS]
        actions = actor(states)
        actor_params = self.actor_param_handler.get_policy_critic_data(actor)
        actor_embedding = critic.embed(actor_params)
        staked_actor_embedding = actor_embedding.repeat(states.shape[0],1)
        actor_loss = -critic.evaluate(staked_actor_embedding, states, actions).mean()

        return actor_loss

    def create_update_arguments(self, **kwargs):
        critic_update_arguments, actor_update_arguments = super().create_update_arguments(**kwargs)

        critic_update_arguments[constants.ARGS_ACTOR] = actor_update_arguments[constants.ARGS_ACTOR]

        return critic_update_arguments, actor_update_arguments

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class NStepStatePCAC(PCAC):

    def __init__(self,
                discount: float,
                n_steps: int,
                environment_state_shape: Tuple[int],
                actor_param_handler: MLPParamHandler,
                tau: float = 0.05,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self.effective_discount = discount ** n_steps
        self.tau = tau

        if(tau != None):
            self.update_critic_update_arguments = self._update_target_net

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_REWARDS: {"shape": (1,)},
                            constants.DATA_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_LAST_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_TERMINALS: {"shape": (1,)}}

        self._actor_data_info = {constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

        self.loss = torch.nn.MSELoss()

    def calculate_critic_loss(self, critic, **kwargs):
        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        partial_values = kwargs[constants.DATA_REWARDS]
        start_states = kwargs[constants.DATA_OBSERVATIONS]
        last_states = kwargs[constants.DATA_LAST_OBSERVATIONS]
        dones = kwargs[constants.DATA_TERMINALS]
        target_net = kwargs[constants.ARGS_CRITIC_TARGET_NET] if constants.ARGS_CRITIC_TARGET_NET in kwargs else critic

        not_dones =  ~dones.type(torch.bool).squeeze()
        relevant_parameters = {key:params[not_dones] for key, params in parameters.items()}
        with torch.no_grad():
            partial_values[not_dones] += self.effective_discount * target_net(relevant_parameters, last_states[not_dones])

        values = partial_values.detach()

        predicted_values = critic(parameters, start_states)
        critic_loss = self.loss(predicted_values, values)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs):

        states = kwargs[constants.DATA_OBSERVATIONS]
        actor_params = self.actor_param_handler.get_policy_critic_data(actor)

        actor_embedding = critic.embed(actor_params)
        staked_actor_embedding = actor_embedding.repeat(states.shape[0],1)
        actor_loss = -critic.evaluate(staked_actor_embedding, states).mean()

        return actor_loss

    def create_update_arguments(self, **kwargs):
        critic_update_arguments, actor_update_arguments = super().create_update_arguments(**kwargs)
        if(self.tau != None):
            critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET] = copy.deepcopy(kwargs[constants.ARGS_CRITIC])
        #actor_update_arguments[constants.ARGS_CRITIC] = critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET]

        return critic_update_arguments, actor_update_arguments

    def _update_target_net(self, critic_update_arguments, actor_update_arguments):
        update_target_net(source_net=critic_update_arguments[constants.ARGS_CRITIC], target_net=critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET], tau= self.tau)

    def save_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        super().save_checkpoint(critic_update_arguments, actor_update_arguments, save_dir, prefix=prefix)
        torch.save(critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET].state_dict(), os.path.join(save_dir, prefix+constants.ARGS_CRITIC_TARGET_NET+".pt"))

    def load_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        super().load_checkpoint(critic_update_arguments, actor_update_arguments, save_dir, prefix=prefix)
        critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.ARGS_CRITIC_TARGET_NET+".pt")))

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class StateActionPCAC(PCAC):

    def __init__(self,
                environment_state_shape: Tuple[int],
                environment_action_shape: Tuple[int],
                actor_param_handler: MLPParamHandler,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(
                        actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_RETURNS: {"shape": (1,)},
                            constants.DATA_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_ACTIONS: {"shape": environment_action_shape}}

        self._actor_data_info = {constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

        self.loss = torch.nn.MSELoss()

    def calculate_critic_loss(self, critic, **kwargs):
        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        values = kwargs[constants.DATA_RETURNS]
        states = kwargs[constants.DATA_OBSERVATIONS]
        actions = kwargs[constants.DATA_ACTIONS]

        predicted_values = critic(parameters, states, actions)

        critic_loss = self.loss(predicted_values, values)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs):
        actor_params = self.actor_param_handler.get_policy_critic_data(actor)
        states = kwargs[constants.DATA_OBSERVATIONS]

        actions = actor(states)
        actor_embedding = critic.embed(actor_params)
        staked_actor_embedding = actor_embedding.repeat(states.shape[0],1)
        actor_loss = -critic.evaluate(staked_actor_embedding, states, actions).mean()

        return actor_loss

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class StateActionParamlessPCAC(PCAC):

    def __init__(self,
                actor_param_handler: StateActionHandler,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(
                        actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_RETURNS: {"shape": (1,)}}

        self._actor_data_info = {}

        self.loss = torch.nn.MSELoss()

    def calculate_critic_loss(self, critic, **kwargs):
        states, actions = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        values = kwargs[constants.DATA_RETURNS]

        predicted_values = critic(states, actions)
        critic_loss = self.loss(predicted_values, values)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs):
        #do rollout of current policy
        states, _ = self.actor_param_handler.get_policy_critic_data(copy.deepcopy(actor))

        #forward pass again for differentiability
        actions = actor(states)
        actor_loss = -critic(states, actions).mean()

        return actor_loss

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class ComparingStartStatePCAC(PCAC):

    def __init__(self,
                actor_param_handler: MLPParamHandler,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_RETURNS: {"shape": torch.Size([1])}}

        #self._actor_data_info = {}
        self._actor_data_info = {**self.actor_param_handler.replay_data_info}

        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def calculate_critic_loss(self, critic, **kwargs) -> None:
        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        split = int(parameters[list(parameters.keys())[0]].shape[0]/2)
        parameters_1 = {key:value[:split] for key, value in parameters.items()}
        parameters_2 = {key:value[split:] for key, value in parameters.items()}

        values_1 = kwargs[constants.DATA_RETURNS][:split]
        values_2 = kwargs[constants.DATA_RETURNS][split:]

        equals = values_1 == values_2
        targets = (values_1 > values_2).type(torch.float32)
        targets[equals] = 0.5

        predicted_values = critic(parameters_1, parameters_2)

        critic_loss = self.loss(predicted_values, targets)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs) -> None:
        actor_params = self.actor_param_handler.get_policy_critic_data(actor)
        other_params = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        batch_size = other_params[list(other_params.keys())[0]].shape[0]

        #detached_actor_params = {key:value.detach() for key, value in actor_params.items()}

        actor_embedding = critic.embed(actor_params)
        staked_actor_embedding = actor_embedding.repeat(batch_size,1)
        other_embeddings = critic.embed(other_params)
        actor_loss = -critic.evaluate(staked_actor_embedding, other_embeddings).mean()
        
        #actor_optimizer.zero_grad()
        #actor_loss = -critic(actor_params, detached_actor_params)
        #actor_loss.backward()
        #actor_optimizer.step()

        return actor_loss

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info

class ComparingStatePCAC(PCAC):

    def __init__(self,
                environment_state_shape: Tuple[int],
                actor_param_handler: MLPParamHandler,
                log_parameters: bool = False,
                **kwargs) -> None:

        super().__init__(actor_param_handler=actor_param_handler,
                        log_parameters=log_parameters,
                        **kwargs)

        self._critic_data_info = {**self.actor_param_handler.replay_data_info, 
                            constants.DATA_RETURNS: {"shape": torch.Size([1])},
                            constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}
                            }

        self._actor_data_info = {**self.actor_param_handler.replay_data_info,
                                constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

        self.loss = torch.nn.BCEWithLogitsLoss()

    def calculate_critic_loss(self, critic, **kwargs) -> None:
        parameters = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        states = kwargs[constants.DATA_OBSERVATIONS]

        split = int(states.shape[0]/2)
        states_1 = states[:split]
        states_2 = states[split:]
        parameters_1 = {key:value[:split] for key, value in parameters.items()}
        parameters_2 = {key:value[split:] for key, value in parameters.items()}

        values_1 = kwargs[constants.DATA_RETURNS][:split]
        values_2 = kwargs[constants.DATA_RETURNS][split:]

        equals = values_1 == values_2
        targets = (values_1 > values_2).type(torch.float32)
        targets[equals] = 0.5

        predicted_values = critic(states_1 , states_2, parameters_1, parameters_2)

        critic_loss = self.loss(predicted_values, targets)

        return critic_loss

    def calculate_actor_loss(self, actor, critic, **kwargs) -> None:
        actor_params = self.actor_param_handler.get_policy_critic_data(actor)
        other_params = self.actor_param_handler.format_replay_buffer_data(**kwargs)
        states = kwargs[constants.DATA_OBSERVATIONS]
        batch_size = states.shape[0]

        split = int(batch_size/2)
        states_1 = torch.vstack([states[:split],states[:split]])
        states_2 = torch.vstack([states[split:],states[split:]])

        actor_embedding = critic.embed(actor_params)
        staked_actor_embedding = actor_embedding.repeat(batch_size,1)
        other_embeddings = critic.embed(other_params)
        actor_loss = -critic.evaluate(states_1, states_2, staked_actor_embedding, other_embeddings).mean()

        return actor_loss

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info


    
    