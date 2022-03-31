from typing import Tuple
import torch
import numpy as np
import copy
import os

from tc_algorithms.actor_critic_base import ActorCriticBase
from tc_utils import constants
from tc_utils.util import update_target_net, init_weights

class TD3(ActorCriticBase):
    def __init__(self, 
                environment_state_shape: Tuple[int],
                environment_action_shape: Tuple[int],
                discount: float,
                noise_std: float,
                noise_clip: float,
                tau: float,
                **kwargs) -> None:
        super().__init__(**kwargs)

        self.discount = discount
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.tau = tau

        self.loss = torch.nn.MSELoss()

        self._critic_data_info = {
                            constants.DATA_ACTIONS: {"shape": environment_action_shape},
                            constants.DATA_REWARDS: {"shape": (1,)},
                            constants.DATA_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_LAST_OBSERVATIONS: {"shape": environment_state_shape},
                            constants.DATA_TERMINALS: {"shape": (1,)}}

        self._actor_data_info = {constants.DATA_OBSERVATIONS: {"shape": environment_state_shape}}

    def update_critic(self, data_sampler, num_updates, **kwargs):
        critic = kwargs[constants.ARGS_CRITIC]
        critic_2 = kwargs[constants.ARGS_CRITIC_2]
        critic_optimizer = kwargs[constants.ARGS_CRITIC_OPTIMIZER]
        critic_optimizer_2 = kwargs[constants.ARGS_CRITIC_2_OPTIMIZER]
        target_actor = kwargs[constants.ARGS_ACTOR_TARGET_NET]
        target_critic = kwargs[constants.ARGS_CRITIC_TARGET_NET]
        target_critic_2 = kwargs[constants.ARGS_CRITIC_TARGET_NET_2]

        losses_1 = []
        losses_2 = []
        for _ in range(num_updates):
            batch_dict = data_sampler()
            targets = self._calculate_targets(target_actor=target_actor, target_critic_1=target_critic, target_critic_2=target_critic_2, **batch_dict)
            losses_1.append(self._update_critic(critic=critic, critic_optimizer=critic_optimizer, targets=targets, **batch_dict))
            losses_2.append(self._update_critic(critic=critic_2, critic_optimizer=critic_optimizer_2, targets=targets, **batch_dict))


        return {constants.LOG_TAG_CRITIC_LOSS+"_1": np.mean(losses_1), constants.LOG_TAG_CRITIC_LOSS+"_2": np.mean(losses_2)}

    def _update_critic(self, critic, critic_optimizer, targets, **batch_dict):
            loss = self.calculate_critic_loss(critic=critic, targets=targets, **batch_dict)

            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

            return loss.item()

    def _calculate_targets(self, target_actor, target_critic_1, target_critic_2, **kwargs):
        rewards = kwargs[constants.DATA_REWARDS]
        next_states = kwargs[constants.DATA_LAST_OBSERVATIONS]
        dones = kwargs[constants.DATA_TERMINALS]
        not_dones =  ~dones.type(torch.bool).squeeze()

        with torch.no_grad():
            next_actions = target_actor(next_states[not_dones])
            noisy_next_actions = torch.clamp(torch.rand_like(next_actions) * self.noise_std, min=-self.noise_clip, max=self.noise_clip) + next_actions
            rewards[not_dones] += self.discount * torch.min(target_critic_1(next_states[not_dones], noisy_next_actions), target_critic_2(next_states[not_dones], noisy_next_actions))

            targets = rewards.detach()

        return targets

    def calculate_critic_loss(self, critic, targets, **kwargs):
        states = kwargs[constants.DATA_OBSERVATIONS]
        actions = kwargs[constants.DATA_ACTIONS]

        predictions = critic(states, actions)
        loss = self.loss(predictions, targets)

        return loss

    def calculate_actor_loss(self, actor, critic, **kwargs):
        states = kwargs[constants.DATA_OBSERVATIONS]

        actions = actor(states)
        loss = -critic(states, actions).mean()

        return loss

    def update_critic_update_arguments(self, critic_update_arguments, actor_update_arguments):
        update_target_net(source_net=critic_update_arguments[constants.ARGS_CRITIC], target_net=critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET], tau=self.tau)
        update_target_net(source_net=critic_update_arguments[constants.ARGS_CRITIC_2], target_net=critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET_2], tau=self.tau)
        update_target_net(source_net=actor_update_arguments[constants.ARGS_ACTOR], target_net=critic_update_arguments[constants.ARGS_ACTOR_TARGET_NET], tau=self.tau)

    def create_update_arguments(self, **kwargs):
        critic_update_arguments, actor_update_arguments = super().create_update_arguments(**kwargs)
        critic_update_arguments[constants.ARGS_CRITIC_2] = copy.deepcopy(critic_update_arguments[constants.ARGS_CRITIC])
        critic_update_arguments[constants.ARGS_CRITIC_2].apply(init_weights)
        critic_update_arguments[constants.ARGS_CRITIC_2_OPTIMIZER] = self.critic_optimizer(params=critic_update_arguments[constants.ARGS_CRITIC_2].parameters(), **self.actor_optimizer_parameters)
        critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET] = copy.deepcopy(critic_update_arguments[constants.ARGS_CRITIC])
        critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET_2] = copy.deepcopy(critic_update_arguments[constants.ARGS_CRITIC_2])
        critic_update_arguments[constants.ARGS_ACTOR_TARGET_NET] = copy.deepcopy(actor_update_arguments[constants.ARGS_ACTOR])

        return critic_update_arguments, actor_update_arguments

    def save_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        super().save_checkpoint(critic_update_arguments, actor_update_arguments, save_dir, prefix=prefix)

        torch.save(critic_update_arguments[constants.ARGS_CRITIC_2].state_dict(), os.path.join(save_dir, prefix+constants.ARGS_CRITIC_2+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_CRITIC_2_OPTIMIZER].state_dict(), os.path.join(save_dir, prefix+constants.ARGS_CRITIC_2_OPTIMIZER+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET].state_dict(), os.path.join(save_dir, prefix+constants.ARGS_CRITIC_TARGET_NET+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET_2].state_dict(), os.path.join(save_dir, prefix+constants.ARGS_CRITIC_TARGET_NET_2+".pt"))
        torch.save(critic_update_arguments[constants.ARGS_ACTOR_TARGET_NET].state_dict(), os.path.join(save_dir, prefix+constants.ARGS_ACTOR_TARGET_NET+".pt"))

    def load_checkpoint(self, critic_update_arguments, actor_update_arguments, save_dir, prefix=""):
        super().load_checkpoint(critic_update_arguments, actor_update_arguments, save_dir, prefix=prefix)

        critic_update_arguments[constants.ARGS_CRITIC_2].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.ARGS_CRITIC_2+".pt")))
        critic_update_arguments[constants.ARGS_CRITIC_2_OPTIMIZER].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.ARGS_CRITIC_2_OPTIMIZER+".pt")))
        critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.ARGS_CRITIC_TARGET_NET+".pt")))
        critic_update_arguments[constants.ARGS_CRITIC_TARGET_NET_2].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.ARGS_CRITIC_TARGET_NET_2+".pt")))
        critic_update_arguments[constants.ARGS_ACTOR_TARGET_NET].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.ARGS_ACTOR_TARGET_NET+".pt")))

    @property
    def critic_data_info(self):
        return self._critic_data_info

    @property
    def actor_data_info(self):
        return self._actor_data_info