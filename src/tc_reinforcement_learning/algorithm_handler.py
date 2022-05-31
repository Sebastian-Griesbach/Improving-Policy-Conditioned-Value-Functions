from abc import ABC, abstractmethod
import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import os
import copy
from typing import Dict


from tc_algorithms.actor_critic_base import ActorCriticBase
from tc_reinforcement_learning.exploration_module import ExplorationPolicyModule
from tc_reinforcement_learning.rollout_handler import RolloutHandler
from tc_reinforcement_learning.replay_buffer import ReplayBuffer
from tc_utils.util import extract_data, init_weights, update_target_net
from tc_utils import constants

class ACAlgorithmHandler(ABC):
    def __init__(self, 
                actor: torch.nn.Module,
                critic: torch.nn.Module,
                algorithm: ActorCriticBase,
                exploration_module: ExplorationPolicyModule,
                rollout_handler: RolloutHandler,
                replay_buffer: ReplayBuffer,
                device: str = "cpu",
                num_evaluation_runs: int = 50,
                batch_size: int = 256,
                **kwargs) -> None:
        """Interface for algorithm handler. This object takes care of communication between relevant components for training and allows for easy implementation of meta algorithms (algorithms modifications that are independent of specific algorithms.)

        Args:
            actor (torch.nn.Module): Actor module.
            critic (torch.nn.Module): Critic module.
            algorithm (ActorCriticBase): The actual Actor Critic algorithm.
            exploration_module (ExplorationPolicyModule): Exploration module.
            rollout_handler (RolloutHandler): Object that handles Rollout for all purposes, usually exploration and evaluation.
            replay_buffer (ReplayBuffer): Replay buffer that stores a provides data for training.
            device (str, optional): Device to run training on. Defaults to "cpu".
            num_evaluation_runs (int, optional): How many episodes are executed per evaluation. Defaults to 50.
            batch_size (int, optional): Batch size of data requested from the replay buffer. Defaults to 256.
        """
        super().__init__()

        self.algorithm = algorithm
        self.exploration_module = exploration_module
        self.rollout_handler = rollout_handler
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.num_evaluation_runs = num_evaluation_runs

        self.actor = actor.train().to(device=device, non_blocking=True)
        self.critic = critic.train().to(device=device, non_blocking=True)
        self.device = device

        self.critic_update_arguments, self.actor_update_arguments = algorithm.create_update_arguments(
            **{constants.ARGS_CRITIC: critic, 
            constants.ARGS_ACTOR:actor})

        self.actor_data_keys = self.algorithm.actor_data_info.keys()
        self.critic_data_keys = self.algorithm.critic_data_info.keys()

        self.actor_data_sampler = lambda: self._sample_and_extract(self.actor_data_keys)
        self.critic_data_sampler = lambda: self._sample_and_extract(self.critic_data_keys)

    def get_policy(self):
        """Provides the policy in a easily usable form. Takes numpy array as observation and returns numpy array as action. Normalization in handled accordingly to training.

        Returns:
            Callable: Policy in an easily usable form.
        """
        policy = self.get_evaluation_policy().cpu()
        return self.rollout_handler.norm_wrap_policy(policy_module=policy)
    
    def _sample_and_extract(self, keys):
        """Extracts relevant data from the replay buffer batch according to keys and transfers them to the training device.

        Args:
            keys (List[str]): List of keys that are to be extracted from the data.

        Returns:
            Dict: Dictionary that only contains the pairs with the relevant keys.
        """
        data_batch = self.replay_buffer.sample(self.batch_size)
        return extract_data(data_batch=data_batch, keys=keys, device=self.device)


    def save_checkpoint(self, save_dir, prefix=""):
        """Saves checkpoint of all stateful parts.

        Args:
            save_dir (str): Directory to save to.
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        self.exploration_module.save_checkpoint(save_dir, prefix)
        self.rollout_handler.save_checkpoint(save_dir, prefix)
        self.replay_buffer.save_checkpoint(save_dir, prefix)
        self.algorithm.save_checkpoint(critic_update_arguments=self.critic_update_arguments, 
                                        actor_update_arguments=self.actor_update_arguments, 
                                        save_dir=save_dir, prefix=prefix)


    def load_checkpoint(self, save_dir, prefix=""):
        """Loads Checkpoint.

        Args:
            save_dir (str): Directory to load checkpoint from
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        self.exploration_module.load_checkpoint(save_dir, prefix)
        self.rollout_handler.load_checkpoint(save_dir, prefix)
        self.replay_buffer.load_checkpoint(save_dir, prefix)
        self.algorithm.load_checkpoint(critic_update_arguments=self.critic_update_arguments, 
                                        actor_update_arguments=self.actor_update_arguments, 
                                        save_dir=save_dir, prefix=prefix)
    
    @property
    def replay_buffer_info(self):
        """Returns data about how to set up the replay buffer according the specific algorithm.

        Returns:
            Dict: Description about required data for replay buffer setup.
        """
        return self.algorithm.replay_buffer_info

    @property
    def critic_data_info(self):
        """Data needed for critic updates.

        Returns:
            Dict: Dictionary containing description of data needed for critic update.
        """
        return self.algorithm.critic_data_info

    @property
    def actor_data_info(self):
        """Data needed for actor updates.

        Returns:
            _type_: Dictionary containing description of data needed for actor update.
        """
        return self.algorithm.actor_data_info

    def update_critic(self, num_updates):
        """Update the critic according to specific algorithm.

        Args:
            num_updates (int): How many update steps are performed.

        Returns:
            Dict: Dictionary containing data for logging.
        """
        logging_data = self.algorithm.update_critic(data_sampler=self.critic_data_sampler, 
                                                    num_updates=num_updates,
                                                    **self.critic_update_arguments)

        self.algorithm.update_critic_update_arguments(critic_update_arguments=self.critic_update_arguments, actor_update_arguments=self.actor_update_arguments)
        return logging_data

    def update_actor(self, num_updates):
        """Update the actor according to specific algorithm.

        Args:
            num_updates (_type_): How many update steps are performed.

        Returns:
            _type_: Dictionary containing data for logging.
        """
        logging_data = self.algorithm.update_actor(data_sampler=self.actor_data_sampler, 
                                                    num_updates=num_updates,
                                                    **self.actor_update_arguments)

        self.algorithm.update_actor_update_arguments(critic_update_arguments=self.critic_update_arguments, actor_update_arguments=self.actor_update_arguments)
        return logging_data
    
    @abstractmethod
    def get_evaluation_policy(self, **kwargs):
        """Acquire policy for evaluation purposes.
        """
        ...

    @abstractmethod
    def get_exploration_policy(self, **kwargs):
        """Acquire policy for exploration purposes.
        """
        ...

    @abstractmethod
    def explore(self, warm_up):
        """Perform exploration.

        Args:
            warm_up (bool): Whether this exploration is part of the warmup phase or not.
        """
        ...

    @abstractmethod
    def evaluate(self):
        """Perform evaluation.
        """
        ...

class DefaultACAlgorithmHandler(ACAlgorithmHandler):

    def __init__(self,
                algorithm: ActorCriticBase,
                exploration_module: ExplorationPolicyModule,
                rollout_handler: RolloutHandler,
                replay_buffer: ReplayBuffer,
                num_evaluation_runs: int = 50,
                batch_size: int = 256,
                rollout_device = "cpu",
                **kwargs) -> None:
        """Algorithm handler, that handles Actor Critic algorithms as it is usually the case.

        Args:
            see ACAlgorithmHandler
        """

        super().__init__(algorithm = algorithm,
                        exploration_module = exploration_module,
                        rollout_handler = rollout_handler,
                        replay_buffer = replay_buffer,
                        num_evaluation_runs = num_evaluation_runs,
                        batch_size = batch_size,
                        **kwargs
                        )

        self.rollout_device = rollout_device

    def get_exploration_policy(self):
        return copy.deepcopy(self.actor).to(device=self.rollout_device, non_blocking=True)
    
    def get_evaluation_policy(self):
        return copy.deepcopy(self.actor).to(device=self.rollout_device, non_blocking=True)

    def explore(self, warm_up):
        if(not warm_up):
            exploration_policy = self.exploration_module.explorativ(self.get_exploration_policy())
        else:
            exploration_policy = self.exploration_module.random()
        replay_policy_data, logging_policy_data = self.algorithm.get_policy_replay_data(exploration_policy)
        replay_exploration_data, logging_exploration_data, time_steps = self.rollout_handler.exploration_rollout(policy=exploration_policy)
        replay_data = {**replay_policy_data, **replay_exploration_data}
        logging_data = {**logging_policy_data, **logging_exploration_data}
        self.replay_buffer.add(replay_data)

        return logging_data, time_steps

    def evaluate(self):
        evaluation_policy = self.get_evaluation_policy()
        _, logging_policy_data = self.algorithm.get_policy_replay_data(evaluation_policy)
        logging_evaluation_data = self.rollout_handler.evaluation_rollout(policy=evaluation_policy, num_evaluation_runs=self.num_evaluation_runs)
        logging_data = {**logging_policy_data, **logging_evaluation_data}
        return logging_data


class MultiActorACAlgorithmHandler(DefaultACAlgorithmHandler):

    def __init__(self,
                algorithm: ActorCriticBase,
                actor: torch.nn.Module,
                num_actors: int,
                critic: torch.nn.Module,
                exploration_module: ExplorationPolicyModule,
                rollout_handler: RolloutHandler,
                replay_buffer: ReplayBuffer,
                num_evaluation_runs: int = 50,
                batch_size: int = 256,
                update_only_explored_policy: bool = False,
                explore_every_policy: bool = False,
                **kwargs) -> None:
        """Algorithm handler that holds a list of actors, instead of one, which may all be used for exploration and updating.

        Args:
            actor (torch.nn.Module): Actor module. This will be duplicated and each duplicate will be newly initialized.
            num_actors (int): How often to duplicate the actor.
            update_only_explored_policy (bool, optional): Only a actor that has be used for exploration during the last exploration phase will be updated in the actor update step. Defaults to False.
            explore_every_policy (bool, optional): Uses each actor for exploration in each exploration. This increases the amount of environment interactions by the factor num_actors. Defaults to False.
            See DefaultACAlgorithmHandler
        """

        super().__init__(algorithm = algorithm,
                        actor = actor,
                        critic = critic,
                        exploration_module = exploration_module,
                        rollout_handler = rollout_handler,
                        replay_buffer = replay_buffer,
                        num_evaluation_runs = num_evaluation_runs,
                        batch_size = batch_size,
                        **kwargs
                        )

        self.last_explored = None
        if(explore_every_policy):
            self.update_actor = lambda num_updates: self._update_actor(num_updates=num_updates)
            self.explore = self._explore_all_actors
        else:
            self.explore = self._explore_random_actor
            if(update_only_explored_policy):
                self.update_actor = lambda num_updates: self._update_specific_actor(id=self.last_explored, num_updates=num_updates)
            else:
                self.update_actor = lambda num_updates: self._update_actor(num_updates=num_updates)
        
        self.num_actors = num_actors

        self.actors = [self.actor_update_arguments[constants.ARGS_ACTOR]]
        self.actor_optimizers = [self.actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER]]
        for _ in range(self.num_actors-1):
            self.actors.append(copy.deepcopy(actor).apply(init_weights))
            self.actor_optimizers.append(self.algorithm.actor_optimizer(params=self.actors[-1].parameters(), **self.algorithm.actor_optimizer_parameters))

    def get_exploration_policy(self):
        policy, _ = self._get_random_exploration_policy()
        return policy

    def _get_random_exploration_policy(self):
        """Selects a random actor for exploration purposes.

        Returns:
            torch.nn.module: Policy for exploration purposes on the right device.
        """
        id = np.random.choice(list(range(self.num_actors)))
        return self.get_specific_exploration_policy(id=id), id

    def get_specific_exploration_policy(self, id):
        """Retrieves a specific policy for exploration purposes.

        Args:
            id (int): ID of the actor to use.

        Returns:
            torch.nn.module: Policy for exploration purposes on the right device.
        """
        return copy.deepcopy(self.actors[id]).to(device=self.rollout_device, non_blocking=True)
    
    def get_evaluation_policy(self):
        """Retrieves best actor for evaluation.

        Returns:
            torch.nn.module: Best actor of all current actors.
        """
        return self.get_specific_evaluation_policy(self._get_best_policy_id())

    def get_specific_evaluation_policy(self, id):
        """Get specific actor with ID for evaluation purposes.

        Args:
            id (int): ID of actor to retrieve.

        Returns:
            torch.nn.module: Actor with ID id.
        """
        return copy.deepcopy(self.actors[id]).eval().to(device=self.rollout_device, non_blocking=True)

    def _get_best_policy_id(self):
        """Find best actor.

        Returns:
            int: ID of best actor.
        """
        evaluation_returns = []
        for i in range(self.num_actors):
            logging_data = self.evaluate_specific_actor(id=i)
            evaluation_returns.append(logging_data["evaluation_returns"])

        return argmax(evaluation_returns)

    def _update_actor(self, num_updates):
        """Update all actors and accumulate logging data.

        Args:
            num_updates (int): Number of update steps.

        Returns:
            Dict: Accumulated logging data
        """
        accumulated_logging_data = {}
        for i in range(self.num_actors):
            logging_data = self._update_specific_actor(id = i, num_updates=num_updates)
            accumulated_logging_data.update(logging_data)
        return accumulated_logging_data

    def _update_specific_actor(self, id, num_updates):
        """Update a specific actor.

        Args:
            id (int): ID of actor to update.
            num_updates (int): Number of update steps.

        Returns:
            Dict: Logging data with added actor ID.
        """
        self.actor_update_arguments[constants.ARGS_ACTOR] = self.actors[id]
        self.actor_update_arguments[constants.ARGS_ACTOR_OPTIMIZER] = self.actor_optimizers[id]
        logging_data = self.algorithm.update_actor(data_sampler=self.actor_data_sampler, num_updates=num_updates, **self.actor_update_arguments)
        logging_data = {key+f"_{id}":value for key, value in logging_data.items()}
        return logging_data

    def explore_specific_actor(self, warm_up, id):
        """Performs an exploration step with a specific actor.

        Args:
            warm_up (bool): Whether this exploration step is part of the warmup phase or not.
            id (int): ID of the actor to use for exploration.

        Returns:
            logging_data (Dict): Logging data
            time_steps (int): Executed time steps in the environment.
        """
        if(not warm_up):
            exploration_policy = self.exploration_module.explorativ(self.get_specific_exploration_policy(id=id))
        else:
            exploration_policy = self.exploration_module.random()
        replay_policy_data, logging_policy_data = self.algorithm.get_policy_replay_data(exploration_policy)
        replay_exploration_data, logging_exploration_data, time_steps = self.rollout_handler.exploration_rollout(policy=exploration_policy)
        replay_data = {**replay_policy_data, **replay_exploration_data}
        logging_data = {**logging_policy_data, **logging_exploration_data}
        self.replay_buffer.add(replay_data)

        return logging_data, time_steps

    def _explore_random_actor(self, warm_up):
        """Explore a random actor,

        Args:
            warm_up (bool): Whether this exploration step is part of the warmup phase or not.

        Returns:
            Dict: Logging data with added actor ID.
        """
        id = np.random.choice(list(range(self.num_actors)))
        self.last_explored = id
        logging_data, time_steps = self.explore_specific_actor(warm_up=warm_up, id=id)
        logging_data = {key+f"_{id}":value for key, value in logging_data.items()}
        return logging_data, time_steps

    def _explore_all_actors(self, warm_up):
        """Execute an exploration step with each actor.

        Args:
            warm_up (bool): Whether this exploration step is part of the warmup phase or not.

        Returns:
            Dict: Accumulated logging data.
        """
        combined_logging_data = {}
        total_timesteps = 0
        for id in range(self.num_actors):
            self.last_explored = id
            logging_data, time_steps = self.explore_specific_actor(warm_up=warm_up, id=id)
            combined_logging_data.update({key+f"_{id}":value for key, value in logging_data.items()})
            total_timesteps += time_steps
        return combined_logging_data, total_timesteps

    def explore(self, warm_up):
        raise NotImplementedError("This function is set in the constructor of this class.")

    def evaluate_specific_actor(self, id):
        """Evaluate actor with ID id.

        Args:
            id (int): ID of actor to evaluate.

        Returns:
            Dict: Logging data.
        """
        evaluation_policy = self.get_specific_evaluation_policy(id=id)
        _, logging_policy_data = self.algorithm.get_policy_replay_data(evaluation_policy)
        logging_evaluation_data = self.rollout_handler.evaluation_rollout(policy=evaluation_policy, num_evaluation_runs=self.num_evaluation_runs)
        logging_data = {**logging_policy_data, **logging_evaluation_data}
        return logging_data

    def evaluate(self):
        accumulated_logging_data = {}
        for i in range(self.num_actors):
            logging_data = self.evaluate_specific_actor(id=i)
            logging_data = {key+f"_{i}":value for key, value in logging_data.items()}
            accumulated_logging_data.update(logging_data)

        best_actor_return = max(list(accumulated_logging_data.values()))
        accumulated_logging_data[constants.DATA_EVALUATION_RETURNS] = best_actor_return
        return accumulated_logging_data

    def save_checkpoint(self, save_dir, prefix=""):
        super().save_checkpoint(save_dir=save_dir, prefix=prefix)

        for i in range(self.num_actors):
            torch.save(self.actors[i].state_dict(), os.path.join(save_dir, prefix+constants.FILE_ACTOR_STATE_DICT+f"_{i}"+".pt"))
            torch.save(self.actor_optimizers[i].state_dict(), os.path.join(save_dir, prefix+constants.FILE_ACTOR_OPTIMIZER_STATE_DICT+f"_{i}"+".pt"))

    def load_checkpoint(self, save_dir, prefix=""):
        super().load_checkpoint(save_dir=save_dir, prefix=prefix)

        for i in range(self.num_actors):
            self.actors[i].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_ACTOR_STATE_DICT+f"_{i}"+".pt")))
            self.actor_optimizers[i].load_state_dict(torch.load(os.path.join(save_dir, prefix+constants.FILE_ACTOR_OPTIMIZER_STATE_DICT+f"_{i}"+".pt")))

        

