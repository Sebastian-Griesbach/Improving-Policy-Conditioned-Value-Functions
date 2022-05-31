from abc import ABC, abstractmethod
import gym
import copy
import numpy as np

from tc_utils.statistics import OnlineNormalization
from tc_utils import constants
from tc_utils.util import calculate_discounted_future_rewards

class EnvironmentHandler(ABC):

    def __init__(self, 
                exploration_environment: gym.Env,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """The environment handler interacts with the gym actual environment and provides the required information ot the algorithm to the replay buffer.

        Args:
            exploration_environment (gym.Env): Environment used for exploration.
            evaluation_environment (gym.Env, optional): Environment used for evaluation. If None this is the same as the exploration_environment. Defaults to None.
            normalize_observations (bool, optional): Whether to normalize observation before passing them to the policy (this does not affect the data given to the replay buffer). Defaults to True.
            normalization_n_bound (int, optional): Whether the online normalization has a maximal n value. Defaults to np.inf.
        """

        self.exploration_environment = exploration_environment

        if(evaluation_environment == None):
            self.evaluation_environment = copy.deepcopy(self.exploration_environment)
        else:
            self.evaluation_environment = evaluation_environment
        
        to_normalize_info = {}
        if(normalize_observations):
            to_normalize_info[constants.DATA_OBSERVATIONS] = {"shape": self.exploration_environment.observation_space.shape}

        self.online_normalization = OnlineNormalization(to_normalize_info=to_normalize_info, n_bound=normalization_n_bound)

    def normalize_obs(self, obs, update=True):
        """Normalize observation and put into dictionary.

        Args:
            obs (np.ndarray): Observation to normalize
            update (bool, optional): Whether to update online normalization with this value or not. Defaults to True.

        Returns:
            np.ndarray: Normalized observation
        """
        data_dict = {constants.DATA_OBSERVATIONS: obs}
        return self._normalize(data_dict=data_dict, update=update)[constants.DATA_OBSERVATIONS]

    def _normalize(self, data_dict, update=True):
        """Normalize data dictionary.

        Args:
            data_dict (Dict): Data to normalize.
            update (bool, optional): Whether to update online normalization with this value or not. Defaults to True.

        Returns:
            Dict: Normalized data dictionary.
        """
        if(update):
            self.online_normalization.update_stats(data_dict)
        return self.online_normalization.normalize(data_dict)

    def _evaluation_rollout(self, policy, environment, update_normalization):
        """Execute an evaluation rollout

        Args:
            policy: Policy to evaluate.
            environment: Environment to use for evaluation.
            update_normalization (bool): Whether to update online normalization with this value or not.

        Returns:
            float: Return of the rollout.
        """
        obs = environment.reset().reshape(1,-1)
        norm_obs = self.normalize_obs(obs = obs, update = update_normalization)
        done = False
        rewards = []
        while not done:
            action = policy(norm_obs).reshape(environment.action_space.shape)
            obs, reward, done, _ = environment.step(action)
            norm_obs = self.normalize_obs(obs = obs.reshape(1,-1), update = update_normalization).reshape(1,-1)
            rewards.append(reward)

        return sum(rewards)

    def save_checkpoint(self, save_dir, prefix=""):
        """Saves checkpoint of all stateful parts.

        Args:
            save_dir (str): Directory to save to.
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        self.online_normalization.save_checkpoint(save_dir, prefix+"environment_handler_")

    def load_checkpoint(self, save_dir, prefix=""):
        """Loads Checkpoint.

        Args:
            save_dir (str): Directory to load checkpoint from
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        self.online_normalization.load_checkpoint(save_dir, prefix+"environment_handler_")

    @abstractmethod
    def explore(self, policy, update_normalization=True):
        """Execute exploration and return relevant information (depending on algorithm)

        Args:
            policy: Policy to use for exploration.
            update_normalization (bool, optional): Whether to update online normalization with this value or not. Defaults to True.
        """
        ...

    @abstractmethod
    def evaluate(self, policy, num_runs):
        """Execute evaluation of policy

        Args:
            policy: Policy to evaluate.
            num_runs (int): Number of evaluation episodes to perform.
        """
        ...


class StartStateMonteCarloEnvironmentHandler(EnvironmentHandler):

    def __init__(self,
                exploration_environment: gym.Env,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment Handler that completes a full episode and records only the total return.

        Args:
            See EnvironmentHandler
        """

        super().__init__(exploration_environment = exploration_environment,
                        evaluation_environment = evaluation_environment,
                        normalize_observations = normalize_observations,
                        normalization_n_bound = normalization_n_bound)

    def explore(self, policy, update_normalization=True):
        _return, time_steps = self._rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        _return = _return.reshape(1,-1)
        replay_data = {constants.DATA_RETURNS: _return}
        log_data = {constants.DATA_EXPLORATION_RETURNS: _return.item()} 
        return (replay_data, log_data, time_steps)

    def evaluate(self, policy, num_runs):
        returns = [self._rollout(policy, self.evaluation_environment, update_normalization=False)[0] for _ in range(num_runs)]
        return {constants.DATA_EVALUATION_RETURNS: np.mean(returns).item()}

    def _rollout(self, policy, environment, update_normalization):
        obs = environment.reset().reshape(1,-1)
        obs = self.normalize_obs(obs = obs, update = update_normalization)
        done = False
        rewards = []
        while not done:
            action = policy(obs).reshape(environment.action_space.shape)
            obs, reward, done, _ = environment.step(action)
            obs = self.normalize_obs(obs = obs.reshape(1,-1), update = update_normalization)
            rewards.append(reward)
            
        return np.expand_dims(sum(rewards), 0), len(rewards)

class StateMonteCarloEnvironmentHandler(EnvironmentHandler):

    def __init__(self,
                exploration_environment: gym.Env,
                discount: float = None,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment Handler that completes a full episode and calculates back the Monte Carlo return for each time step individually.

        Args:
            See EnvironmentHandler
            discount (float, optional): Discount factor for future rewards. If none the results will be same as if it was 1, but is calculated more efficiently. Defaults to None.
        """

        super().__init__(exploration_environment = exploration_environment,
                        evaluation_environment = evaluation_environment,
                        normalize_observations = normalize_observations,
                        normalization_n_bound = normalization_n_bound)

        if(discount == None):
            self.calculate_future_rewards = lambda rewards: np.flip(np.cumsum(np.flip(rewards)))
        else:
            self.discount = discount
            self.calculate_future_rewards = lambda rewards: calculate_discounted_future_rewards(discount = self.discount, rewards = rewards)

    def explore(self, policy, update_normalization=True):
        states, future_rewards, rollout_return, time_steps = self._exploration_rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        replay_data = {constants.DATA_RETURNS: future_rewards, constants.DATA_OBSERVATIONS: states}
        log_data = {constants.DATA_EXPLORATION_RETURNS: rollout_return.item()} 
        return (replay_data, log_data, time_steps)

    def evaluate(self, policy, num_runs):
        returns = [self._evaluation_rollout(policy, self.evaluation_environment, update_normalization=False) for _ in range(num_runs)]
        return {constants.DATA_EVALUATION_RETURNS: np.mean(returns).item()}

    def _exploration_rollout(self, policy, environment, update_normalization):
        obs = environment.reset().reshape(1,-1)
        norm_obs = self.normalize_obs(obs = obs, update = update_normalization)
        done = False
        states = []
        rewards = []
        while not done:
            states.append(obs)
            action = policy(norm_obs).reshape(environment.action_space.shape)
            obs, reward, done, _ = environment.step(action)
            obs = obs.reshape(1,-1)
            norm_obs = self.normalize_obs(obs = obs, update = update_normalization).reshape(1,-1)
            rewards.append(reward)
            
        future_rewards = self.calculate_future_rewards(rewards)
        states = np.vstack(states)
        
        return states, future_rewards, sum(rewards), len(rewards)

class StateTDEnvironmentHandler(EnvironmentHandler):

    def __init__(self,
                exploration_environment: gym.Env,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment handler that records data that is needed for TD errors without actions, namely: state, reward, next state, done

        Args:
            See EnvironmentHandler
        """

        super().__init__(exploration_environment = exploration_environment,
                        evaluation_environment = evaluation_environment,
                        normalize_observations = normalize_observations,
                        normalization_n_bound = normalization_n_bound)

    def explore(self, policy, update_normalization=True):
        states, rewards, next_states, dones, rollout_return, time_steps = self._exploration_rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        replay_data = {constants.DATA_REWARDS: rewards, constants.DATA_OBSERVATIONS: states, constants.DATA_TERMINALS: dones, constants.DATA_LAST_OBSERVATIONS: next_states}
        log_data = {constants.DATA_EXPLORATION_RETURNS: rollout_return.item()} 
        return (replay_data, log_data, time_steps)

    def evaluate(self, policy, num_runs):
        returns = [self._evaluation_rollout(policy, self.evaluation_environment, update_normalization=False) for _ in range(num_runs)]
        return {constants.DATA_EVALUATION_RETURNS: np.mean(returns).item()}

    def _exploration_rollout(self, policy, environment, update_normalization):
        obs = environment.reset().reshape(1,-1)
        norm_obs = self.normalize_obs(obs = obs, update = update_normalization)
        done = False
        states = []
        rewards = []
        dones = []
        while not done:
            states.append(obs)
            action = policy(norm_obs).reshape(environment.action_space.shape)
            obs, reward, done, _ = environment.step(action)
            dones.append(done)
            obs = obs.reshape(1,-1)
            norm_obs = self.normalize_obs(obs = obs, update = update_normalization).reshape(1,-1)
            rewards.append(reward)
            
        rewards = np.vstack(rewards).reshape(-1,1)
        states = np.vstack(states)
        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]
        dones = np.expand_dims(np.array(dones), 1)
        
        return states, rewards, next_states, dones, sum(rewards), len(rewards)

class StateActionTDEnvironmentHandler(EnvironmentHandler):

    def __init__(self,
                exploration_environment: gym.Env,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment handler that records data that is needed for TD errors with actions, namely: state, action, reward, next state, done

        Args:
            See EnvironmentHandler
        """

        super().__init__(exploration_environment = exploration_environment,
                        evaluation_environment = evaluation_environment,
                        normalize_observations = normalize_observations,
                        normalization_n_bound = normalization_n_bound)

    def explore(self, policy, update_normalization=True):
        states, actions, rewards, next_states, dones, rollout_return, time_steps = self._exploration_rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        replay_data = {constants.DATA_REWARDS: rewards, constants.DATA_OBSERVATIONS: states, constants.DATA_TERMINALS: dones, constants.DATA_ACTIONS: actions, constants.DATA_LAST_OBSERVATIONS: next_states}
        log_data = {constants.DATA_EXPLORATION_RETURNS: rollout_return.item()} 
        return (replay_data, log_data, time_steps)

    def evaluate(self, policy, num_runs):
        returns = [self._evaluation_rollout(policy, self.evaluation_environment, update_normalization=False) for _ in range(num_runs)]
        return {constants.DATA_EVALUATION_RETURNS: np.mean(returns).item()}

    def _exploration_rollout(self, policy, environment, update_normalization):
        obs = environment.reset().reshape(1,-1)
        norm_obs = self.normalize_obs(obs = obs, update = update_normalization)
        done = False
        states = []
        actions = []
        rewards = []
        dones = []
        while not done:
            states.append(obs)
            action = policy(norm_obs).reshape(environment.action_space.shape)
            actions.append(action)
            obs, reward, done, _ = environment.step(action)
            dones.append(done)
            obs = obs.reshape(1,-1)
            norm_obs = self.normalize_obs(obs = obs, update = update_normalization).reshape(1,-1)
            rewards.append(reward)
            
        rewards = np.vstack(rewards).reshape(-1,1)
        states = np.vstack(states)
        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]
        actions = np.vstack(actions)
        dones = np.expand_dims(np.array(dones), 1)
        
        return states, actions, rewards, next_states, dones, sum(rewards), len(rewards)

class StateActionMonteCarloEnvironmentHandler(StateMonteCarloEnvironmentHandler):

    def __init__(self, 
                exploration_environment: gym.Env,
                discount: float = None,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment handler similar to StateMonteCarloEnvironmentHandler. It additionally also records the actions during the episode.

        Args:
            See StateMonteCarloEnvironmentHandler
        """
        
        super().__init__(exploration_environment, 
                        discount=discount,
                        evaluation_environment=evaluation_environment, 
                        normalize_observations=normalize_observations, 
                        normalization_n_bound=normalization_n_bound)

    def explore(self, policy, update_normalization=True):
        states, actions, discounted_future_rewards, rollout_return, time_steps = self._exploration_rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        replay_data = {constants.DATA_RETURNS: discounted_future_rewards, constants.DATA_OBSERVATIONS: states, constants.DATA_ACTIONS: actions}
        log_data = {constants.DATA_EXPLORATION_RETURNS: rollout_return.item()} 
        return (replay_data, log_data, time_steps)

    def _exploration_rollout(self, policy, environment, update_normalization):
        obs = environment.reset().reshape(1,-1)
        norm_obs = self.normalize_obs(obs = obs, update = update_normalization)
        done = False
        states = []
        actions = []
        rewards = []
        while not done:
            states.append(obs)
            action = policy(norm_obs).reshape(environment.action_space.shape)
            actions.append(action)
            obs, reward, done, _ = environment.step(action)
            obs = obs.reshape(1,-1)
            norm_obs = self.normalize_obs(obs = obs, update = update_normalization).reshape(1,-1)
            rewards.append(reward)

        future_rewards = self.calculate_future_rewards(rewards)
        states = np.vstack(states)
        actions = np.vstack(actions)
        
        return states, actions, future_rewards, sum(rewards), len(rewards)

class StartStateActionMonteCarloEnvironmentHandler(StateMonteCarloEnvironmentHandler):

    def __init__(self, 
                exploration_environment: gym.Env,
                discount: float = None,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment handler similar to StateActionMonteCarloEnvironmentHandler, but it only records the total return across the episode.

        Args:
            See StateMonteCarloEnvironmentHandler
        """


        super().__init__(exploration_environment, 
                        discount=discount,
                        evaluation_environment=evaluation_environment, 
                        normalize_observations=normalize_observations, 
                        normalization_n_bound=normalization_n_bound)

    def explore(self, policy, update_normalization=True):
        states, actions, rollout_return, time_steps = self._exploration_rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        replay_data = {constants.DATA_RETURNS: rollout_return, constants.DATA_OBSERVATIONS: states, constants.DATA_ACTIONS: actions}
        log_data = {constants.DATA_EXPLORATION_RETURNS: rollout_return.item()} 
        return (replay_data, log_data, time_steps)

    def _exploration_rollout(self, policy, environment, update_normalization):
        obs = environment.reset().reshape(1,-1)
        norm_obs = self.normalize_obs(obs = obs, update = update_normalization)
        done = False
        states = []
        actions = []
        rewards = []
        while not done:
            states.append(obs)
            action = policy(norm_obs).reshape(environment.action_space.shape)
            actions.append(action)
            obs, reward, done, _ = environment.step(action)
            obs = obs.reshape(1,-1)
            norm_obs = self.normalize_obs(obs = obs, update = update_normalization).reshape(1,-1)
            rewards.append(reward)

        states = np.expand_dims(np.vstack(states), 0)
        actions = np.expand_dims(np.vstack(actions), 0)
        
        return states, actions, np.expand_dims(sum(rewards), 0), len(rewards)

class StateNStepEnvironmentHandler(EnvironmentHandler):

    def __init__(self,
                n_steps: int,
                discount: float,
                exploration_environment: gym.Env,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment Handler that executes N steps in the environment and records the first state, the last state, the partial discounted future reward and whether the episode terminated.

        Args:
            n_steps (int): How many time steps are executed per exploration call.
            discount (float): Future reward discount factor.
            see EnvironmentHandler
        """

        super().__init__(exploration_environment = exploration_environment,
                        evaluation_environment = evaluation_environment,
                        normalize_observations = normalize_observations,
                        normalization_n_bound = normalization_n_bound)

        self.n_steps = n_steps
        self.discount = discount
        self.done = False
        self.current_state = np.expand_dims(self.exploration_environment.reset(), 0)

    def explore(self, policy, update_normalization=True):
        start_state, future_reward, last_state, done, time_steps = self._exploration_rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        replay_data = {constants.DATA_REWARDS: future_reward, 
                        constants.DATA_OBSERVATIONS: start_state, 
                        constants.DATA_LAST_OBSERVATIONS: last_state, 
                        constants.DATA_TERMINALS: done}
        log_data = {constants.DATA_EXPLORATION_REWARDS: future_reward.item()} 
        return (replay_data, log_data, time_steps)

    def evaluate(self, policy, num_runs):
        returns = [self._evaluation_rollout(policy, self.evaluation_environment, update_normalization=False) for _ in range(num_runs)]
        return {constants.DATA_EVALUATION_RETURNS: np.mean(returns).item()}

    def _exploration_rollout(self, policy, environment, update_normalization):
        if self.done:
            start_state = np.expand_dims(environment.reset(), 0)
            self.done = False
            norm_obs = self.normalize_obs(obs = start_state, update = update_normalization)
        else:
            start_state = self.current_state
            norm_obs = self.normalize_obs(obs = start_state, update = False)
            
        rewards = []
        steps = 0
        while steps < self.n_steps and not self.done:
            action = policy(norm_obs).reshape(environment.action_space.shape)
            obs, reward, self.done, _ = environment.step(action)
            self.current_state = np.expand_dims(obs, 0)
            norm_obs = self.normalize_obs(obs = self.current_state, update = update_normalization)
            rewards.append(reward)
            steps += 1

        done = self.done
        last_state = self.current_state
        future_reward = calculate_discounted_future_rewards(discount=self.discount, rewards=rewards)[0].reshape(1,1)
        
        return start_state, future_reward, last_state, done, steps

class StateActionEnvironmentHandler(EnvironmentHandler):

    def __init__(self,
                n_steps: int,
                exploration_environment: gym.Env,
                evaluation_environment: gym.Env = None,
                normalize_observations: bool = True,
                normalization_n_bound: int = np.inf):
        """Environment handler that performs n time steps in the environment and meanwhile records state, action, reward, next state, done at EACH time step.

        Args:
            n_steps (int): Number of time steps to execute per exploration call.
            see EnvironmentHandler
        """

        super().__init__(exploration_environment = exploration_environment,
                        evaluation_environment = evaluation_environment,
                        normalize_observations = normalize_observations,
                        normalization_n_bound = normalization_n_bound)

        self.n_steps = n_steps
        self.done = False
        self.current_state = np.expand_dims(self.exploration_environment.reset(), 0)

    def explore(self, policy, update_normalization=True):
        states, actions, rewards, next_states, dones, steps = self._exploration_rollout(policy, self.exploration_environment, update_normalization=update_normalization)
        replay_data = {constants.DATA_REWARDS: rewards, 
                        constants.DATA_OBSERVATIONS: states, 
                        constants.DATA_LAST_OBSERVATIONS: next_states, 
                        constants.DATA_TERMINALS: dones,
                        constants.DATA_ACTIONS: actions}
        log_data = {constants.DATA_EXPLORATION_REWARDS: rewards.mean().item()} 
        return (replay_data, log_data, steps)

    def evaluate(self, policy, num_runs):
        returns = [self._evaluation_rollout(policy, self.evaluation_environment, update_normalization=False) for _ in range(num_runs)]
        return {constants.DATA_EVALUATION_RETURNS: np.mean(returns).item()}

    def _exploration_rollout(self, policy, environment, update_normalization):
        steps = 0
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        if self.done:
            self.current_state = np.expand_dims(environment.reset(), 0)
            self.done = False
            norm_obs = self.normalize_obs(obs = self.current_state, update = update_normalization)
        else:
            norm_obs = self.normalize_obs(obs = self.current_state, update = False)
            
        while steps < self.n_steps:
            states.append(self.current_state)
            action = policy(norm_obs).reshape(environment.action_space.shape)
            actions.append(action)
            obs, reward, self.done, _ = environment.step(action)

            rewards.append(reward)
            dones.append(self.done)

            self.current_state = np.expand_dims(obs, 0)
            next_states.append(self.current_state)
            norm_obs = self.normalize_obs(obs = self.current_state, update = update_normalization)
            steps += 1
            if(self.done):
                self.current_state = np.expand_dims(environment.reset(), 0)
                self.done = False
                norm_obs = self.normalize_obs(obs = self.current_state, update = update_normalization)

        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
        dones = np.expand_dims(np.array(dones), 0)
        
        return states, actions, rewards, next_states, dones, steps