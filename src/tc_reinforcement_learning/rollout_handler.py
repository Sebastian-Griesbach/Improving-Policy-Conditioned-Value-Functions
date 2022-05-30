from tc_reinforcement_learning.environment_handler import EnvironmentHandler
from tc_utils.util import numpy_wrap_module, extract_data

class RolloutHandler():
    def __init__(self, 
                environment_handler: EnvironmentHandler,
                to_device: str= "cpu") -> None:
        """Handles rollouts, mostly used to simplify dependencies in certain cases where a rollout has to be performed in atypical places.

        Args:
            environment_handler (EnvironmentHandler): The Object that actually handles the environment.
            to_device (str, optional): Only used in update Rollout, determines to which device rollout data is moved. Defaults to "cpu".
        """
        self.environment_handler = environment_handler
        self.to_device = to_device

    def exploration_rollout(self, policy):
        """Executes an exploration Rollout/ exploration step depending on the environment_handler

        Args:
            policy: Policy to use for exploration

        Returns:
            replay_exploration_data (Dict): Dictionary containing exploration data, (for replay buffer)
            logging_exploration_data (Dict): Dictionary containing logging data
            time_steps (int): Number of time steps executed in the environment
        """
        replay_exploration_data, logging_exploration_data, time_steps = self.environment_handler.explore(policy)
        return replay_exploration_data, logging_exploration_data, time_steps

    def evaluation_rollout(self, policy, num_evaluation_runs):
        """Executes multiple evaluation rollouts.

        Args:
            policy: policy to evaluate
            num_evaluation_runs (int): Number of evaluation episodes to run.

        Returns:
            Dict: Dictionary of evaluation data
        """
        policy = numpy_wrap_module(policy, device="cpu")
        logging_evaluation_data = self.environment_handler.evaluate(policy, num_runs=num_evaluation_runs)
        return logging_evaluation_data

    def norm_wrap_policy(self, policy_module):
        """Wraps policy in a way such that it can be passed normalized numpy arrays and will handle them properly.

        Args:
            policy_module (torch.nn.module): Policy to wrap

        Returns:
            Callable: Wrapped Policy
        """
        numpy_wrapped_policy = numpy_wrap_module(policy_module, "cpu")
        norm_wrapped_policy = lambda _input: numpy_wrapped_policy(self.environment_handler.normalize_obs(_input.reshape(1,-1), update=False).squeeze())
        return norm_wrapped_policy

    def save_checkpoint(self, save_dir, prefix=""):
        """Saves checkpoint of all stateful parts.

        Args:
            save_dir (str): Directory to save to.
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        self.environment_handler.save_checkpoint(save_dir, prefix)

    def load_checkpoint(self, save_dir, prefix=""):
        """Loads Checkpoint.

        Args:
            save_dir (str): Directory to load checkpoint from
            prefix (str, optional): Prefix string for file names. Defaults to "".
        """
        self.environment_handler.load_checkpoint(save_dir, prefix)

class ModuleRolloutHandler(RolloutHandler):

    def exploration_rollout(self, policy):
        """Wrap policy and execute exploration.

        Args:
            policy (torch.nn.module): Policy to use for exploration.

        Returns:
            see exploration_rollout() in RolloutHandler
        """
        policy = numpy_wrap_module(policy, device="cpu")
        return super().exploration_rollout(policy)

    def update_rollout(self, policy, extraction_keys):
        """ Used if rollout is part of policy representation and therefore a rollout has to be performed during update steps.

        Args:
            policy: Policy to rollout
            extraction_keys (List[str]): List of relevant Dictionary of data to return

        Returns:
            _type_: _description_
        """
        policy = policy.to(device="cpu", non_blocking=True)
        policy = numpy_wrap_module(policy, device="cpu")
        replay_exploration_data, _ = self.environment_handler.explore(policy)
        normalized_data = self.environment_handler._normalize(data_dict=replay_exploration_data, update=False)
        return extract_data(data_batch=normalized_data, keys=extraction_keys, device=self.to_device)
