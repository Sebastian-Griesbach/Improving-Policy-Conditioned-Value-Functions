from tc_reinforcement_learning.environment_handler import EnvironmentHandler
from tc_utils.util import numpy_wrap_module, extract_data
from torch._C import device

class RolloutHandler():
    def __init__(self, 
                environment_handler: EnvironmentHandler,
                to_device: str= "cpu") -> None:
        self.environment_handler = environment_handler
        self.to_device = to_device

    def exploration_rollout(self, policy):
        replay_exploration_data, logging_exploration_data, time_steps = self.environment_handler.explore(policy)
        return replay_exploration_data, logging_exploration_data, time_steps

    def evaluation_rollout(self, policy, num_evaluation_runs):
        policy = numpy_wrap_module(policy, device="cpu")
        logging_evaluation_data = self.environment_handler.evaluate(policy, num_runs=num_evaluation_runs)
        return logging_evaluation_data

    def norm_wrap_policy(self, policy_module):
        numpy_wrapped_policy = numpy_wrap_module(policy_module, "cpu")
        norm_wrapped_policy = lambda _input: numpy_wrapped_policy(self.environment_handler.normalize_obs(_input.reshape(1,-1), update=False).squeeze())
        return norm_wrapped_policy

    def save_checkpoint(self, save_dir, prefix=""):
        self.environment_handler.save_checkpoint(save_dir, prefix)

    def load_checkpoint(self, save_dir, prefix=""):
        self.environment_handler.load_checkpoint(save_dir, prefix)

class ModuleRolloutHandler(RolloutHandler):

    def exploration_rollout(self, policy):
        policy = numpy_wrap_module(policy, device="cpu")
        return super().exploration_rollout(policy)

    def update_rollout(self, policy, extraction_keys):
        """
        Used for parameterless state action methods
        """
        policy = policy.to(device="cpu", non_blocking=True)
        policy = numpy_wrap_module(policy, device="cpu")
        replay_exploration_data, _ = self.environment_handler.explore(policy)
        normalized_data = self.environment_handler._normalize(data_dict=replay_exploration_data, update=False)
        return extract_data(data_batch=normalized_data, keys=extraction_keys, device=self.to_device)
