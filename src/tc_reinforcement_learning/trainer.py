import torch
from tqdm import tqdm
import os

from tc_reinforcement_learning.algorithm_handler import ACAlgorithmHandler
from tc_logging.log_handler import LogHandler
from tc_utils import constants

class ActorCriticTrainer():
    def __init__(self, 
    ac_algo: ACAlgorithmHandler,
    critic_update_freq: int = 1,
    critic_update_steps: int = 10,
    actor_update_freq: int = 1, 
    actor_update_steps: int = 10,
    evaluation_freq: int = 500,
    warmup_steps: int = 1000,
    use_exploration_steps: bool = False,
    log_handler: LogHandler = None,
    disable_tqdm: bool = False) -> None:
        """Creates train which manages all components to execute training routine.

        Args:
            ac_algo (ACAlgorithmHandler): Object that manages the specific algorithm.
            critic_update_freq (int, optional): Frequency by which the critic is updated related to number of update steps. Defaults to 1.
            critic_update_steps (int, optional): Number of critic update steps per update. Defaults to 10.
            actor_update_freq (int, optional): Frequency by which the actor is updated related to number of update steps. Defaults to 1.
            actor_update_steps (int, optional): Number of critic update steps per update. Defaults to 10.
            evaluation_freq (int, optional): Frequency by wich the policy is evaluated related to number of exploration steps. Defaults to 500.
            warmup_steps (int, optional): Number of warmup exploration steps. Defaults to 1000.
            use_exploration_steps (bool, optional): Whether to use total time_step (including warmup) or only exploration steps for logging. Defaults to False.
            log_handler (LogHandler, optional): Object that manages logging. Defaults to None.
            disable_tqdm (bool, optional): Disables tqdm (don't show progress bar). Defaults to False.
        """

        self.ac_algo = ac_algo
        self.disable_tqdm = disable_tqdm

        self.critic_update_freq = critic_update_freq
        self.critic_update_steps = critic_update_steps
        self.actor_update_freq = actor_update_freq
        self.actor_update_steps = actor_update_steps
        self.evaluation_freq = evaluation_freq
        self.warmup_steps = warmup_steps

        if (log_handler != None):
            self.log_handler = log_handler
            self.enable_logging = True
        else:
            self.log_handler = None
            self.enable_logging = False

        self.actor_data_keys = self.ac_algo.actor_data_info.keys()
        self.critic_data_keys = self.ac_algo.critic_data_info.keys()

        self.total_exploration_steps = 0
        self.total_train_steps = 0
        self.total_time_steps = 0

        if(use_exploration_steps):
            self.get_logging_step = lambda: self.total_exploration_steps
        else:
            self.get_logging_step = lambda: self.total_time_steps

    def train(self, num_train_steps: int):
        """Execute training routine.

        Args:
            num_train_steps (int): number of training steps. Here this means number of exploration steps. Updates are execute according to Update frequency and number of update steps.
        """

        num_warmup_steps = max(self.warmup_steps - self.total_exploration_steps, 0)
        for _ in tqdm(range(num_warmup_steps), disable=self.disable_tqdm, desc="warmup"):
            self.explore(enable_logging=False, warm_up=True)

        for _ in tqdm(range(num_train_steps), disable=self.disable_tqdm, desc="training"):

            self.explore(enable_logging=self.enable_logging, warm_up=False)

            if(self.total_train_steps % self.critic_update_freq == 0):
                critic_log_data = self.ac_algo.update_critic(num_updates=self.critic_update_steps)
                if self.enable_logging:
                    self.log_handler.log_data(data_dict = critic_log_data, step = self.get_logging_step())

            if(self.total_train_steps % self.actor_update_freq == 0):
                actor_log_data = self.ac_algo.update_actor(num_updates=self.actor_update_steps)
                if self.enable_logging:
                    self.log_handler.log_data(data_dict = actor_log_data, step = self.get_logging_step())

            if(self.enable_logging and self.total_train_steps % self.evaluation_freq == 0):
                self.evaluate()

            self.total_train_steps += 1

        if(self.enable_logging):
            if(self.total_train_steps % self.evaluation_freq != 1):
                self.evaluate()
            self.log_handler.flush_logger()

    def explore(self, enable_logging : bool =True, warm_up : bool = False):
        """Execute exploration step.

        Args:
            enable_logging (bool, optional): Whether this step is logged. Defaults to True.
            warm_up (bool, optional): Whether this step is part of the warmup. Defaults to False.
        """

        logging_data, time_steps = self.ac_algo.explore(warm_up=warm_up)
        self.total_time_steps += time_steps
        self.total_exploration_steps += 1

        if enable_logging:
            self.log_handler.log_data(data_dict = logging_data, step = self.get_logging_step())

    def evaluate(self):
        """Execute evaluation.
        """

        logging_data = self.ac_algo.evaluate()
        
        if self.enable_logging:
            self.log_handler.log_data(data_dict = logging_data, step = self.get_logging_step())

    def get_policy(self):
        """Extract policy.

        Returns:
            Callable: Wrapped policy that takes unnormalized observation as np array and returns action as np array.
        """
        return self.ac_algo.get_policy()

    def save_checkpoint(self, save_dir : str, prefix : str =""):
        """Saves the current trainer and all sub-components.

        Args:
            save_dir (str): Directory to save files to.
            prefix (str, optional): Prefix for file names. Defaults to "".
        """
        torch.save((self.total_train_steps, self.total_exploration_steps, self.total_time_steps), os.path.join(save_dir, prefix+constants.FILE_TRAINER_STEP_COUNT+".pt"))

        self.ac_algo.save_checkpoint(save_dir=save_dir, prefix=prefix)

        if(self.enable_logging):
            self.log_handler.save_checkpoint(save_dir, prefix)

    def load_checkpoint(self, save_dir : str, prefix=""):
        """Loads trainer and all subcomponents from a checkpoint.

        Args:
            save_dir (str): Directory of checkpoint.
            prefix (str, optional): Prefix of file names. Defaults to "".
        """
        self.total_train_steps, self.total_exploration_steps, self.total_time_steps = torch.load(os.path.join(save_dir, prefix+constants.FILE_TRAINER_STEP_COUNT+".pt"))

        self.ac_algo.load_checkpoint(save_dir=save_dir, prefix=prefix)

        if(self.enable_logging):
            self.log_handler.load_checkpoint(save_dir, prefix)
