import cpprb
from typing import Dict, List
import numpy as np
import math
from itertools import cycle
from operator import itemgetter
from abc import ABC, abstractmethod
import os

from tc_utils.statistics import OnlineNormalization
import torch
import tc_utils.constants as constants

class ReplayBuffer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add(self, transition_dict: Dict[str, np.ndarray]):
        ...

    @abstractmethod
    def sample(self, batch_size: int):
        ...

    @abstractmethod
    def save_checkpoint(self, save_dir, prefix=""):
        ...

    @abstractmethod
    def load_checkpoint(self, save_dir, prefix=""):
        ...

class FastReplayBuffer(ReplayBuffer):
    def __init__(self, max_size: int, replay_buffer_info: Dict) -> None:
        super().__init__()
        self.replay_buffer = cpprb.ReplayBuffer(max_size, replay_buffer_info)

    def add(self, transition_dict):
        self.replay_buffer.add(**transition_dict)

    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def save_checkpoint(self, save_dir, prefix=""):
        self.replay_buffer.save_transitions(os.path.join(save_dir,prefix+constants.FILE_REPLAY_BUFFER))

    def load_checkpoint(self, save_dir, prefix=""):
        self.replay_buffer.clear()
        self.replay_buffer.load_transitions(os.path.join(save_dir,prefix+constants.FILE_REPLAY_BUFFER))

class NormalizingReplayBuffer(ReplayBuffer):
    def __init__(self, max_size: int, replay_buffer_info: Dict, to_normalize: List[str]) -> None:
        super().__init__()
        self.replay_buffer = cpprb.ReplayBuffer(max_size, replay_buffer_info)
        self.max_size = max_size

        to_normalize_info = {key: replay_buffer_info[key] for key in to_normalize}

        self.online_normalization = OnlineNormalization(to_normalize_info=to_normalize_info)

    def add(self, transition_dict):
        self.replay_buffer.add(**transition_dict)
        self.online_normalization.update_stats(transition_dict)

    def sample(self, batch_size):
        sample = self.replay_buffer.sample(batch_size)

        normalized_sample = self.online_normalization.normalize(sample)

        return normalized_sample

    def save_checkpoint(self, save_dir, prefix=""):
        self.replay_buffer.save_transitions(os.path.join(save_dir,prefix + constants.FILE_REPLAY_BUFFER))
        self.online_normalization.save_checkpoint(save_dir, prefix + constants.FILE_REPLAY_BUFFER_PREFIX)

    def load_checkpoint(self, save_dir, prefix=""):
        self.replay_buffer.clear()
        self.replay_buffer.load_transitions(os.path.join(save_dir,prefix + constants.FILE_REPLAY_BUFFER))
        self.online_normalization.load_checkpoint(save_dir, prefix + constants.FILE_REPLAY_BUFFER_PREFIX)

class RedundancyReplayBuffer(ReplayBuffer):
    RB_KEY_REDUNDANCY_ID = "redundancy_id"

    def __init__(self, max_size: int, replay_buffer_info: Dict, to_normalize: List[str], redundancy_keys: List[str], num_redundancys: int) -> None:
        super().__init__()
        self.num_unique_entries = math.ceil(max_size/num_redundancys)

        redundancy_replay_buffer_info = {key:replay_buffer_info[key] for key in redundancy_keys}
        to_normalize_info = {key: replay_buffer_info[key] for key in to_normalize}

        if(len(redundancy_keys)>1):
            self.itemgetter = itemgetter(*redundancy_keys)
        else:
            self.itemgetter = lambda _dict: (itemgetter(*redundancy_keys)(_dict),)

        self.redundancy_buffer = cpprb.ReplayBuffer(self.num_unique_entries, redundancy_replay_buffer_info)

        for redundancy_key in redundancy_keys:
            del replay_buffer_info[redundancy_key]
            
        replay_buffer_info[self.RB_KEY_REDUNDANCY_ID] = {"shape": (1,), "dtype": np.int16}
        
        self.replay_buffer = cpprb.ReplayBuffer(max_size, replay_buffer_info)
        self.max_size = max_size
        self.num_redundancys = num_redundancys
        self.redundancy_keys = redundancy_keys

        self.online_normalization = OnlineNormalization(to_normalize_info=to_normalize_info)

    def add(self, transition_dict):
        self.online_normalization.update_stats(transition_dict)

        redundant_data = dict(zip(self.redundancy_keys, self.itemgetter(transition_dict)))

        redundancy_id = self.redundancy_buffer.add(**redundant_data)
        
        for key in self.redundancy_keys:
            del transition_dict[key]

        transition_dict[self.RB_KEY_REDUNDANCY_ID] = np.full((self.num_redundancys,1),redundancy_id, dtype=np.int16)

        self.replay_buffer.add(**transition_dict)

    def sample(self, batch_size):
        sample = self.replay_buffer.sample(batch_size)

        redundancy_ids = sample[self.RB_KEY_REDUNDANCY_ID].squeeze()
        del sample[self.RB_KEY_REDUNDANCY_ID]
        sample_update = self.redundancy_buffer._encode_sample(redundancy_ids)

        sample.update(sample_update)

        normalized_sample = self.online_normalization.normalize(sample)

        return normalized_sample

    def save_checkpoint(self, save_dir, prefix=""):
        self.replay_buffer.save_transitions(os.path.join(save_dir, prefix + constants.FILE_REPLAY_BUFFER))
        self.redundancy_buffer.save_transitions(os.path.join(save_dir, prefix + constants.FILE_REDUNDANCY_REPLAY_BUFFER))
        self.online_normalization.save_checkpoint(save_dir, prefix + constants.FILE_REPLAY_BUFFER_PREFIX)

    def load_checkpoint(self, save_dir, prefix=""):
        self.replay_buffer.clear()
        self.replay_buffer.load_transitions(os.path.join(save_dir,prefix + constants.FILE_REPLAY_BUFFER))
        self.redundancy_buffer.clear()
        self.redundancy_buffer.load_transitions(os.path.join(save_dir, prefix + constants.FILE_REDUNDANCY_REPLAY_BUFFER))
        self.online_normalization.load_checkpoint(save_dir,prefix+constants.FILE_REPLAY_BUFFER_PREFIX)