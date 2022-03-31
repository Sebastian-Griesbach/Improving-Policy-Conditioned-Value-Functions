from typing import Dict
import numpy as np
import os
import torch

class OnlineNormalization():

    AVOID_ZERO_DEVISION_CONSTANT = 1e-8
    FILE_SAVE_NAME = "online_normalization_stats"

    def __init__(self, to_normalize_info: Dict, n_bound: int = np.inf) -> None:
        self.means = {key: np.zeros((1,*to_normalize_info[key]["shape"])) for key in to_normalize_info.keys()}
        self.square_mean_diffs = {key: np.zeros((1,*to_normalize_info[key]["shape"])) for key in to_normalize_info.keys()}
        self.stds = {key: np.zeros((1,*to_normalize_info[key]["shape"])) for key in to_normalize_info.keys()}
        self.to_normalize_keys = to_normalize_info.keys()
        self.num_updates = {key: 0 for key in to_normalize_info.keys()}
        self.n_bound = n_bound

    def update_stats(self, data_dict):

        for key in set(data_dict.keys()) & set(self.to_normalize_keys):

            num_entries = data_dict[key].shape[0]
            for i in range(num_entries):

                self.num_updates[key] += 1
                n = self.num_updates[key] if self.num_updates[key] < self.n_bound else self.n_bound

                #Welford's online algorithm
                last_mean = self.means[key].copy()
                self.means[key] += (data_dict[key][i] - self.means[key]) /n
                self.square_mean_diffs[key] += (data_dict[key][i] - last_mean) * (data_dict[key][i] - self.means[key])
                var = self.square_mean_diffs[key] / (n-1) if n > 1 else np.square(self.means[key])
                self.stds[key] = np.sqrt(var)

    def normalize(self, data_dict):
        for key in set(data_dict.keys()) & set(self.to_normalize_keys):
            data_dict[key] = (data_dict[key] - self.means[key]) / (self.stds[key] + self.AVOID_ZERO_DEVISION_CONSTANT)

        return data_dict

    def save_checkpoint(self, save_dir, prefix=""):
        torch.save((self.means, self.stds, self.square_mean_diffs, self.num_updates), os.path.join(save_dir, prefix+self.FILE_SAVE_NAME+".pt"))

    def load_checkpoint(self, save_dir, prefix=""):
        self.means, self.stds, self.square_mean_diffs, self.num_updates = torch.load(os.path.join(save_dir, prefix+self.FILE_SAVE_NAME+".pt"))