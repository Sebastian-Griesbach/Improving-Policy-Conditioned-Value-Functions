import torch
import os

from tc_logging.logger import Logger, LogReader

class SimpleLogger(Logger):
    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir = log_dir)
        self.log_path = self.log_path+".pt"
        self.logged_data = {}

    def log_scalar(self, tag, scalar, step):
        self._log_value(tag = tag, value=scalar, step=step)

    def log_tensor(self, tag, tensor, step):
        self._log_value(tag = tag, value=tensor, step=step)

    def _log_value(self, tag, value, step):
        tag_dict = self.logged_data.get(tag, {})
        tag_dict[step] = value
        self.logged_data[tag] = tag_dict

    def flush(self):
        torch.save(self.logged_data, self.log_path)

    def save_checkpoint(self, save_dir, prefix=""):
        torch.save((self.log_path, self.logged_data), os.path.join(save_dir, prefix+"log.pt"))

    def load_checkpoint(self, save_dir, prefix=""):
        self.log_path, self.logged_data = torch.load(os.path.join(save_dir, prefix+"log.pt"))

class SimpleLogReader(LogReader):
    def __init__(self, log_path) -> None:
        super().__init__(log_path = log_path)

    def read_scalar(self, tag):
        return self._load_data(tag)

    def read_tensor(self, tag):
        return self._load_data(tag)

    def _load_data(self, tag):
        return torch.load(self.log_path)[tag]