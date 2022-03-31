import numpy as np

from tc_logging.logger import Logger, LogReader

class LogHandler():
    def __init__(self,
                logger: Logger,
                log_reader: LogReader):

        self.logger = logger
        self.log_reader = log_reader

    def log_data(self, data_dict, step):
        for tag, value in data_dict.items():
            if(type(value) != np.ndarray):
                self.logger.log_scalar(tag=tag, scalar=value, step=step)
            else:
                self.logger.log_tensor(tag=tag, tensor=value, step=step)

    def flush_logger(self):
        self.logger.flush()

    def read_data(self, tag, tensor=False):
        if(not tensor):
            return self.log_reader.read_scalar(tag=tag)
        else:
            return self.log_reader.read_tensor(tag=tag)

    def save_checkpoint(self, save_dir, prefix=""):
        self.logger.save_checkpoint(save_dir, prefix)

    def load_checkpoint(self, save_dir, prefix=""):
        self.logger.load_checkpoint(save_dir, prefix)
        self.log_reader.log_path = self.logger.log_path

