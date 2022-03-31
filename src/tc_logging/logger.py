from abc import ABC, abstractmethod
import time
import os

class Logger(ABC):
    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.log_dir = log_dir
        if(not os.path.isdir(log_dir)):
            os.mkdir(log_dir)

        self.run = f"run{int(time.time())}"
        log_path = os.path.join(log_dir, self.run)
        self.log_path = os.path.abspath(log_path)

    @abstractmethod
    def log_scalar(self, tag, scalar, step):
        ...

    @abstractmethod
    def log_tensor(self, tag, tensor, step):
        ...

    @abstractmethod
    def flush(self):
        ...

    @abstractmethod
    def save_checkpoint(self, save_dir, prefix=""):
        ...

    @abstractmethod
    def load_checkpoint(self, save_dir, prefix=""):
        ...

class LogReader(ABC):
    def __init__(self, log_path) -> None:
        super().__init__()
        self.log_path = log_path

    @abstractmethod
    def read_scalar(self, tag):
        ...

    @abstractmethod
    def read_tensor(self, tag):
        ...

