from typing import Text
import tensorflow as tf
import io
import json
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import warnings

from tc_utils.util import numpy_to_tensor, tensor_to_numpy
from tc_logging.logger import Logger, LogReader

class TensorBoardLogger(Logger):
    def __init__(self, log_dir: str):
        super().__init__(log_dir = log_dir)

        self.file_writer = tf.summary.create_file_writer(self.log_path)

    def log_scalar(self, tag, scalar, step):
        with self.file_writer.as_default():
            tf.summary.scalar(tag, scalar, step=step)

    def log_tensor(self, tag, tensor, step):
        numpy_array = tensor_to_numpy(tensor)
        serialized = self._serialize_numpy_array(numpy_array)
        self._log_text(tag, serialized, step)

    def flush(self):
        pass
    
    def save_checkpoint(self, save_dir, prefix=""):
        warnings.warn("Checkpoint function not implemented for Tensorboard Logger.")

    def load_checkpoint(self, save_dir, prefix=""):
        warnings.warn("Checkpoint function not implemented for Tensorboard Logger.")

    def _log_text(self, tag, text, step):
        with self.file_writer.as_default():
            tf.summary.text(tag, text, step=step)

    def _serialize_numpy_array(self, np_array: np.array) -> str:
        try:
            memfile = io.BytesIO()
            np.save(memfile, np_array)
            memfile.seek(0)
            serialized = json.dumps(memfile.read().decode('latin-1'))
            memfile.close()
            return serialized
        except Exception as err:
            raise Exception(f'An exception occured while serializing a numpy array: \n{err}')

class TensorBoardLogReader(LogReader):
    def __init__(self, log_path: str, size_guide: int = 10000000) -> None:
        super().__init__(log_path=log_path)

        self.size_guidance = {event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                            event_accumulator.IMAGES: 0,
                            event_accumulator.AUDIO: 0,
                            event_accumulator.SCALARS: 0,
                            event_accumulator.HISTOGRAMS: 0,
                            event_accumulator.TENSORS: size_guide,}

        self.event_acc = event_accumulator.EventAccumulator(log_path, self.size_guidance)
        self.event_acc.Reload()

    def read_scalar(self, tag):
        self.event_acc.Reload()
        return_dict = {}
        for entry in self.event_acc.Tensors(tag):
            step = entry.step
            scalar = self._decode_scalar(entry.tensor_proto)
            return_dict[step] = scalar

        return return_dict

    def read_tensor(self, tag):
        self.event_acc.Reload()
        return_dict = {}
        for entry in self.event_acc.Tensors(tag):
            step = entry.step
            bitstring = entry.tensor_proto.string_val[0]
            np_array = self._deserialize_numpy_array(bitstring)
            return_dict[step] = numpy_to_tensor(np_array)

        return return_dict

    def _decode_scalar(self, val):
        tensor_bytes = val.tensor_content
        tensor_dtype = val.dtype
        tensor_shape = [x.size for x in val.tensor_shape.dim]
        tensor_array = tf.io.decode_raw(tensor_bytes, tensor_dtype)
        tensor_array = tf.reshape(tensor_array, tensor_shape)
        return tensor_array.numpy()

    def _deserialize_numpy_array(self, serialized: str) -> np.array:
        try:
            memfile = io.BytesIO()
            memfile.write(json.loads(serialized).encode('latin-1'))
            memfile.seek(0)
            np_array = np.load(memfile)
            memfile.close()
            return np_array
        except Exception as err:
            raise Exception(f'An exception occured while deserializing a numpy array: \n{err}')