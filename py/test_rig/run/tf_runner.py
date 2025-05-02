import os

import numpy as np
import tflite_runtime.interpreter as tfi

from test_rig.config import TFLITE_DELEGATE_PATH, TFLITE_FILE_PATH, INPUT_TYPE_NP, BATCH_SIZE
from test_rig.run.runner_class import Runner, DevType


# import tensorflow.lite as tfi


class TFRunner(Runner):
    def __init__(
            self, cycles: int,
            device: DevType,
            model_path=TFLITE_FILE_PATH,
            dtype=INPUT_TYPE_NP,
            reshape=True,
    ):
        super().__init__(cycles, device)
        if device == "cpu":
            delegates = []
        elif device == "npu":
            os.environ["USE_GPU_INFERENCE"] = "0"
            delegates = [
                tfi.load_delegate(TFLITE_DELEGATE_PATH, [])
            ]
        # elif device == "gpu":
        #     os.environ["USE_GPU_INFERENCE"] = "1"
        #     delegates = [
        #         tfi.load_delegate(TFLITE_DELEGATE_PATH,[])
        #     ]
        else:
            raise Exception(f"Device {self.device} is not supported")

        self.model = tfi.Interpreter(
            model_path,
            experimental_delegates=delegates,
        )
        self.dtype = dtype
        self.reshape = reshape
        self.model.allocate_tensors()

    def load_data(self):
        input_details = self.model.get_input_details()
        for input_detail in input_details:
            shape = input_detail['shape']
            size = (BATCH_SIZE, *shape[1:]) if self.reshape else shape
            input_data = np.full(fill_value=[1e3], shape=size,
                                 # dtype=np.uint8 #INPUT_TYPE_NP
                                 dtype=self.dtype
                                 )
            if self.reshape:
                self.model.resize_tensor_input(input_detail['index'], input_data.shape)
                self.model.allocate_tensors()
            self.model.set_tensor(input_detail['index'], input_data)

    def step(self):
        return self.model.invoke()
