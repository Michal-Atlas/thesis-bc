import os

import numpy as np
import tflite_runtime.interpreter as tfi

from test_rig.config import TFLITE_DELEGATE_PATH, TFLITE_FILE_PATH, INPUT_SHAPE, INPUT_TYPE_NP, MOBILENET_PATH
from test_rig.run.runner_class import Runner, DevType


class TFRunner(Runner):
    def __init__(
            self, cycles: int,
            device: DevType,
            model_path=TFLITE_FILE_PATH,
            dtype=INPUT_TYPE_NP,
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
            # num_threads=,
        )
        self.dtype = dtype
        self.model.allocate_tensors()

    def load_data(self):
        input_details = self.model.get_input_details()
        input_data = np.full(fill_value=[1e3], shape=input_details[0]['shape'],
                             # dtype=np.uint8 #INPUT_TYPE_NP
                             dtype=self.dtype
                             )
        self.model.set_tensor(input_details[0]['index'], input_data)

    def step(self):
        return self.model.invoke()
