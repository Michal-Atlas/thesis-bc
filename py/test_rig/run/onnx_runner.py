from typing import List, Union, Literal

import numpy as np
import onnxruntime as rt

from test_rig.config import ONNX_MODEL_PATH, INPUT_SHAPE, INPUT_TYPE_NP, MOBILENET_TFLITE_PATH
from test_rig.run.runner_class import Runner, DevType


class ONNXRunner(Runner):
    def __init__(self, cycles: int, device: DevType, model_path=ONNX_MODEL_PATH):
        super().__init__(cycles, device)
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif self.device == "npu":
            providers = ["VSINPUExecutionProvider"]
        elif self.device == "gpu":
            providers = ["CUDAExecutionProvider"]
        else:
            raise Exception(f"Device {self.device} is not supported")

        options = rt.SessionOptions()
        # options.enable_profiling = True

        self.session = rt.InferenceSession(
            model_path,
            options,
            providers=providers,
        )

        self.output_names = [n.name for n in self.session.get_outputs()]
        self.inputs = {n.name: np.full(
            fill_value=[np.random.randn()],
            shape=n.shape,
            dtype=INPUT_TYPE_NP,
        ) for n in self.session.get_inputs()}

    def load_data(self):
        pass

    def step(self):
        self.session.run(
            input_feed=self.inputs,
            output_names=self.output_names,
        )
        # prof_file = self.session.end_profiling()
        # print(prof_file)
