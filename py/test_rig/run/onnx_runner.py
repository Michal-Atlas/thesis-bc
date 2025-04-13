from typing import List, Union, Literal

import numpy as np
import onnxruntime as rt
from onnxruntime import NodeArg

from test_rig.config import ONNX_MODEL_PATH, INPUT_SHAPE, INPUT_TYPE_NP, MOBILENET_PATH
from test_rig.run.runner_class import Runner


class ONNXRunner(Runner):
    def run(self, device, cycles):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "npu":
            providers = ["VSINPUExecutionProvider"]
        elif device == "gpu":
            providers = ["CUDAExecutionProvider"]
        else:
            raise Exception(f"Device {device} is not supported")
        session = rt.InferenceSession(
            ONNX_MODEL_PATH,
            providers=providers
        )

        input_tensor = np.full(
            fill_value=[np.random.randn()],
            shape=INPUT_SHAPE,
            dtype=INPUT_TYPE_NP,
        )
        output_names = [n.name for n in session.get_outputs()]
        input_names = [n.name for n in session.get_inputs()]
        for i in range(cycles):
            print(f"\rCycle {i}/{cycles}...", end="")

            session.run(
                input_feed={k: input_tensor for k in input_names},
                output_names=output_names,
            )
        print()
