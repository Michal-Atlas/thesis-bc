import cv2 as cv
import numpy as np

from test_rig.config import ONNX_MODEL_PATH, INPUT_SHAPE, TFLITE_FILE_PATH, INPUT_TYPE_NP
from test_rig.run.runner_class import Runner, DevType


class CVRunner(Runner):
    def __init__(self, cycles: int, device: DevType):
        super().__init__(cycles, device)

        if device == "cpu":
            backend = cv.dnn.DNN_BACKEND_DEFAULT
            target = cv.dnn.DNN_TARGET_CPU
        elif device == "npu":
            backend = cv.dnn.DNN_BACKEND_TIMVX
            target = cv.dnn.DNN_TARGET_NPU
        else:
            raise Exception(f"Device {self.device} is not supported")

        # self.m = cv.dnn.readNet(ONNX_MODEL_PATH)
        self.m = cv.dnn.readNet(TFLITE_FILE_PATH)
        self.m.setPreferableBackend(backend)
        self.m.setPreferableTarget(target)

    def load_data(self):
        self.m.setInput(np.full(
            fill_value=[1.0],
            shape=INPUT_SHAPE,
            dtype=INPUT_TYPE_NP,
        ))

    def step(self):
        return self.m.forward()
