import cv2 as cv
import numpy as np

from test_rig.config import ONNX_MODEL_PATH


def run():
    m = cv.dnn.readNet(ONNX_MODEL_PATH)
    m.setPreferableBackend(cv.dnn.DNN_BACKEND_TIMVX)
    m.setPreferableTarget(cv.dnn.DNN_TARGET_NPU)
    m.setInput(np.full(fill_value=[1.0], shape=(1,28,28)))
    return m.forward()
