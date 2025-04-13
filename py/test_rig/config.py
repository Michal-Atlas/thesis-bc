import numpy as np

SAVED_MODEL_PATH = 'test_model.raw'
TFLITE_FILE_PATH = 'test_model.tflite'
ONNX_MODEL_PATH = 'test_model.onnx'
MOBILENET_PATH = '/usr/bin/tensorflow-lite-2.16.2/examples/mobilenet_v1_1.0_224_quant.tflite'
TFLITE_DELEGATE_PATH = '/usr/lib/libvx_delegate.so'

INPUT_SHAPE = (32, 32, 16, 16)

INPUT_TYPE_NP = np.float32
