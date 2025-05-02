import numpy as np

SAVED_MODEL_PATH = 'test_model'
KERAS_MODEL_PATH = f'{SAVED_MODEL_PATH}.keras'
TFLITE_FILE_PATH = f'{SAVED_MODEL_PATH}.tflite'
ONNX_MODEL_PATH = f'{SAVED_MODEL_PATH}.onnx'
MOBILENET_TFLITE_PATH = '/usr/bin/tensorflow-lite-2.16.2/examples/mobilenet_v1_1.0_224_quant.tflite'
MOBILENET_ONNX_PATH = 'mobilenet_v1_1.0_224_quant.onnx'
TFLITE_DELEGATE_PATH = '/usr/lib/libvx_delegate.so'

# INPUT_SHAPE = (32, 32, 16, 16)
# INPUT_SHAPE = (1, 224, 512, 3)
INPUT_SHAPE = (1, 224, 224, 3)
CYCLES = 5
INPUT_TYPE_NP = np.float32
# INPUT_TYPE_NP = np.int32
MODEL_LENGTH = 12
NPU_DEBUG = False
NPU_CACHE = True
BATCH_SIZE = 512