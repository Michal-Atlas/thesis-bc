import tflite_runtime.interpreter as tfi
import numpy as np

from test_rig.config import TFLITE_FILE_PATH


# from ai_edge_litert.interpreter import Interpreter
def run():
    interpreter = tfi.Interpreter(
        TFLITE_FILE_PATH,
        experimental_delegates=[
            tfi.load_delegate("/usr/lib/libvx_delegate.so",[])
        ])
    interpreter.allocate_tensors()

    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    my_signature = interpreter.get_signature_runner()

    # my_signature is callable with input as arguments.
    return my_signature(x=np.full(fill_value=[1e3], shape=(200,200), dtype=np.float32))
