import numpy as np
import tflite_runtime.interpreter as tfi

from test_rig.config import TFLITE_DELEGATE_PATH, MOBILENET_PATH
from test_rig.run.runner_class import Runner

class TFRunner(Runner):
    def run(self, device):
        model = tfi.Interpreter(
            MOBILENET_PATH,
            # TFLITE_FILE_PATH,
            experimental_delegates=[
                tfi.load_delegate(TFLITE_DELEGATE_PATH,[])
            ],
            # num_threads=,
        )
        model.allocate_tensors()


        input_details = model.get_input_details()
        output_details = model.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        input_data = np.full(
            fill_value=[243],
            shape=(1, width, height, 3),
            dtype=np.uint8,
        )
        model.set_tensor(input_details[0]['index'], input_data)

        model.invoke()

        output_data = model.get_tensor(output_details[0]['index'])

        return output_data

        # # There is only 1 signature defined in the model,
        # # so it will return it by default.
        # # If there are multiple signatures then we can pass the name.
        # my_signature = model.get_signature_runner()
        #
        # # my_signature is callable with input as arguments.
        # return my_signature(x=np.full(fill_value=[1e3], shape=INPUT_SHAPE, dtype=INPUT_TYPE_NP))
