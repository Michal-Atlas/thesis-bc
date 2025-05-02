import keras
import numpy as np

from test_rig.config import ONNX_MODEL_PATH, KERAS_MODEL_PATH
from test_rig.run.runner_class import Runner
import tvm.relay as relay
import tvm


class TVMRunner(Runner):

    def load_data(self):
        pass

    def __init__(self, device, cycles: int):
        super().__init__(cycles, device)
        self.model, self.params = relay.frontend.from_keras(
            keras.models.load_model(KERAS_MODEL_PATH)
        )

    def step(self):
        x = np.random.randn(1, 3, 224, 224)

        # https://tvm.apache.org/docs/v0.9.0/how_to/compile_models/from_onnx.html
        target = "llvm"

        with tvm.transform.PassContext(opt_level=1):
            executor = relay.build_module.create_executor(
                "graph",
                self.model,
                tvm.npu(0),
                target,
                self.params
            ).evaluate()

            tvm_output = executor(tvm.nd.array(x)).numpy()
            print(tvm_output)
