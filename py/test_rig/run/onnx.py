import numpy as np
import onnxruntime as rt

from test_rig.config import ONNX_MODEL_PATH, INPUT_SHAPE


def run():
    session = rt.InferenceSession(ONNX_MODEL_PATH, providers=["VSINPUExecutionProvider"])
    session.run(
        input_feed=
        {
            "x": np.full(
                fill_value=[1.0],
                shape=INPUT_SHAPE,
                dtype=np.float32
            )
        },
        output_names=["x"]
    )

    input_tensor = np.full(
        fill_value=[1.0],
        shape=INPUT_SHAPE,
        dtype=np.float32
    )
    return session.run(None, {"x": input_tensor})
