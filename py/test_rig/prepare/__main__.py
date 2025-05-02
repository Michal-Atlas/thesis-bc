import torch

import tensorflow as tf
from test_rig.prepare import tf_prep, onnx_prep, tvm_prep

if __name__ == "__main__":
    # torch.set_default_dtype(torch.uint8)
    for m in [
        onnx_prep,
        # tf_prep,
        # tvm_prep,
    ]:
        m.save()
