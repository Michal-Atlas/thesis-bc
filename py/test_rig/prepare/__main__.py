import torch

from test_rig.prepare import tf, onnx, tvm_prep

if __name__ == "__main__":
    # torch.set_default_dtype(torch.uint8)
    for m in [
        # onnx,
        tf,
        # tvm_prep
    ]:
        m.save()
