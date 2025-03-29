import torch.nn as nn
import torch

from test_rig.config import ONNX_MODEL_PATH, INPUT_SHAPE

class OnnxModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def save():
    model = OnnxModule()

    onnx_program = torch.onnx.export(
        model,
        (torch.randn(*INPUT_SHAPE),),
        dynamo=True,
        report=True,

        optimize=True,
        verify=True,
    )

    onnx_program.save(ONNX_MODEL_PATH)
