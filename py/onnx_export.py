import torch

import tensor

model_path = "content/test_variable.tflite"

model = tensor.TestModel()

onnx_program = torch.onnx.export(model, torch.randn(1,280,2800), dynamo=True, report=True)

onnx_program.save("model.onnx")
