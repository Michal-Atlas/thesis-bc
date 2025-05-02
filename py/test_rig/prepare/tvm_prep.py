def save():
    model = relay.frontend.from_onnx(
        onnx.load(ONNX_MODEL_PATH)
    )


