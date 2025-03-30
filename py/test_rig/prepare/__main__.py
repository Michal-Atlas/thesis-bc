from test_rig.prepare import tf, onnx, tvm_prep

if __name__ == "__main__":
    for m in [onnx,tf,tvm_prep]:
        m.save()