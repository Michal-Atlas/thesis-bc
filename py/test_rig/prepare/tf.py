import tensorflow as tf

from test_rig.config import SAVED_MODEL_PATH, TFLITE_FILE_PATH, INPUT_SHAPE, INPUT_TYPE_NP


class TFModule(tf.Module):
    def __init__(self):
        super().__init__()


    @tf.function(input_signature=[tf.TensorSpec(shape=INPUT_SHAPE, dtype=INPUT_TYPE_NP)])
    def add(self, x):
        return {"result": x + 4}


def save():
    # Save the model
    module = TFModule()
    # You can omit the signatures argument and a default signature name will be
    # created with name 'serving_default'.
    tf.saved_model.save(
        module, SAVED_MODEL_PATH,
        # # You can omit the signatures argument and a default signature name will be
        # # created with name 'serving_default'.
        signatures={'my_signature': module.add.get_concrete_function()}
    )

    # Convert the model using TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    tflite_model = converter.convert()
    with open(TFLITE_FILE_PATH, 'wb') as f:
        f.write(tflite_model)
