import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.keras import models, layers

from test_rig.config import SAVED_MODEL_PATH, TFLITE_FILE_PATH, INPUT_SHAPE, INPUT_TYPE_NP, MODEL_LENGTH


class TFModule(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=INPUT_SHAPE, dtype=INPUT_TYPE_NP)])
    def add(self, x):
        for _ in range(MODEL_LENGTH):
            # [filter_height, filter_width, in_channels, out_channels]
            conv_weights = tf.Variable(tf.random.normal([3, 3, 3, 32]))
            x = tf.nn.conv2d(
                x,
                conv_weights,
                strides=[1, 1, 1, 1],
                padding='SAME',
            )
            x = tf.nn.relu(x)
            depthwise_weights = tf.Variable(tf.random.normal([3, 3, 32, 1]))
            x = tf.nn.depthwise_conv2d(
                x,
                depthwise_weights,
                strides=[1, 1, 1, 1],
                padding='SAME',
            )
            x = tf.nn.relu(x)

        # x = tf.nn.avg_pool2d
        x = tf.reduce_mean(x, axis=[1, 2])
        return {"result": x}


def save():
    # Save the model
    # model = TFModule()

    model = models.Sequential()
    ta = tf.fill(
        value=1e3,
        dims=INPUT_SHAPE,
        # dtype=torch.float32,
    )
    model.add(layers.InputLayer(
        # input_shape=INPUT_SHAPE[1:],
        input_tensor=ta,
    ))
    for i in range(MODEL_LENGTH):
        model.add(layers.Conv2D(
            32,
            (1, 1),
            activation='relu',
            # input_shape=(320, 320, 3)  # (32, 32, 3)
        ))
        model.add(layers.DepthwiseConv2D(
            (3, 3),
            strides=[1, 1],
            padding='SAME',
            depth_multiplier=1,
        ))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(units=10, activation='softmax'))

    # You can omit the signatures argument and a default signature name will be
    # created with name 'serving_default'.
    tf.saved_model.save(
        model, SAVED_MODEL_PATH,
        # # You can omit the signatures argument and a default signature name will be
        # # created with name 'serving_default'.
        # signatures={'my_signature': model.sig.get_concrete_function()}
    )

    # Convert the model using TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_type = np.uint8
    # https://stackoverflow.com/questions/56856262/how-to-quantize-inputs-and-outputs-of-optimized-tflite-model
    # def representative_data_gen():
    #     for input_value in range(100):
    #         # Model has only one input so each data point has one element.
    #         yield [(torch.full(INPUT_SHAPE,
    #                            fill_value=input_value,
    #                            dtype=torch.float32,
    #                            ))]
    #
    # converter.representative_dataset = representative_data_gen
    # converter.inference_input_type = tf.uint8
    # converter.inference_input_type = tf.float32
    # converter.inference_output_type = tf.uint8
    # converter.target_spec.supported_ops = [
    #     # tf.lite.OpsSet.SELECT_TF_OPS # Imports flex
    # ]
    tflite_model = converter.convert()
    with open(TFLITE_FILE_PATH, 'wb') as f:
        f.write(tflite_model)
