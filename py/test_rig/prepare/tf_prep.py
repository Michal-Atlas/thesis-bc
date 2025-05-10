import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import models, layers
import tensorflow_model_optimization as tfmot

from test_rig.config import SAVED_MODEL_PATH, TFLITE_FILE_PATH, INPUT_SHAPE, MODEL_LENGTH, KERAS_MODEL_PATH, \
    INPUT_TYPE_NP


# class TFModule(tf.Module):
#     def __init__(self):
#         super().__init__()
#
#     @tf.function(input_signature=[tf.TensorSpec(shape=INPUT_SHAPE, dtype=INPUT_TYPE_NP)])
#     def add(self, x):
#         x = tf.reduce_mean(x, axis=[1, 2])
#         return {"result": x}


def save():
    # Save the model
    # model = TFModule()

    # # https://www.tensorflow.org/datasets/keras_example
    # (ds_train, ds_test), ds_info = tfds.load(
    #     'mnist',
    #     split=['train', 'test'],
    #     shuffle_files=True,
    #     as_supervised=True,
    #     with_info=True,
    # )
    # # dataset_shape = ds_info.features.shape['image']
    # #
    # model = (tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(10)
    # ]))
    # # ta = tf.fill(
    # #     value=1e3,
    # #     dims=INPUT_SHAPE,
    # #     # dtype=torch.float32,
    # # )
    #
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # )
    # #
    # def normalize_img(image, label):
    #     """Normalizes images: `uint8` -> `float32`."""
    #     return tf.cast(image, tf.float32) / 255., label
    # #
    # ds_train = ds_train.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    # ds_train = ds_train.batch(128)
    # ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    # #
    # ds_test = ds_test.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = ds_test.batch(128)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    # #
    # model.fit(
    #     ds_train,
    #     epochs=6,
    #     validation_data=ds_test,
    # )
    #
    # # pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    # #     initial_sparsity=0.0, final_sparsity=0.5,
    # #     begin_step=2000, end_step=4000)
    #
    # # model.build()
    # model.save(KERAS_MODEL_PATH)
    # model.summary()

    model = (
        # tf.keras.applications.ConvNeXtTiny
        # tf.keras.applications.VGG19
        # tf.keras.applications.ResNet152
        # tf.keras.applications.MobileNetV3Large
        tf.keras.applications.MobileNetV3Small
        # tf.keras.applications.EfficientNetB0
        # tf.keras.applications.xception.Xception
            (
        input_shape=(224, 224, 3),
        # input_shape=(1280, 720, 3),
        include_top=False,
        # batch_size=1,
        # dtype=INPUT_TYPE_NP,
    ))
    # model = keras.applications.VGG16(
    #     input_shape=INPUT_SHAPE[1:],
    # )
    # model.compile()
    model.save(
        KERAS_MODEL_PATH,
    )

    # You can omit the signatures argument and a default signature name will be
    # created with name 'serving_default'.
    tf.saved_model.save(
        model,
        SAVED_MODEL_PATH,
        # # You can omit the signatures argument and a default signature name will be
        # # created with name 'serving_default'.
        # signatures={'my_signature': model.sig.get_concrete_function()},
    )

    # Convert the model using TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    converter.optimizations = [
        # Optimization allows quantization
        # Prevents i/o must be f32 errors
        tf.lite.Optimize.DEFAULT
    ]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_type = np.float32
    dtype = tf.float32
    converter.target_spec.supported_types = [
        # tf.float32,
        dtype,
    ]

    # https://stackoverflow.com/questions/56856262/how-to-quantize-inputs-and-outputs-of-optimized-tflite-model
    def representative_data_gen():
        # for data in ds_test.unbatch().batch(1):
        #     yield [data[0]]
        for _ in range(100):
            # Model has only one input so each data point has one element.
            yield [(np.random.rand(*INPUT_SHAPE).astype(np.float32))]

    converter.representative_dataset = representative_data_gen
    converter.inference_input_type = dtype
    # converter.inference_input_type = tf.float32
    converter.inference_output_type = dtype
    # converter.inference_output_type = tf.float32
    # converter.target_spec.supported_ops = [
    #     # tf.lite.OpsSet.SELECT_TF_OPS # Imports flex
    # ]
    tflite_model = converter.convert()
    with open(TFLITE_FILE_PATH, 'wb') as f:
        f.write(tflite_model)
