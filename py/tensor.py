import tensorflow as tf

import torch.nn as nn

class TestModel(nn.Module):
  def __init__(self):
    super(TestModel, self).__init__()

  # @tf.function(input_signature=[tf.TensorSpec(shape=[1, 10], dtype=tf.float32)])
  def forward(self, x):
    '''
    Simple method that accepts single input 'x' and returns 'x' + 4.
    '''
    # Name the output 'result' for convenience.
    return {'result' : x + 21}

SAVED_MODEL_PATH = 'content/saved_models/test_variable'
TFLITE_FILE_PATH = 'content/test_variable.tflite'

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
#tf.saved_model.save(
#    module, SAVED_MODEL_PATH,
#    signatures={'my_signature':module.add.get_concrete_function()})

# Convert the model using TFLiteConverter
#converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
#tflite_model = converter.convert()
#with open(TFLITE_FILE_PATH, 'wb') as f:
#  f.write(tflite_model)

