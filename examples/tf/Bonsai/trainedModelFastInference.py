
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=r"examples\tf\Bonsai\usps10\TFBonsaiResults\11_22_20_09_05_23\bonsai_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_data = np.load(r"examples\tf\Bonsai\usps10\test.npy")
input = np.array(input_data[:1,:], dtype="float32")
print(input)
interpreter.set_tensor(input_details[0]['index'], input)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
