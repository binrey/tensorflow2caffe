import numpy as np
import tensorflow as tf
from make_tflite import batch_size

nbatches = 100

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
# Load training and eval data

((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
eval_data = np.stack([eval_data] * 3, axis=-1)
eval_labels = eval_labels

preds = []
labs = []
for i in range(nbatches):
    batch_data = eval_data[i:i+batch_size]
    batch_labs = eval_labels[i:i+batch_size]
    interpreter.set_tensor(input_details[0]['index'], batch_data)
    interpreter.invoke()
    preds += list(interpreter.get_tensor(output_details[0]['index']))
    labs += list(batch_labs)


print("accuracy: {}".format(sum([i == j for i, j in zip(preds, labs)])/batch_size/nbatches*100))
