from utils import *
from tf_model_bnorm import load_for_infer
import tensorflow as tf
import numpy as np

batch_size = 10
num_calibration_steps = 100

((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data.astype(np.float32)
train_data = np.stack([train_data] * 3, axis=-1)
train_labels = train_labels.astype(np.int32)

converter = tf.lite.TFLiteConverter.from_session(
    sess=load_for_infer(batch_size=batch_size),
    input_tensors=[tf.get_default_graph().get_tensor_by_name("input:0")],
    output_tensors=[tf.get_default_graph().get_tensor_by_name("classes:0")])

def representative_dataset_gen():
    for i in range(0, num_calibration_steps*batch_size, batch_size):
        yield [train_data[i:i+batch_size]]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_quant_model)
