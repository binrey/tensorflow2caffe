from signcls_model import load_for_infer, load_data, NETNAME, input_op, predict_op
import tensorflow as tf
import numpy as np

batch_size = 1
calibration_steps = 1000

if __name__ == "__main__":
    train_data, train_labels, *_ = load_data(calibration_steps*batch_size)
    train_data = train_data.astype(np.float32)

    converter = tf.lite.TFLiteConverter.from_session(
        sess=load_for_infer(batch_size),
        input_tensors=[tf.get_default_graph().get_tensor_by_name(input_op+":0")],
        output_tensors=[tf.get_default_graph().get_tensor_by_name(predict_op+":0")])

    def representative_dataset_gen():
        for i in range(0, calibration_steps):
            yield [train_data[i*batch_size : (i+1)*batch_size]]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    with open("{}.tflite".format(NETNAME), "wb") as f:
        f.write(tflite_quant_model)
