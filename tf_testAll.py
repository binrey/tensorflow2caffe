import tensorflow as tf
import numpy as np
from tf_model_bnorm import load_for_infer


def calc_test(sess, test_size=1000):
    _, (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

    eval_data = eval_data.astype(np.float32)
    eval_data = np.stack([eval_data] * 3, axis=-1)
    eval_labels = eval_labels.astype(np.int32)  # not required

    predicts = []
    for i in range(0, test_size, 100):
        eval_batch = eval_data[i:i+100]
        predicts += list(sess.run(tf.get_default_graph().get_tensor_by_name("classes:0"),
                         feed_dict={"input:0": eval_batch}))

    return sum(predicts == eval_labels[:test_size])/len(eval_labels)


if __name__ == "__main__":
    accuracy = calc_test(load_for_infer())
    print("=======================================================================")
    print("tot. accuracy: {}".format(accuracy))
