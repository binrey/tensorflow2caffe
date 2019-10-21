from utils import *
import numpy as np
from matplotlib import pyplot as plt
from main_bnorm import *

_, (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

eval_data = eval_data.astype(np.float32)
eval_data = np.stack([eval_data] * 3, axis=-1)
eval_labels = eval_labels.astype(np.int32)  # not required

sess = load_for_infer(100)
predicts = []
for i in range(100, 10100, 100):
    eval_batch = eval_data[i-100:i]
    predicts += list(sess.run(tf.get_default_graph().get_tensor_by_name("classes:0"),
                     feed_dict={"input:0": eval_batch}))

print("=======================================================================")
print("tot. accuracy: {}".format(sum(predicts == eval_labels)/len(eval_labels)))

