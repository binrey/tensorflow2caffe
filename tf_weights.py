# Convert weights from Tensorflow format to Caffe and save them

from utils import *
import numpy as np
from tf_model_bnorm import load_for_infer

sess = load_for_infer(1)
layers2save = ["conv", "bnorm", "dense"]

vars = tf.all_variables()
for var in vars:
    if not (var.name.startswith(tuple(layers2save))):
        continue
    value = sess.run(var)
    fname = "-".join(var.name.replace(":0", "").split("/")[:2])
    print("{}: {} -> ".format(fname, value.shape), end="")
    # Select first dense layer for special transformation
    if var.name == "dense1/kernel:0":
        value = transpose_weights(value, [7, 7, 64, -1], [2, 0, 1, 3])
    if len(value.shape)==4:
        value = value.transpose([3, 2, 0, 1])
    elif len(value.shape)==2:
        value = value.transpose([1, 0])
    print(value.shape)

    np.save("./tf_weights/{}.npy".format(fname), value)

