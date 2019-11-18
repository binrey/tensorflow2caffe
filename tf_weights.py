# Convert weights from Tensorflow format to Caffe and save them
import tensorflow as tf
from mnist_model import NETNAME, last_conv_shape, load_for_infer
import numpy as np
import os
from utils import *

rootdir = os.path.join("./tf_weights/", NETNAME)
if os.path.exists(rootdir) and os.path.isdir(rootdir):
    shutil.rmtree(rootdir)
os.mkdir(rootdir)

layers_counts = {"moving-mean": 0,
                 "moving-var": 0,
                 "bn-beta": 0,
                 "bn-gamma": 0,
                 "conv-kernel": 1,
                 "conv-bias": 1,
                 "dense-kernel": 1,
                 "dense-bias": 1}

sess = load_for_infer()

with open(os.path.join("{}_selected_layers.txt".format(NETNAME)), "r") as f:
    layers2save = f.read().splitlines()

vars = tf.all_variables()
[print(var.name) for var in vars]

first_dense = None

print("{1}\nParse and save variables to ./tf_weights/{0}\n{1}".format(NETNAME, "".join(60*["-"])))
for var in vars:
    if not (var.name in layers2save):
        continue
    value = sess.run(var)
    fname = rename_tf_layer(var.name, layers_counts)#"-".join(var.name.replace(":0", "").split("/"))
    print("{:<50} -> {:<50} {} -> ".format(var.name, fname, value.shape), end="")
    # Select first dense layer for special transformation
    if first_dense is None:
        if "dense" in var.name and "kernel" in var.name:
            first_dense = True
            value = transpose_weights(value, last_conv_shape, [2, 0, 1, 3])
    if len(value.shape)==4:
        value = value.transpose([3, 2, 0, 1])
    elif len(value.shape)==2:
        value = value.transpose([1, 0])
    print(value.shape)

    np.save(os.path.join(rootdir, "{}.npy".format(fname)), value)

