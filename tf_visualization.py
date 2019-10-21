from tf_model_bnorm import load_for_infer
from PIL import Image
from utils import vi_convs, vi_denses
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

conv_ops = ["input", "bnorm0/FusedBatchNorm", "conv1/Conv2D", "bnorm1/FusedBatchNorm", "relu1",
            "pool1/MaxPool", "conv2/Conv2D", "bnorm2/FusedBatchNorm", "relu2", "pool2/MaxPool"]
denses_ops = ["flatten/Reshape", "dense1/Relu", "dense2/BiasAdd", "probs"]

sess = load_for_infer()
op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
print("\n".join(op_names))

img_num = 0
img = np.array(Image.open("./imgs/test/img{}.png".format(img_num)), dtype=np.float32)
img = np.stack([img] * 3, axis=-1)


def resfun(op_name):
    op = sess.graph.get_tensor_by_name(op_name + ":0")
    return sess.run(op, feed_dict={"input:0": [img]})[0]

vi_denses(denses_ops, resfun, "denses-tf.png")
vi_convs(conv_ops, resfun, "convs-tf.png")

