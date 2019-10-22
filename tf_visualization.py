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

img_num = 3

eval_data = np.zeros([10, 28, 28, 3], dtype=np.uint8)
for i in range(10):
    img_arr = np.array(Image.open("./imgs/test/img{}.png".format(i)))
    img_arr = np.stack([img_arr] * 3, axis=-1)
    eval_data[i] = img_arr

def resfun(op_name):
    op = sess.graph.get_tensor_by_name(op_name + ":0")
    return sess.run(op, feed_dict={"input:0": [eval_data[img_num]]})[0]

# Visualize conv and dense layers
vi_denses(denses_ops, resfun, "denses-tf.png")
vi_convs(conv_ops, resfun, "convs-tf.png")

# Test model on 10 test images
preds = sess.run(sess.graph.get_tensor_by_name("probs:0"), feed_dict={"input:0": eval_data})
plt.subplots(figsize=(10, 5))
for i in range(10):
    label = preds[i].argmax()
    conf = round(preds[i][label], 3)
    plt.subplot(2, 5, i+1)
    plt.imshow(eval_data[i])
    plt.title("{} : {:2.4}".format(label, conf))
    plt.axis("off")
    plt.suptitle("tensorflow test", fontsize=18)
plt.savefig("./imgs/res10-tf.png")
plt.close()
