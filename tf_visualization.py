from mnistcls_model import *
from PIL import Image
from utils import vi_convs, vi_denses, vi_res10
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


savedir = "./imgs/{}/tf".format(NETNAME)
if os.path.exists(savedir) and os.path.isdir(savedir):
    shutil.rmtree(savedir)
os.mkdir(savedir)

sess = load_for_infer()
op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
print("\n".join(op_names))

img_num = 3

eval_data = np.zeros([10]+input_shape, dtype=np.uint8)
for i in range(10):
    img_arr = np.array(Image.open("./imgs/test/{}/img{}.png".format(data_folder, i)).resize(input_shape[0:2]))
    if len(img_arr.shape) == 2:
        img_arr = np.expand_dims(img_arr, axis=-1)
    eval_data[i] = img_arr


def resfun(op_name):
    op = sess.graph.get_tensor_by_name(op_name+":0")
    return sess.run(op, feed_dict={input_op+":0": [eval_data[img_num]]})[0]

# Visualize conv and dense layers
rootdir = "./imgs/{}/tf".format(NETNAME)
vi_convs(conv_ops, resfun, os.path.join(rootdir, "convs.png"))
vi_denses(denses_ops, resfun, os.path.join(rootdir, "denses.png"))

# Test model on 10 test images
res = sess.run(sess.graph.get_tensor_by_name(predict_op+":0"),
              {input_op+":0": eval_data.astype(np.float32)})
vi_res10(eval_data, res, num2lab, os.path.join(rootdir, "res10.png"))