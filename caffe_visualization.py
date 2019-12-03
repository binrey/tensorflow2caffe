import sys
sys.path.append("/home/rybin/My/caffe-jacinto/python")
import matplotlib
matplotlib.use('Agg')
import caffe
from mnistcls_model import *

import shutil
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils import vi_convs, vi_denses, vi_res10

savedir = "./imgs/{}/caffe".format(NETNAME)
if os.path.exists(savedir) and os.path.isdir(savedir):
    shutil.rmtree(savedir)
os.mkdir(savedir)

weights_folder = "./tf_weights/{}/".format(NETNAME)
with open(os.path.join("./selected_layers", NETNAME, "caffe_ops2show.txt"), "r") as f:
    ops2show = f.read().splitlines()
conv_ops = ops2show[:ops2show.index("---")]
dense_ops = ops2show[ops2show.index("---")+1:]

np_weights = {}
layers = os.listdir(weights_folder)
layers.sort()
for l in layers:
    l = l.replace(".npy", "")
    if "kernel" in l:
    # Load kernels
        data = np.load(os.path.join(weights_folder, l+".npy"))
        np_weights.update({l: data})
        print("{}:{}".format(l, data.shape))
    else:
    # load biases
        data = np.load(os.path.join(weights_folder, l+".npy"))
        np_weights.update({l: data})
        print("{}:{}".format(l, data.shape))

# Load caffe model from prototxt file and assign weights to layers
net = caffe.Net('nets/{}.prototxt'.format(NETNAME), caffe.TEST)
for layer_name in net.params.keys():
    if layer_name.startswith("moving"):
        name_adds = ["-mean", "-var"]
    elif layer_name.startswith("bn"):
        name_adds = ["-gamma", "-beta"]
    elif layer_name.startswith(tuple(["conv", "dense"])):
        name_adds = ["-kernel", "-bias"]
    else:
        continue
    for i, nadd in enumerate(name_adds):
        if i < len(net.params[layer_name]):
            net.params[layer_name][i].data[...] = np_weights[layer_name + nadd].astype(np.float32)
    print(layer_name)



# Load test image for visualizations
img_num = 3
eval_data = np.zeros([10]+[input_shape[-1]]+input_shape[:2], dtype=np.uint8)
for i in range(10):
    img_arr = np.array(Image.open("./imgs/test/{}/img{}.png".format(data_folder, i)).resize(input_shape[0:2]))
    if len(img_arr.shape) == 2:
        img_arr = np.expand_dims(img_arr, axis=0)
    eval_data[i] = img_arr

net.blobs["input"].data[...] = eval_data
res = net.forward()["prob"]


# Visualization of conv layers outputs
def resfun(op_name):
    return net.blobs[op_name].data[img_num]
vi_convs(conv_ops, resfun, os.path.join(savedir, "convs.png"), "caffe convs outputs", "Caffe")
# Visualization of fully connected layers
vi_denses(dense_ops, resfun, os.path.join(savedir, "denses.png"), "caffe dense outputs")
# Run caffe model on test images
vi_res10(eval_data, res, num2lab, os.path.join(savedir, "res10.png"), "caffe test-10")