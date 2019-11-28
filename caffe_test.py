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
from utils import vi_convs, vi_denses

weights_folder = "./tf_weights/{}/".format(NETNAME)

np_weights = {}
layers = os.listdir(weights_folder)
layers.sort()
for l in layers:
    l = l.replace(".npy", "")# for w in os.listdir(weights_folder) if "kernel" in w]
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
img = Image.open("./imgs/test/{}/img3.png".format(data_folder)).resize(input_shape[:2])
img_arr = np.array(img).astype(np.float32)
if len(img_arr.shape)<3:
    img_arr = np.stack([img_arr] * 3, axis=-1)
img_arr = img_arr.transpose([2, 0, 1])
net.blobs["input"].data[...] = [img_arr]
res = net.forward()["prob"]

# Visualization of conv layers outputs
#conv_ops = ["input", "bnorm0", "conv1", "bnorm1", "relu1", "pool1", "conv2", "bnorm2", "relu2", "pool2"]
conv_ops = ["input", "conv01", "relu01", "conv02", "relu02"]

rootdir = "./imgs/{}".format(NETNAME)
#if os.path.exists(rootdir) and os.path.isdir(rootdir):
#    shutil.rmtree(rootdir)
#os.mkdir(rootdir)
def resfun(op_name):
    return net.blobs[op_name].data[0]
vi_convs(conv_ops, resfun, "{}/convs-caffe.png".format(NETNAME), "Caffe")

# Visualization of fully connected layers
denses_ops = ["flatten", "dense01", "dense02", "prob"]
vi_denses(denses_ops, resfun, "{}/denses-caffe.png".format(NETNAME))

# Run caffe model on test images
plt.subplots(figsize=(10, 5))
for i in range(10):
    img = np.array(Image.open("./imgs/test/{}/img{}.png".format(data_folder, i)).resize(input_shape[:2]))
    img = np.stack([img] * 3, axis=-1)
    net.blobs['input'].data[...] = img.transpose([2, 0, 1]).astype(np.float32)
    preds = net.forward()["prob"]

    label = preds[0].argmax()
    conf = round(preds[0][label], 3)
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title("--{}--\n{:2.4}".format(num2lab[label], conf))
    plt.axis("off")
    plt.suptitle("caffe test", fontsize=18)
plt.savefig("./imgs/{}/res10-caffe.png".format(NETNAME))