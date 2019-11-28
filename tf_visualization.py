from mnistcls_model import *
from PIL import Image
from utils import vi_convs, vi_denses
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


savedir = "./imgs/{}".format(NETNAME)
#if os.path.exists(savedir) and os.path.isdir(savedir):
#    shutil.rmtree(savedir)
#os.mkdir(savedir)

sess = load_for_infer()
op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
print("\n".join(op_names))

img_num = 3

eval_data = np.zeros([10]+input_shape, dtype=np.uint8)
for i in range(10):
    img_arr = np.array(Image.open("./imgs/test/{}/img{}.png".format(data_folder, i)).resize(input_shape[0:2]))
    img_arr = np.stack([img_arr] * 3, axis=-1)
    eval_data[i] = img_arr

def resfun(op_name):
    op = sess.graph.get_tensor_by_name(op_name+":0")
    return sess.run(op, feed_dict={input_op+":0": [eval_data[img_num]]})[0]

# Visualize conv and dense layers
vi_denses(denses_ops, resfun, "{}/denses-tf.png".format(NETNAME))
vi_convs(conv_ops, resfun, "{}/convs-tf.png".format(NETNAME))

# Test model on 10 test images
preds = sess.run(sess.graph.get_tensor_by_name(predict_op+":0"),
                 {input_op+":0": eval_data.astype(np.float32)})
plt.subplots(figsize=(10, 5))
for i in range(10):
    label = preds[i].argmax()
    conf = round(preds[i][label], 3)
    plt.subplot(2, 5, i+1)
    plt.imshow(eval_data[i])
    plt.title("--{}--\n{:2.4}".format(num2lab[label], conf))
    plt.axis("off")
    plt.suptitle("tensorflow test", fontsize=18)
plt.savefig(savedir+"/res10-tf.png")
plt.close()
