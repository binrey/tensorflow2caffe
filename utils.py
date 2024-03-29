#import tensorflow as tf
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt


def transpose_weights(m, last_conv_shape, transp_seq):
    """
    last_conv_shape - conv weights are written in format (w, h, c_conv, c_dense), where
    w - width, h - height, c_conv - channel of last conv layer, b - channels of first dense layer,
    For example first dense has shape [inp, out] - dense layer is written in format (inp, out), where
    inp - input flatten size, out - number of nodes in dense layer. Thats means [228, 1024] -> [(3, 3, 32), 1024],
    where conv weights are included. The pipeline of this function:
    1. reshape m from [228, 1024] back to [3, 3, 32, 1024]
    2. change dimensions in caffe order [32, 3, 3, 1024]
    3. reshape back to initial shape [228, 1024]
    """
    return m.reshape(last_conv_shape).transpose(transp_seq).reshape(m.shape)


def freeze_graph(sess):
    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names
    )

    # Finally we serialize and dump the output graph to the filesystem
    output_file = "./frozen_graph.pb"
    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


def simp_save():
    out_dir = "./saved_graph"
    if len(os.listdir(out_dir)):
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    sess = load_graph()
    g = tf.get_default_graph()
    inp_op = g.get_tensor_by_name("Reshape:0")
    out_op = g.get_tensor_by_name("ArgMax:0")
    inputs_dict = {
        "input": inp_op
    }
    outputs_dict = {
        "output": out_op
    }
    tf.saved_model.simple_save(
        sess, out_dir, inputs_dict, outputs_dict
    )
    print("saved model in {}".format(out_dir))


def vi_convs(op_names, resfun, out_name, title="convs outputs", conv_format="TF"):
    """ Outputs of conv layers """
    plt.close()
    fig, axs = plt.subplots(len(op_names), 10, figsize=(10, 1.3*len(op_names)))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
    nop = 0
    for op_name in op_names:
        lres = resfun(op_name)
        if len(lres.shape)<3:
            continue
        print("{}: mean={:.2f}, var={:.2f}".format(op_name.ljust(30, "."), lres.mean(), lres.var()))
        for i in range(10):
            ax = axs[nop, i]
            if conv_format == "TF":
                filt = lres[:, :, i] if i < lres.shape[-1] else np.zeros(lres.shape[:-1]).astype(np.uint8)
            elif conv_format == "Caffe":
                filt = lres[i, :, :] if i < lres.shape[0] else np.zeros(lres.shape[1:]).astype(np.uint8)
            ax.set_title("{}|{}".format(str(filt.min())[:4], str(filt.max())[:4]))
            if i == 0:
                ax.set_ylabel(op_name[:12])
            ax.tick_params('x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params('y', which='both', left=False, right=False, labelleft=False)
            ax.imshow(filt)
        nop += 1

    fig.suptitle(title, fontsize=18)
    plt.savefig(out_name)


def vi_denses(op_names, resfun, out_name, title="dense outputs"):
    """ Outputs of fully-connected layers """
    plt.close()
    fig, axs = plt.subplots(len(op_names), 1, figsize=(10, 1.6*len(op_names)))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.05)
    nop = 0
    for op_name in op_names:
        filt = resfun(op_name)
        if len(filt.shape)>1:
            continue
        ax = axs[nop]
        ax.bar(range(len(filt)), filt, 1)
        ax.grid(True)
        ax.set_ylabel(op_name)
        nop += 1
    fig.suptitle(title, fontsize=18)
    plt.savefig(out_name)


def vi_res10(eval_data, preds, num2lab, img_name, title):
    eval_data = eval_data.squeeze()
    plt.subplots(figsize=(10, 5))
    for i in range(10):
        label = preds[i].argmax()
        conf = round(preds[i][label], 3)
        plt.subplot(2, 5, i + 1)
        plt.imshow(eval_data[i])
        plt.title("--{}--\n{:2.4}".format(num2lab[label], conf))
        plt.axis("off")
        plt.suptitle(title, fontsize=18)
    plt.savefig(img_name)


def rename_tf_layer(name, lcounts):
    rename_dict = {"batch_norm": "bn",
                   "FusedBatchNorm_mul_0_param": "gamma",
                   "FusedBatchNorm_add_param": "beta"}

    name = name.replace(":0", "")
    targ_name = ""
    for k, v in rename_dict.items():
        if k in name:
            name = name.replace(k, v)

    if "bn" in name and "moving_mean" in name:
        targ_name = "moving-mean"
    if "bn" in name and "moving_variance" in name:
        targ_name = "moving-var"
    if "bn" in name and "beta" in name:
        targ_name = "bn-beta"
    if "bn" in name and "gamma" in name:
        targ_name = "bn-gamma"
    if "conv" in name and "kernel" in name:
        targ_name = "conv-kernel"
    if "conv" in name and "bias" in name:
        targ_name = "conv-bias"
    if "dense" in name and "kernel" in name:
        targ_name = "dense-kernel"
    if "dense" in name and "bias" in name:
        targ_name = "dense-bias"
    if len(targ_name):
        name = targ_name.replace("-", "{:02d}-".format(lcounts[targ_name]))
        lcounts[targ_name] += 1
        return name
    else:
        return None