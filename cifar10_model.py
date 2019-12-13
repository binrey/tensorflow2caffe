import tensorflow as tf
import numpy as np
import os
import shutil
from matplotlib import pyplot as plt
import pickle

tf.logging.set_verbosity(tf.logging.INFO)

NETNAME = "cifar10"
num2lab = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
input_op = "input"
input_shape = [32, 32, 3]
last_conv_shape = [2, 2, 128]
predict_op = "probs"
data_folder = "mnist"


def load_data(trainsize, validsize):
    # Load training and eval data
    train_data, train_labels, eval_data, eval_labels = \
        load_cifar_10_data('./data/cifar10-batches-py', 1)

    train_data = train_data[:trainsize].astype(np.float32)
    train_labels = train_labels[:trainsize].astype(np.int32)

    eval_data = eval_data[:validsize].astype(np.float32)
    eval_labels = eval_labels[:validsize].astype(np.int32)

    return train_data, train_labels, eval_data, eval_labels


def inner_structure(input_layer, is_training):
    x = tf.add(input_layer, tf.constant(-128, tf.float32), name="input/Add")
    ks = [3, 3]
    for i in range(0, 2):
        # Convolutional Layer 1
        x = tf.layers.conv2d(
            name="conv{}1".format(i),
            inputs=x,
            filters=32*(i+1),
            kernel_size=ks[i],
            padding="same",
            use_bias=False)
        # Batch normalization layer 1
        x = tf.layers.batch_normalization(x, training=is_training, name="bn{}1".format(i))
        # ReLu activation 1
        x = tf.nn.relu(x, "relu{}1".format(i))
        # Convolutional Layer 2
        x = tf.layers.conv2d(
            name="conv{}2".format(i),
            inputs=x,
            filters=32*(i+1),
            kernel_size=ks[i],
            padding="same",
            use_bias=False)
        # Batch normalization layer 2
        x = tf.layers.batch_normalization(x, training=is_training, name="bn{}2".format(i))
        # ReLu activation 1
        x = tf.nn.relu(x, "relu{}2".format(i))
        # Pooling Layer 1
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2, name="pool{}".format(i))

    # Flatten layer
    x = tf.layers.flatten(x)
    # Dense Layer 1
    x = tf.layers.dense(x, units=512, activation=tf.nn.relu, name="dense1")
    x = tf.layers.dropout(x, rate=0.5, training=is_training)
    # Logits Layer
    logits = tf.layers.dense(x, units=10, name="output")
    return logits


def cnn_model_fn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False

    input_layer = tf.reshape(features["input"], [-1]+input_shape)
    logits = inner_structure(input_layer, is_training)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gradients, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }

    train_writer = tf.summary.FileWriter(os.path.join("./logs"), graph=tf.get_default_graph())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)


def load_for_infer(batch_size=None, writelogs=False, load_weights=True):
    sess = tf.Session()
    log_dir = "./logs"

    # Construct the model
    input_layer = tf.placeholder(dtype=tf.float32,
                                 shape=[batch_size]+input_shape,
                                 name="input")

    logits = inner_structure(input_layer, is_training=False)

    classes = tf.argmax(input=logits, axis=1, name="classes")
    probs = tf.nn.softmax(logits, name="probs")
    if load_weights:
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint("./tmp/" + NETNAME))
    if writelogs:
        infer_writer = tf.summary.FileWriter(os.path.join(log_dir), graph=sess.graph)
    return sess

def load_ckpt(rootpath):
    sess = tf.Session()
    ckpt_last = tf.train.latest_checkpoint(rootpath)
    tf.train.import_meta_graph(ckpt_last+".meta")
    sess = tf.Session(graph=sess.graph)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_last)
    return sess

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


def load_cifar_10_data(data_dir, nparts=6, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, nparts+1):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_labels, cifar_test_data, cifar_test_labels


if __name__ == "__main__":
    """show it works"""

    cifar_10_dir = './data/cifar10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

    #sess = load_for_infer(None, True, False)

    #sess = load_ckpt("./tmp")
    #grad_ops = [n for n in tf.get_default_graph().as_graph_def().node if n.name.startswith("grad")]
    #sess.run(fetches=["gradients/conv1/Conv2D_grad/Conv2DBackpropFilter:0"])