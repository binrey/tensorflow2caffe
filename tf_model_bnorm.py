import tensorflow as tf
import numpy as np
import os
import shutil

tf.logging.set_verbosity(tf.logging.INFO)


def inner_structure(input_layer, is_training):
    x = tf.layers.batch_normalization(input_layer, training=is_training, name="bnorm0")
    # Convolutional Layer 1
    x = tf.layers.conv2d(
        name="conv1",
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        use_bias=False)
    # Batch normalization layer 1
    x = tf.layers.batch_normalization(x, training=is_training, name="bnorm1")
    # ReLu activation 1
    x = tf.nn.relu(x, "relu1")
    # Pooling Layer #1
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2, name="pool1")

    # Convolutional Layer 2
    x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        use_bias=False,
        name="conv2")
    # Batch normalization layer 2
    x = tf.layers.batch_normalization(x, training=is_training, name="bnorm2")
    # ReLu activation 2
    x = tf.nn.relu(x, name='relu2')
    # Pooling Layer 2
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2, name="pool2")
    # Flatten layer
    x = tf.layers.flatten(x)
    # Dense Layer 1
    x = tf.layers.dense(x, units=1024, activation=tf.nn.relu, name="dense1")
    #dropout = tf.layers.dropout(
    #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(x, units=10, name="dense2")
    return logits


def cnn_model_fn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 3])
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
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)

def load_for_infer(batch_size=None, writelogs=False, load_weights=True):
    sess = tf.Session()
    log_dir = "./logs"

    # Construct the model
    input_layer = tf.placeholder(dtype=tf.float32,
                                 shape=(batch_size, 28, 28, 3),
                                 name="input")

    logits = inner_structure(input_layer, is_training=False)

    classes = tf.argmax(input=logits, axis=1, name="classes")
    probs = tf.nn.softmax(logits, name="probs")
    if load_weights:
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint("./tmp"))
    if writelogs:
        infer_writer = tf.summary.FileWriter(os.path.join(log_dir), graph=sess.graph)
    return sess

if __name__ == "__main__":
    sess = load_for_infer(None, True, False)
    op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print("\n".join(op_names))