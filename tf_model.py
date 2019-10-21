import tensorflow as tf
import numpy as np
import os
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

def inner_structure(input_layer):
    use_bias = True
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        name="conv1",
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        use_bias=use_bias)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(name="pool1", inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        name="conv2",
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        use_bias=use_bias)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(name="pool2", inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.layers.flatten(pool2)
    dense = tf.layers.dense(name="dense1", inputs=pool2_flat, units=1024, activation=tf.nn.relu, use_bias=use_bias)
    #dropout = tf.layers.dropout(
    #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(name="dense2", inputs=dense, units=10, use_bias=use_bias)
    return logits

def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 3])

    logits = inner_structure(input_layer)

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
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
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

def load_for_infer(batch_size=None, writelogs=False):
    sess = tf.Session()
    log_dir = "./logs"

    # Construct the model
    input_layer = tf.placeholder(dtype=tf.float32,
                                 shape=(batch_size, 28, 28, 3),
                                 name = "input")

    logits = inner_structure(input_layer)

    classes = tf.argmax(input=logits, axis=1, name="classes")
    probs = tf.nn.softmax(logits, name="probs")

    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=tf.train.latest_checkpoint("./tmp"))
    if writelogs:
        infer_writer = tf.summary.FileWriter(os.path.join(log_dir), graph=sess.graph)
    return sess

if __name__ == "__main__":
    load_for_infer()