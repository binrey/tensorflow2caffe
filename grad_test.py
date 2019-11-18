from pruning_test import sparse_np_variables
from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tf_testAll


# Define graph
# ----------------------------------------------------------------------------------------------------------------------
is_training = True
input_layer = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3], name="input")
labels_op = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="labs")

x = tf.layers.batch_normalization(input_layer, training=is_training, name="bnorm0", epsilon=1e-5)
# Convolutional Layer 1
x = tf.layers.conv2d(
    name="conv1",
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    use_bias=False)
# Batch normalization layer 1
x = tf.layers.batch_normalization(x, training=is_training, name="bnorm1", epsilon=1e-5)
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
x = tf.layers.batch_normalization(x, training=is_training, name="bnorm2", epsilon=1e-5)
# ReLu activation 2
x = tf.nn.relu(x, name='relu2')
# Pooling Layer 2
x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=2, name="pool2")
# Flatten layer
x = tf.layers.flatten(x)
# Dense Layer 1
x = tf.layers.dense(x, units=1024, activation=tf.nn.relu, name="dense1")
# dropout = tf.layers.dropout(
#    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
# Logits Layer
logits = tf.layers.dense(x, units=10, name="dense2")
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_op, logits=logits)
predict_op = tf.argmax(input=logits, axis=1, name="pclasses")
acc, acc_op = tf.metrics.accuracy(labels=labels_op, predictions=predict_op, name="acc")

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
masks = [tf.placeholder(dtype=np.float32, shape=None, name="mask_{}".format(i)) for i in range(12)]
with tf.control_dependencies(update_ops):
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    masked_grads = [g*m for g, m in zip(gradients, masks)]
    train_op = optimizer.apply_gradients([tuple([g, v]) for g, v in zip(masked_grads, variables)],
                                         global_step=tf.train.get_global_step())

def plog(epoch):
    sess.run(tf.local_variables_initializer())
    logloss, _ = sess.run([loss, acc_op], {"input:0": valid_data, "labs:0": valid_labs})
    logacc = sess.run(acc)
    print("{:>3} - {:6.3f} {:6.3f}".format(epoch, logloss, logacc))

def sparsity_test():
    spars = []
    for tf_var in tf.global_variables():
        if (tf_var.name.startswith(tuple(["conv", "dense"]))) and "kernel" in tf_var.name:
            test_var = sess.run(tf_var)
            spars.append((test_var == 0).sum() / test_var.size * 100)
    print("sparsity test >>> min = {:.2f}%, max = {:.2f}%".format(min(spars), max(spars)))

# Load graph and load weights
# ----------------------------------------------------------------------------------------------------------------------
sess = tf.Session()
sess.run(tf.local_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess=sess, save_path=tf.train.latest_checkpoint("./tmp"))
sparsity_test()

# Load data
# ----------------------------------------------------------------------------------------------------------------------
test_size = 1000

((train_data, train_labs),
 (valid_data, valid_labs)) = tf.keras.datasets.mnist.load_data()

train_data = train_data.astype(np.float32)
train_data = np.stack([train_data] * 3, axis=-1)[:10000]
train_labs = np.expand_dims(train_labs, -1)[:10000]

valid_data = valid_data.astype(np.float32)
valid_data = np.stack([valid_data] * 3, axis=-1)[:test_size]
valid_labs = np.expand_dims(valid_labs, -1)[:test_size]

# Start test
# ----------------------------------------------------------------------------------------------------------------------

plog(0)
sparse_shedule = [0.97]
for i, sp in enumerate(sparse_shedule):
    mvalues = sparse_np_variables(sess, sp, variables)

    fdict = {i: d for i, d in zip(masks, mvalues)}

    # Start retrain
    plog(0)
    batch_size = 100
    maxit = int(60/(len(sparse_shedule)-i))
    for epoch in range(maxit):
        for i in range(0, train_data.shape[0], batch_size):
            fdict.update({"input:0": train_data[i:i+batch_size], "labs:0": train_labs[i:i+batch_size]})
            sess.run(fetches=train_op, feed_dict=fdict)
        plog(epoch+1)

    sparsity_test()
