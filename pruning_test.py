import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tf_model_bnorm import load_for_infer
from tf_testAll import calc_test

def sparse_variables(sparsity):
    vars = tf.global_variables()
    layers2save = ["dense", "conv"]

    for var in vars:
        fname = "-".join(var.name.replace(":0", "").split("/")[:2])
        if not (fname.startswith(tuple(layers2save))) or "kernel" not in fname:
            continue
        fvar = tf.reshape(var, [-1])
        sorted_ind = tf.argsort(tf.abs(fvar), axis=-1, direction="DESCENDING", stable=False, name=None)
        sorted_var = tf.gather(fvar, sorted_ind)
        tlen = tf.size(fvar)
        part_len = tf.cast(tf.multiply(tf.constant(1.-sparsity, tf.float32), tf.cast(tlen, tf.float32)), tf.int32)

        nonzero_range = tf.range(0, part_len, dtype=tf.int32)
        nonzero_part = tf.gather(sorted_var, nonzero_range)
        concat_var = tf.concat([nonzero_part, tf.zeros(tlen-part_len)], axis=0)
        re_ids = tf.scatter_nd(tf.reshape(sorted_ind, [-1, 1]), tf.range(0, tlen), [tlen])
        pruned_var = tf.gather(concat_var, re_ids)
        pruned_var = tf.reshape(pruned_var, var.shape)
        sess.run(tf.assign(var, pruned_var))

        value = sess.run(pruned_var)
        print("{}: {} -> sparsity {}%".format(fname, value.shape, (value == 0).sum()/value.flatten().shape[0]*100))

acc = []
sess = load_for_infer()
spars_vals = np.arange(0, 0.1, 0.1)
for sp in spars_vals:
    sparse_variables(sp)
    acc.append(calc_test(sess))

plt.plot(spars_vals, acc)
plt.grid("on")
plt.show()




