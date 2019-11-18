import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mnist_model import load_for_infer
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


def sparse_np_variables(session, sparsity, vars=None):
    if not vars:
        vars = tf.global_variables()
    layers2save = ["dense", "conv"]
    masks = []
    for i, var in enumerate(vars):
        fname = "-".join(var.name.replace(":0", "").split("/")[:2])
        if not (fname.startswith(tuple(layers2save))) or "kernel" not in fname:
            masks.append(np.ones(var.shape))
            continue
        np_var = session.run(var)#np.random.randint(0, 10, 10)#
        fvar = np_var.flatten()
        sorted_ind = np.argsort(-np.abs(fvar))
        re_ind = np.argsort(sorted_ind)
        sorted_var = fvar[sorted_ind]
        tot_len = fvar.size
        keep_len = int(np.rint((1-sparsity)*tot_len))
        ones_part = np.ones(keep_len)
        zeros_part = np.zeros(tot_len - keep_len)
        mask = np.hstack([ones_part, zeros_part])
        mask = mask[re_ind].reshape(np_var.shape)
        masked_var = np_var*mask
        session.run(tf.assign(var, masked_var))
        masks.append(mask)
        test_sp = (masked_var == 0).sum()/masked_var.size*100
        print("{:15}: {:15} -> sparsity {:6.2f}%".format(fname, str(masked_var.shape), test_sp))
    return masks


if __name__ == "__main__":
    acc = []
    sess = load_for_infer()
    spars_vals = np.arange(0, 0.1, 0.1)
    for sp in spars_vals:
        m = sparse_np_variables(sess, sp)
        acc.append(calc_test(sess))

    plt.plot(spars_vals, acc)
    plt.grid("on")
    plt.show()




