import tensorflow as tf
import sys
import numpy as np
samplereader_root = "/home/rybin/My/sign-classifier"
sys.path.append(samplereader_root)
import defaults as defs
from SampleGenerator import SampleReader

NETNAME = "signcls"

num2lab = ['5','10','15','20','20_off','25','30','30_off','40','40_off','50','50_off','60',\
           '60_off','70','70_off','80','80_off','90','100','100_off','110','120','130',\
           'Pesh_perehod','Pesh_perehod_red','Ostorojno_deti','Ustupit_dorogu','Dorojnie_raboti',\
           'Stop','Obgon_zapret','Obgon_ne_zapret','Nerovnost','Nerovnost_red','Glav_doroga',\
           'No_limits','No_entry','Trash']

conv_ops = ["inputs/input", "inputs/batch_norm/FusedBatchNorm", "conv_40/Relu", "conv_41/Relu", "conv_32/Relu"]
denses_ops = ["flatten", "dense/add", "output"]
input_op = "inputs/input"
input_shape = [48, 48, 3]
last_conv_shape = [-1, 6, 6, 128]
predict_op = "output"
data_folder = "signs"

def load_for_infer(bsize=None):
    rootpath = "./tmp/" + NETNAME
    ckpt_last = tf.train.latest_checkpoint(rootpath)
    tf.train.import_meta_graph(ckpt_last + ".meta")
    sess = tf.Session()
    tf.train.Saver().restore(sess=sess, save_path=ckpt_last)
    return sess


def load_data(datasize):
    loader = SampleReader(data_bases=[samplereader_root+"/data/itelma/",
                                      samplereader_root+"/data/mgu/",
                                      ],
                          sel_signs_file=samplereader_root+"/index.csv",
                          colorspace=defs.ColorSpace.RGB,
                          aug=None,
                          synt=None,
                          valid_size=0.2
                          )

    train_data, train_labels, _ = loader.load_batch_train(datasize)
    eval_data, eval_labels, eval_signs = loader.load_batch_eval(datasize)
    return train_data, train_labels.argmax(1), eval_data, eval_labels.argmax(1)

#if __name__ == "__main__":
#    load_data(10)