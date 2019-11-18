import sys
import os
from signcls_model import NETNAME, last_conv_shape
from utils import *
sys.path.append("/home/rybin/My/quant/flatbuffers")
from tflite import Model
import numpy as np
import shutil

layers_counts = {"bn_moving-mean": 0,
                 "bn_moving-var": 0,
                 "bn-beta": 0,
                 "bn-gamma": 0,
                 "conv-kernel": 1,
                 "conv-bias": 1,
                 "dense-kernel": 1,
                 "dense-bias": 1}

num2type = [np.float32, np.float16, np.int32, np.uint8, np.int64, np.str, np.bool, np.int16, np.complex, np.int8]
root_dir = "./tflite_weights/"+NETNAME

if os.path.exists(root_dir) and os.path.isdir(root_dir):
    shutil.rmtree(root_dir)
os.mkdir(root_dir)
os.mkdir(root_dir+"/int")
os.mkdir(root_dir+"/float")

model = Model.Model.GetRootAsModel(open("./"+NETNAME+".tflite", "rb").read(), 0)
graph = model.Subgraphs(0)

# Parse names of tflite model and convert them into standart format
tname2id = {}
first_dense = False
print("__________\ntensors in tflite model:")
for i in range(graph.TensorsLength()):
    tname = graph.Tensors(i).Name().decode("utf-8")
    fname = rename_tf_layer(tname, layers_counts)
    if fname:
        print("{:<50} -> {:<50} : {}".format(tname, fname, i))
        tname2id.update({i: fname})

print("__________\nids to tensors names dict:")
print(tname2id)

print("__________\nsummary of parsed layers:")
for i, name in tname2id.items():
    tensor = graph.Tensors(i)
    buff = model.Buffers(tensor.Buffer()).DataAsNumpy()
    quants = tensor.Quantization().ScaleAsNumpy()
    tshape = tensor.ShapeAsNumpy()
    bytes_bshape = buff.shape[0]

    convert_type = np.int32 if "bias" in name else np.int8
    buff = np.frombuffer(buff.tobytes(), dtype=convert_type)
    buff = buff.reshape(tshape)
    first_bshape = buff.shape

    if first_dense is False:
        if "dense" in name and "kernel" in name:
            first_dense = True
            buff = transpose_weights(buff, last_conv_shape, [0, 3, 1, 2])
    if len(buff.shape)==4:
        buff = buff.transpose([0, 3, 1, 2])
    # Multiply int-weights to quantization factor in order to get float-weights
    if len(tshape) > 2:
        qbuff = np.array([q * b for q, b in zip(quants, buff)])
    else:
        qbuff = quants * buff

    minmax = (buff.min(), buff.max())

    type = num2type[tensor.Type()]

    print("{} {} -> {} -> {} -> {}".format(
        name[:40].ljust(30, "."),
        str(bytes_bshape).ljust(8, " "),
        str(type).replace("<class 'numpy.", "").replace("'>", "").ljust(6, " "),
        str(first_bshape).ljust(16, " "),
        str(buff.shape).ljust(16, " "),
        minmax))

    np.save(os.path.join("./tflite_weights", NETNAME, "int", name), buff)
    np.save(os.path.join("./tflite_weights", NETNAME, "float" , name), qbuff)