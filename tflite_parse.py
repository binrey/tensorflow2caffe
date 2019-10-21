import sys
import os
from utils import *
sys.path.append("/home/rybin/My/quant/flatbuffers")
from tflite import Model
import numpy as np

num2type = [np.float32, np.float16, np.int32, np.uint8, np.int64, np.str, np.bool, np.int16, np.complex, np.int8]
weight_layers = ["conv1", "conv2", "dense1", "dense2"]

model = Model.Model.GetRootAsModel(open('./model.tflite', 'rb').read(), 0)
graph = model.Subgraphs(0)

# Parse names of tflite model and convert them into standart format
tname2id = {}
print("__________\ntensors in tflite model:")
for i in range(graph.TensorsLength()):
    tname = graph.Tensors(i).Name().decode("utf-8")
    print(tname)
    tname = tname.split("/")
    if len(tname)<2:
        continue
    if "bias" in tname[1]:
        tname2id.update({i: "-".join([tname[0], "bias"])})
    if  "kernel" in tname[1]:
        tname2id.update({i: "-".join([tname[0], "kernel"])})
print("__________\nids to tensors names dict:")
print(tname2id)

print("__________\nsummary of parsed layers:")
for i, tname in tname2id.items():
    tensor = graph.Tensors(i)
    buff = model.Buffers(tensor.Buffer()).DataAsNumpy()
    quants = tensor.Quantization().ScaleAsNumpy()
    tshape = tensor.ShapeAsNumpy()
    bytes_bshape = buff.shape[0]

    convert_type = np.int32 if "bias" in tname else np.int8
    buff = np.frombuffer(buff.tobytes(), dtype=convert_type)
    buff = buff.reshape(tshape)
    first_bshape = buff.shape
    if tname == "dense1-kernel":
        buff = transpose_weights(buff, [-1, 7, 7, 64], [0, 3, 1, 2])

    if len(buff.shape)==4:
        buff = buff.transpose([0, 3, 1, 2])
    # Multiply int-weights to quantization factor to achive float-weights
    if len(tshape) > 2:
        qbuff = np.array([q * b for q, b in zip(quants, buff)])
    else:
        qbuff = quants * buff

    minmax = (buff.min(), buff.max())

    type = num2type[tensor.Type()]

    print("{} {} -> {} -> {} -> {}".format(
        tname[:40].ljust(30, "."),
        str(bytes_bshape).ljust(8, " "),
        str(type).replace("<class 'numpy.", "").replace("'>", "").ljust(6, " "),
        str(first_bshape).ljust(16, " "),
        str(buff.shape).ljust(16, " "),
        minmax))

    np.save(os.path.join("./tflite_weights","int", tname), buff)
    np.save(os.path.join("./tflite_weights","float", tname), qbuff)