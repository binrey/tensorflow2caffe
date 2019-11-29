tf_conv_ops = ["input", "conv1/Conv2D", "relu1", "conv2/Conv2D", "relu2"]
tf_denses_ops = ["flatten/Reshape", "dense1/Relu", "dense2/BiasAdd", "probs"]

caffe_conv_ops = ["input/bias", "conv01", "relu01", "conv02", "relu02"]
caffe_denses_ops = ["flatten", "dense01", "dense02", "prob"]