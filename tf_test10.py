import tensorflow as tf
import numpy as np
from main_bnorm import cnn_model_fn
from matplotlib import pyplot as plt
from PIL import Image


eval_data = np.zeros([10, 28, 28, 3], dtype=np.uint8)
for i in range(10):
    img_arr = np.array(Image.open("./imgs/test/img{}.png".format(i)))
    img_arr = np.stack([img_arr] * 3, axis=-1)
    eval_data[i] = img_arr

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    warm_start_from="./tmp")

pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data.astype(np.float32)},
    batch_size=1,
    shuffle=False)

preds = list(mnist_classifier.predict(input_fn=pred_input_fn))

plt.subplots(figsize=(10, 5))
for i in range(10):
    label = preds[i]["classes"]
    conf = round(preds[i]["probabilities"][label], 3)
    plt.subplot(2, 5, i+1)
    plt.imshow(eval_data[i])
    plt.title("{} : {:2.4}".format(label, conf))
    plt.axis("off")
    plt.suptitle("tensorflow test", fontsize=18)
plt.savefig("./imgs/res10-tf.png")
plt.close()
