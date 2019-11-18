import tensorflow as tf
from matplotlib import pyplot as plt
from mnist_model import load_data, NETNAME, num2lab

test_size = 1000
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="{}.tflite".format(NETNAME))
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
batch_size = input_details[0]["shape"][0]

input_shape = input_details[0]['shape']
# Load training and eval data

_, _, eval_data, eval_labels = load_data(test_size)

preds = []
probs = []
labs = []
i = 0

while i < test_size:
    batch_data = eval_data[i:i+batch_size]
    batch_labs = eval_labels[i:i+batch_size]
    interpreter.set_tensor(input_details[0]['index'], batch_data)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])
    preds += list(logits.argmax(1))
    probs += list(logits.max(1))
    labs += list(batch_labs)
    i += batch_size

print("accuracy: {}%".format(sum([i == j for i, j in zip(preds, labs)])/len(preds)*100))

plt.subplots(figsize=(10, 5))
for i in range(10):
    conf = round(float(probs[i]), 3)
    plt.subplot(2, 5, i+1)
    plt.imshow(eval_data[i])
    plt.title("--{}--\n{:2.4}".format(num2lab[preds[i]], conf))
    plt.axis("off")
    plt.suptitle("tflite test", fontsize=18)
plt.savefig("./imgs/{}/res10-tflite.png".format(NETNAME))
plt.close()