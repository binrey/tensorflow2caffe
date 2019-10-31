from tf_model_bnorm import *

# Load training and eval data
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data.astype(np.float32)
train_data = np.stack([train_data] * 3, axis=-1)[:10000]
train_labels = train_labels.astype(np.int32)[:10000]

eval_data = eval_data.astype(np.float32)
eval_data = np.stack([eval_data] * 3, axis=-1)[:1000]
eval_labels = eval_labels.astype(np.int32)[:1000]

logdir = "./tmp"
if len(os.listdir(logdir)):
    shutil.rmtree(logdir)
    os.mkdir(logdir)

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=logdir)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=10)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=50,
    shuffle=True)

mnist_classifier.train(input_fn=train_input_fn)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
