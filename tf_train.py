from mnistcls_model import *

train_data, train_labels, eval_data, eval_labels = load_data(20000, 5000)

logdir = "./tmp/{}_".format(NETNAME)
if os.path.exists(logdir) and os.path.isdir(logdir):
    shutil.rmtree(logdir)
os.mkdir(logdir)

# Create the Estimator
classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=logdir)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=10)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": train_data.astype(np.float32)},
    y=train_labels,
    batch_size=32,
    num_epochs=40,
    shuffle=True)

classifier.train(input_fn=train_input_fn)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input": eval_data.astype(np.float32)},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
