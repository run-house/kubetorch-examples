# # TensorFlow Multi-node Distributed Training
# A basic example showing how to use Kubetorch to Pythonically run a TensorFlow distributed training script on
# multiple GPUs. We use the `TF_CONFIG` environment variable to set up the distributed training environment, and
# create a separate worker for each rank. We then call the replicas concurrently to trigger coordinated
# multi-node training. We're using two single-GPU instances (and therefore two ranks) with the
# MultiWorkerMirroredStrategy, but this same strategy could be used for other TensorFlow distributed strategies.
#
# Despite it being common to use a launcher script to start distributed training, this approach is more flexible and
# allows for more complex orchestration, such as running multiple training jobs concurrently, handling exceptions,
# running distributed training alongside other tasks on the same cluster. It's also significantly easier to debug
# and monitor, as you can see the output of each rank in real-time and get stack traces if a worker fails.

import json
import os

import kubetorch as kt
import tensorflow as tf


# This is the function that will be run on each worker. It initializes the distributed training environment,
# creates a simple model and optimizer, and runs a training loop.
def train_process():
    # Initialize the distributed training environment,
    # per https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    tf_config = json.loads(os.environ["TF_CONFIG"])
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    num_workers = strategy.num_replicas_in_sync
    print(f"Worker {tf_config['task']['index']} of {num_workers} initialized")

    # Create a simple model and optimizer
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation="relu")])
    optimizer = tf.keras.optimizers.SGD(0.01)

    with strategy.scope():
        model.compile(optimizer=optimizer, loss="mse")

    model.fit(
        tf.data.Dataset.from_tensor_slices(
            (tf.random.normal([1000, 10]), tf.random.normal([1000, 1]))
        ).batch(32)
    )

    print(f"Worker {tf_config['task']['index']} finished")


if __name__ == "__main__":
    # Dispatch the training function to a multi-node cluster with 4 nodes, each with 1 GPU
    gpus = kt.Compute(
        gpus="1", image=kt.Image(image_id="nvcr.io/nvidia/tensorflow:25.02-tf2-py3")
    ).distribute("tensorflow", workers=4)
    remote_train = kt.fn(train_process).to(gpus)

    remote_train()
