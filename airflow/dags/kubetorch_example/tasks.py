import logging
import os
import sys
import time

import kubetorch as kt

S3_BUCKET_NAME = "my-simple-torch-model-example"
PYTORCH_IMAGE_ID = "nvcr.io/nvidia/pytorch:23.10-py3"

# Add the parent directory to Python path to allow imports (local development)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger(__name__)


def data_preprocessing_fn():
    logger.info("Data Preprocessing")
    return True


def data_preprocessing(**kwargs):
    image = kt.images.Debian()
    compute = kt.Compute(
        cpus="0.1",
        image=image,
        inactivity_ttl="10m",
    )
    logger.info("Step 1: Data Preprocessing")
    preprocessor = kt.fn(data_preprocessing_fn, name="data-preproc").to(compute)
    time.sleep(5)
    try:
        success = preprocessor()
        logger.info(f"Data Preprocessed: {success}")
    except Exception as e:
        logger.error(f"Data Preprocessing failed: {e}")
        raise e


# Uncomment to run locally
# if __name__ == "__main__":
#     data_preprocessing()


def run_training(**kwargs):
    from kubetorch_example.trainer import SimpleTrainer

    logger.info("Step 2: Run Training")
    compute = kt.Compute(
        gpus="1",
        image=kt.Image(image_id=PYTORCH_IMAGE_ID),
        launch_timeout=600,
        inactivity_ttl="10m",
    )

    model = kt.cls(SimpleTrainer).to(compute)

    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    model.load_data("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)
        model.test_model()
        model.save_model(
            bucket_name=S3_BUCKET_NAME,
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )


# Uncomment to run locally
# if __name__ == "__main__":
#     run_training()


def deploy_inference(**kwargs):
    from kubetorch_example.trainer import SimpleTrainer

    logger.info("Step 3: Deploy Inference")
    checkpoint_path = f"s3://{S3_BUCKET_NAME}/checkpoints/model_final.pth"
    local_checkpoint_path = "/model.pth"
    img = kt.Image(image_id=PYTORCH_IMAGE_ID).run_bash(
        f"aws s3 cp {checkpoint_path} {local_checkpoint_path}"
    )
    inference_compute = kt.Compute(
        gpus="1",
        image=img,
        launch_timeout=600,
        inactivity_ttl="10m",
    )

    inference = kt.cls(SimpleTrainer).to(
        inference_compute, init_args={"from_checkpoint": local_checkpoint_path}
    )
    # We distribute the inference service as an autoscaling pool of between 0 and 6 replicas, with a maximum concurrency of 16.
    inference.distribute(num_nodes=(0, 6), max_concurrency=16)


# Uncomment to run locally
# if __name__ == "__main__":
#     deploy_inference()
