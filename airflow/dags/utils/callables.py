import logging
import time

import kubetorch as kt

S3_BUCKET_NAME = "my-simple-torch-model-example"
PYTORCH_IMAGE_ID = "nvcr.io/nvidia/pytorch:23.10-py3"


def data_preprocessing_fn():
    logger = logging.getLogger(__name__)
    logger.info("Data Preprocessing")
    return True


def data_preprocessing_callable(**kwargs):
    # Configure logging to use Airflow's handlers
    logger = logging.getLogger(__name__)

    logger.info("Step 1: Data Preprocessing")
    image = kt.Image()  # .pip_install(["torch"])
    compute = kt.Compute(
        cpus="0.1",
        image=image,
        inactivity_ttl="0s",  # Because this is production, destroy immediately on completion
    )
    logger.info("Step 1: Data Preprocessing B")
    preprocessor = kt.fn(data_preprocessing_fn, name="data-preproc").to(compute)
    logger.info("Step 1: Data Preprocessing C")
    time.sleep(5)
    logger.info("Step 1: Data Preprocessing D")
    logger.info(f"Data Preprocessor: {preprocessor}")
    try:
        success = preprocessor()
        logger.info(f"Data Preprocessed: {success}")
    except Exception as e:
        logger.error(f"Data Preprocessing failed: {e}")
        raise e


# We can simply put the dispatch and execution of the model in the callable identical to
# how we have run it locally, ensuring identical research-to-production execution.
def run_training_callable(**kwargs):
    from utils.simple_trainer import SimpleTrainer

    # Configure logging to use Airflow's handlers
    logger = logging.getLogger(__name__)

    logger.info("Step 2: Run Training")
    compute = kt.Compute(
        gpus="1",
        image=kt.Image(image_id=PYTORCH_IMAGE_ID),
        inactivity_ttl="0s",  # Because this is production, destroy immediately on completion
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


# We deploy a new service for inference with the trained model checkpoint. Note that we are defining a new compute
# object rather than reusing the training compute above. Note that we load down the model weights in the image
# to achieve faster cold start times for our inference service.
def deploy_inference_callable(**kwargs):
    from utils.simple_trainer import SimpleTrainer

    # Configure logging to use Airflow's handlers
    logger = logging.getLogger(__name__)

    logger.info("Step 3: Deploy Inference")
    checkpoint_path = f"s3://{S3_BUCKET_NAME}/checkpoints/model_final.pth"
    local_checkpoint_path = "/model.pth"
    img = kt.Image(image_id=PYTORCH_IMAGE_ID).run_bash(
        f"aws s3 cp {checkpoint_path} {local_checkpoint_path}"
    )
    inference_compute = kt.Compute(
        gpus="1",
        image=img,
    )

    inference = kt.cls(SimpleTrainer).to(
        inference_compute, init_args={"from_checkpoint": local_checkpoint_path}
    )
    # We distribute the inference service as an autoscaling pool of between 0 and 6 replicas, with a maximum concurrency of 16.
    inference.distribute(num_nodes=(0, 6), max_concurrency=16)


# if __name__ == "__main__":
#     data_preprocessing_callable()
