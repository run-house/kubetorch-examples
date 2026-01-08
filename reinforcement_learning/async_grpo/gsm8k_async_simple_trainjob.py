import asyncio
from pathlib import Path

import kubetorch as kt
import yaml
from gsm8k_async_simple import simple_async_grpo

from inference import vLLM
from trainer import GRPOTrainer


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# TrainJob manifest for distributed training
TRAINJOB_CONTAINER = {
    "name": "kubetorch",
    "image": "pytorch/pytorch:latest",
    "resources": {
        "requests": {"nvidia.com/gpu": 1},
        "limits": {"nvidia.com/gpu": 1},
    },
}

TRAINJOB_MANIFEST = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainJob",
    "metadata": {
        "name": "",
        "namespace": "default",
    },
    "spec": {
        "runtimeRef": {"name": "torch-distributed"},
        "trainer": {"numNodes": 4},
        "template": {"spec": {"containers": [TRAINJOB_CONTAINER]}},
    },
}


async def main():
    from datasets import load_dataset

    config = load_config()
    MODEL_ID = config["model"]["id"]
    train_config = config.get("training", {})

    print("Loading GSM8K datasets...")
    dataset = load_dataset("gsm8k", "main", split="train")
    test_dataset = load_dataset("gsm8k", "main", split="test")

    # Inference compute with autoscaling
    inference_compute = kt.Compute(
        gpus="1",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").run_bash(
            "uv pip install --system --break-system-packages --no-deps "
            "-r async_grpo/requirements-inference.txt"
        ),
        launch_timeout=1200,
    ).autoscale(min_scale=2, initial_scale=2)

    # Training compute using TrainJob manifest
    num_workers = train_config.get("num_workers", 4)
    TRAINJOB_MANIFEST["spec"]["trainer"]["numNodes"] = num_workers

    train_image = kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").pip_install(
        [
            "'torch>=2.2.0'",
            "transformers==4.56.1",
            "datasets==4.1.0",
            "accelerate==1.10.1",
            "peft==0.17.1",
        ]
    )

    train_compute = kt.Compute.from_manifest(TRAINJOB_MANIFEST)
    train_compute.gpus = 1
    train_compute.image = train_image
    train_compute.launch_timeout = 600
    train_compute.allowed_serialization = ["json", "pickle"]
    train_compute.distributed_config = {"quorum_workers": num_workers}

    print("Deploying services...")
    engine_config = config.get("inference_engine", {})
    trainer_config = config.get("trainer", {})
    inference_service, train_service = await asyncio.gather(
        kt.cls(vLLM).to_async(
            inference_compute,
            init_args={"model_id": MODEL_ID, "engine_config": engine_config},
            get_if_exists=False,
        ),
        kt.cls(GRPOTrainer).to_async(
            train_compute,
            init_args={"model_id": MODEL_ID, "trainer_config": trainer_config},
            get_if_exists=False,
        ),
    )

    inference_service.async_ = True
    train_service.async_ = True

    await train_service.setup()

    await simple_async_grpo(
        dataset,
        test_dataset,
        train_service,
        inference_service,
        config,
    )


if __name__ == "__main__":
    asyncio.run(main())
