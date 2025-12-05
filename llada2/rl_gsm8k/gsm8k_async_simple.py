"""Async GRPO Training for LLaDA2 on GSM8K.

Entry point for reinforcement learning with Group Relative Policy Optimization.
Uses AsyncGRPOScheduler for coordinating inference and training.
"""
import asyncio
import os

import kubetorch as kt

from llada2.inference.sglang_engine import SGLang
from llada2.rl_gsm8k.math_agent import SimpleMathAgent
from llada2.rl_gsm8k.scheduler import AsyncGRPOScheduler
from llada2.rl_gsm8k.trainer import GRPOTrainer


def load_config():
    """Load training config from YAML."""
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(split: str):
    """Load GSM8K dataset."""
    from datasets import load_dataset
    return load_dataset("gsm8k", "main", split=split)


async def deploy_services(config, get_if_exists=True):
    """Deploy inference and training services via kubetorch.

    Returns:
        Tuple of (inference_service, train_service)
    """
    # Inference image (SGLang)
    inference_img = (
        kt.Image(image_id="lmsysorg/sglang:v0.5.6")
        .run_bash(
            "uv pip install --break-system-packages --system "
            "'git+https://github.com/ClawSeven/sglang.git@dev-dllm#subdirectory=python'"
        )
        .set_env_vars({"HF_TOKEN": os.environ["HF_TOKEN"]})
    )
    inference_compute = kt.Compute(
        gpus=1, memory="150Gi", image=inference_img, launch_timeout=1200
    )

    # Training image (PyTorch + FSDP)
    train_img = (
        kt.Image(image_id="nvcr.io/nvidia/ai-workbench/python-cuda126:1.0.2")
        .run_bash(
            "uv pip install --system --break-system-packages "
            "torch torchvision torchaudio triton datasets transformers wandb "
            "diffusers tiktoken torchdata psutil timm einops safetensors pyyaml"
        )
        .pip_install(["bitsandbytes", "liger-kernel"])
        .set_env_vars({
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "HF_TOKEN": os.environ["HF_TOKEN"],
        })
    )

    num_workers = config.get("num_training_workers", 3)
    gpus_per_worker = config.get("num_gpus_per_worker", 1)
    train_compute = kt.Compute(
        gpus=gpus_per_worker,
        memory="150Gi",
        image=train_img,
        launch_timeout=1200,
        allowed_serialization=["json", "pickle"],
    ).distribute("pytorch", workers=num_workers)

    print("Deploying services...")

    async def deploy_training():
        service = await kt.cls(GRPOTrainer).to_async(
            train_compute, init_args={"config": config}, get_if_exists=get_if_exists
        )
        service.async_ = True
        await service.setup()
        return service

    # All in parallel: inference deploy + (training deploy + setup)
    inference_service, train_service = await asyncio.gather(
        kt.cls(SGLang).to_async(
            inference_compute,
            init_args={"model_id": config["model_path"], "checkpoint_version": 0, "config": config},
            get_if_exists=get_if_exists,
        ),
        deploy_training(),
    )
    inference_service.async_ = True

    return inference_service, train_service


async def main():
    """Main entry point."""
    config = load_config()
    print(f"Loaded config from config.yaml")

    # Load dataset
    dataset_split = config.get("train_split", "train[:100]")
    print(f"Loading GSM8K dataset: {dataset_split}")
    dataset = load_dataset(dataset_split)

    # Deploy services
    inference_service, train_service = await deploy_services(config, False)

    # Create agent and scheduler
    agent = SimpleMathAgent(inference_service, checkpoint_version=0)
    scheduler = AsyncGRPOScheduler(
        train_service=train_service,
        agent=agent,
        max_inference_parallel=2,
        max_training_pending=3,
        checkpoint_interval=config.get("checkpoint_interval", 10),
    )

    # Run training
    num_generations = config.get("num_generations", 4)
    batch_size = config.get("global_batch_size", 32) // num_generations

    await scheduler.run(
        dataset,
        num_epochs=config.get("num_epochs", 2),
        batch_size=batch_size,
        num_generations=num_generations,
    )


if __name__ == "__main__":
    asyncio.run(main())
