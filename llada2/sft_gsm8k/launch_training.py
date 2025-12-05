"""Kubetorch launcher for LLaDA2 SFT training.

Usage:
    python launch_training.py
"""
import os

import kubetorch as kt
import yaml


def load_config():
    """Load training config from YAML."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert noise_range list to tuple (required by trainer)
    if isinstance(config.get("noise_range"), list):
        config["noise_range"] = tuple(config["noise_range"])

    return config


def train_llada2_sft(config):
    """Main training execution on remote compute."""
    # Setup data
    from build_gsm8k_dataset import main as data_main
    data_main()

    # Download model
    from huggingface_hub import snapshot_download
    repo_id = config["model_path"].lstrip("./")
    snapshot_download(repo_id=repo_id, local_dir=config["model_path"])

    # Run training
    from llada2.sft_gsm8k.trainer import SFTTrainer
    trainer = SFTTrainer(config)
    return trainer.train()


def main():
    """Main launcher function."""
    print("Initializing LLaDA2 SFT Training...")

    config = load_config()

    img = (
        kt.Image(image_id="nvcr.io/nvidia/ai-workbench/python-cuda126:1.0.2")
        .run_bash(
            "uv pip install --system --break-system-packages "
            "torch torchvision torchaudio datasets transformers wandb "
            "diffusers tiktoken torchdata psutil timm einops safetensors pyyaml"
        )
        .pip_install(["bitsandbytes", "liger-kernel"])
        .set_env_vars({
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        })
    )

    num_workers = 4
    gpus_per_node = 1

    compute = kt.Compute(
        gpus=gpus_per_node,
        image=img,
        launch_timeout=1800,
        allowed_serialization=["pickle", "json"],
        secrets=["huggingface"],
    ).distribute("pytorch", workers=num_workers)

    trainer_fn = kt.fn(train_llada2_sft).to(compute)

    try:
        result = trainer_fn(config)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        return result
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()
