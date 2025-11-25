"""
Kubetorch launcher for DFactory LLaDA2.0 training.

Usage:
    python launch_dfactory.py
"""
import kubetorch as kt

# Hardcoded for convenience, can use argparse instead
CONFIG = {
    # NOTE: Using unmerged model - SGLang doesn't support merged MoE weights
    "model_path": "./inclusionAI/LLaDA2.0-mini-preview",
    "tokenizer_path": "./inclusionAI/LLaDA2.0-mini-preview",
    "train_data_path": "./gsm8k_datasets/gsm8k_train.jsonl",
    "max_seq_len": 1024,
    "noise_range": (0.3, 0.8),
    "mask_token_id": 156895,
    "output_dir": "./llada2_vanilla_outputs",
    "global_batch_size": 16,
    "micro_batch_size": 8,
    "num_epochs": 1,
    "lr": 1e-5,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "log_steps": 10,
    "save_steps": 500,
    "block_size": 32,
    "use_gradient_checkpointing": True,
    "attn_implementation": "sdpa",  # "flex_attention" if FLEX_ATTENTION_AVAILABLE else "sdpa",
}


def train_llada2_bd(CONFIG):
    """
    Main training execution that runs on each replica of remote compute.
    """

    # Setup data
    from build_gsm8k_dataset import main as data_main

    data_main()

    # Download model
    # NOTE: Merge disabled - SGLang doesn't support merged MoE weights
    # Using unmerged model directly
    from huggingface_hub import snapshot_download

    repo_id = "inclusionAI/LLaDA2.0-mini-preview"

    snapshot_download(
        repo_id=repo_id,
        local_dir=repo_id,
    )

    # # Merge conversion disabled - keeping unmerged format
    # import sys
    # import os
    # parent_dir = os.path.dirname(os.path.dirname(__file__))
    # if parent_dir not in sys.path:
    #     sys.path.insert(0, parent_dir)
    # from moe_converter import main as moe_conversion_main
    # local_merged_dir = "LLaDA2.0-mini-preview-moe-merge"
    # moe_conversion_main(repo_id, local_merged_dir, "merge")

    # Run training
    from trainer import LLaDA2Trainer

    trainer = LLaDA2Trainer(CONFIG)
    result = trainer.train()

    return result


def main():
    """
    Main launcher function that sets up Kubetorch compute and runs training.
    """
    print("Initializing DFactory Kubetorch Launcher...")

    img = (
        kt.Image(image_id="nvcr.io/nvidia/ai-workbench/python-cuda126:1.0.2")
        # .run_bash('uv venv --python 3.11')
        # .run_bash("uv pip install --system --break-system-packages --python-version-constraint ignore --no-deps git+https://github.com/ByteDance-Seed/VeOmni.git#egg=veomni[gpu]")
        .run_bash(
            "uv pip install --system --break-system-packages torch torchvision torchaudio datasets transformers wandb diffusers tiktoken torchdata psutil timm einops"
        ).pip_install(["bitsandbytes", "liger-kernel"])
        # .run_bash('uv pip install --system --break-system-packages "git+https://github.com/ByteDance-Seed/VeOmni.git#egg=veomni[gpu]"')
        # uv pip install ".[gpu]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
        # .run_bash('uv sync --python 3.11 --extra gpu')
        # uv pip install git+https://github.com/ByteDance-Seed/VeOmni.git#egg=veomni[gpu] --no-deps
        # uv pip install torch torchvision torchaudio datasets transformers wandb diffusers tiktoken torchdata psutil timm einops
        # .run_bash('ln -s tasks/dataset dataset')
        #
        # .pip_install(["wandb", "omegaconf", "datasets", "transformers"])
        .set_env_vars(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            }
        )
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

    trainer = kt.fn(train_llada2_bd).to(compute)

    try:
        result = trainer(CONFIG)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        return result

    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()
