import json
import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
from typing import Generator

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


@dataclass
class StateDictIterator:
    filepath: str

    def __iter__(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        else:
            state_dict = torch.load(
                self.filepath, map_location="cpu", weights_only=True, mmap=True
            )
            for key in state_dict.keys():
                yield key, state_dict[key]


def moe_merge(state_dict: dict[str, torch.Tensor], config) -> dict[str, torch.Tensor]:
    new_state_dict: dict[str, torch.Tensor] = dict()
    processed_keys: set[str] = set()

    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    first_k_dense_replace = config.first_k_dense_replace

    print(f"Merging {num_layers} layers with {num_experts} experts each")
    proj_types = ["gate_proj", "up_proj", "down_proj"]

    for layer_id in range(first_k_dense_replace, num_layers):
        for proj_type in proj_types:
            expert_weights = []
            current_expert_keys = []

            for expert_id in range(num_experts):
                expert_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_type}.weight"
                assert expert_key in state_dict, f"Missing key: {expert_key}"
                expert_weights.append(state_dict[expert_key])
                current_expert_keys.append(expert_key)

            assert len(expert_weights) == num_experts
            merged_weight = torch.stack(expert_weights, dim=0)
            new_key = f"model.layers.{layer_id}.mlp.experts.{proj_type}"
            new_state_dict[new_key] = merged_weight
            processed_keys.update(current_expert_keys)

            for key in current_expert_keys:
                del state_dict[key]
            print(
                f"✓ Layer {layer_id}.{proj_type}: {expert_weights[0].shape} -> {merged_weight.shape}"
            )
            del expert_weights

    for key, tensor in state_dict.items():
        if key not in processed_keys:
            new_state_dict[key] = tensor

    return new_state_dict


def split_moe_experts(
    merged_state_dict: dict[str, torch.Tensor], config
) -> dict[str, torch.Tensor]:
    split_state_dict: dict[str, torch.Tensor] = dict()

    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        raise ValueError("Could not find 'num_experts' in config.")

    proj_types = ["gate_proj", "up_proj", "down_proj"]
    merged_key_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(" + "|".join(proj_types) + r")$"
    )

    for key, merged_tensor in merged_state_dict.items():
        match = merged_key_pattern.match(key)

        if match:
            layer_id = match.group(1)
            proj_type = match.group(2)

            if not (merged_tensor.dim() > 1 and merged_tensor.shape[0] == num_experts):
                raise ValueError(
                    f"Tensor '{key}' has unexpected shape {merged_tensor.shape}. "
                    f"First dimension should equal num_experts ({num_experts})."
                )

            for expert_id in range(num_experts):
                expert_tensor = merged_tensor[expert_id]
                original_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_type}.weight"
                split_state_dict[original_key] = expert_tensor
        else:
            split_state_dict[key] = merged_tensor

    return split_state_dict


def _get_dtype_size(dtype: torch.dtype) -> int:
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return dtype_sizes.get(dtype, 4)


def _get_shard_info(
    state_dict: dict[str, torch.Tensor],
    save_dtype: torch.dtype,
    shard_size: int,
) -> tuple[bool, int, dict[str, str]]:
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []

    for name, tensor in state_dict.items():
        dtype = save_dtype if save_dtype else tensor.dtype
        tensor_size = tensor.numel() * _get_dtype_size(dtype)

        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

    num_shards = len(shard_list)
    weight_map = {}

    if num_shards == 1:
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = "model.safetensors"
    else:
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            file_name = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


def save_model_weights_vanilla(
    output_path: str,
    state_dict: dict[str, torch.Tensor],
    config,
    tokenizer,
    input_path: str = None,
    save_dtype: torch.dtype = torch.bfloat16,
    shard_size: int = 5_000_000_000,
):
    os.makedirs(output_path, exist_ok=True)

    is_sharded, total_size, weight_map = _get_shard_info(
        state_dict, save_dtype, shard_size
    )

    current_shard = {}
    prev_file_name = None

    for name, tensor in tqdm(state_dict.items(), desc="Saving weights"):
        if save_dtype:
            tensor = tensor.to(dtype=save_dtype)

        if prev_file_name is not None and weight_map[name] != prev_file_name:
            save_file(current_shard, os.path.join(output_path, prev_file_name))
            current_shard = {}

        current_shard[name] = tensor.detach().cpu()
        prev_file_name = weight_map[name]

    if current_shard:
        save_file(current_shard, os.path.join(output_path, prev_file_name))

    if is_sharded:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)

    config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Copy generation_config.json if it exists (critical for diffusion models!)
    import shutil

    generation_config_src = os.path.join(input_path, "generation_config.json")
    if os.path.exists(generation_config_src):
        generation_config_dst = os.path.join(output_path, "generation_config.json")
        shutil.copy2(generation_config_src, generation_config_dst)
        print("Copied generation_config.json")

    print(f"Model saved to {output_path}")


def ensure_moe_merged(model_path: str, rank: int = 0) -> str:
    """Ensure MoE experts are merged for optimized inference/training.

    The merged format stacks expert weights into single tensors, enabling:
    - Faster Triton kernel execution for MoE operations
    - Better memory efficiency
    - Improved serving/training performance

    Each rank creates its own local copy (parallel merge for distributed training).

    Args:
        model_path: Path to model (HuggingFace ID or local path)
        rank: Process rank for logging

    Returns:
        Path to merged model (creates it if needed)
    """
    # Skip if already merged
    if "-moe-merge" in model_path:
        if rank == 0:
            print("Model path already contains '-moe-merge', skipping merge")
        return model_path

    # Generate merged model path
    base_name = os.path.basename(model_path.rstrip("/"))
    merged_path = f"{base_name}-moe-merge"

    # Check if merged model already exists on disk
    if os.path.exists(merged_path):
        config_file = os.path.join(merged_path, "config.json")
        model_file = os.path.join(merged_path, "model.safetensors")
        model_index = os.path.join(merged_path, "model.safetensors.index.json")

        if os.path.exists(config_file) and (
            os.path.exists(model_file) or os.path.exists(model_index)
        ):
            if rank == 0:
                print(f"Found existing merged model at {merged_path}, using it")
            return merged_path

    if not os.path.exists(model_path):
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_path, local_dir=model_path)
        except Exception as e:
            print(f"Warning: Download failed: {e}")

    try:
        main(model_path, merged_path, "merge")
    except Exception as e:
        print(f"Merge failed: {e}, using original path")
        return model_path

    return merged_path


def main(input_path: str, output_path: str, mode: str):
    config_file = os.path.join(output_path, "config.json")
    model_file = os.path.join(output_path, "model.safetensors")
    model_index = os.path.join(output_path, "model.safetensors.index.json")

    if os.path.exists(config_file) and (
        os.path.exists(model_file) or os.path.exists(model_index)
    ):
        print(f"Model already exists at {output_path}, skipping.")
        return

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading config from {input_path}")
    config = AutoConfig.from_pretrained(input_path, trust_remote_code=True)

    print(f"Loading tokenizer from {input_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        input_path, padding_side="right", trust_remote_code=True
    )

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))

    if not safetensor_files:
        pt_files = sorted(glob(os.path.join(input_path, "*.pt")))
        if pt_files:
            safetensor_files = pt_files
        else:
            raise FileNotFoundError(
                f"No .safetensors or .pt files found in {input_path}"
            )

    print(f"Found {len(safetensor_files)} weight files")

    state_dict = {}
    for shard_file in tqdm(safetensor_files, desc="Loading shards"):
        iterator = StateDictIterator(shard_file)
        for name, tensor in iterator:
            state_dict[name] = tensor.cpu()

    print(f"Loaded {len(state_dict)} tensors")

    if mode == "merge":
        print("Merging expert weights...")
        new_state_dict = moe_merge(state_dict, config)
    elif mode == "split":
        print("Splitting expert weights...")
        new_state_dict = split_moe_experts(state_dict, config)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'merge' or 'split'.")

    state_dict.clear()

    print(f"Converted to {len(new_state_dict)} tensors")
    save_model_weights_vanilla(
        output_path, new_state_dict, config, tokenizer, input_path=input_path
    )
    print("Conversion complete!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument(
        "-m", "--mode", type=str, default="merge", choices=["merge", "split"]
    )
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.mode)
