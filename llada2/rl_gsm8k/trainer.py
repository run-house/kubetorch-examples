"""GRPO Trainer for LLaDA2 Diffusion Language Model.

This trainer adapts Group Relative Policy Optimization (GRPO) for LLaDA2's
diffusion-based language model architecture. It combines:
- GRPO's group-relative advantage computation
- LLaDA2's dual-sequence diffusion format
- FSDP for memory-efficient distributed training
- SGLang integration for fast weight hot-swapping
"""

import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import kubetorch as kt 
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer

try:
    import bitsandbytes as bnb

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    from torch.nn.attention.flex_attention import (  # noqa
        create_block_mask,
        flex_attention,
    )

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

# Import LLaDA2 components
from llada2.llada2_moe_vanilla import LLaDA2Config, LLaDA2DecoderLayer, LLaDA2ForCausalLM
from llada2.data import sft_noise_transition
from llada2.utils import create_block_diffusion_mask


class GRPOTrainer:
    """GRPO Trainer for LLaDA2 diffusion model.

    Adapts GRPO (Group Relative Policy Optimization) for diffusion language models:
    - Samples K denoising trajectories per prompt
    - Computes group-relative advantages
    - Updates model to increase likelihood of better trajectories
    """

    def __init__(self, config_path: str = None, config: Dict = None, kt_cached_state=None):
        """Initialize trainer with config.

        Args:
            config_path: Path to YAML config file
            config: Config dictionary (takes precedence over config_path)
        """

        if kt_cached_state:
            print(
                f"Reusing existing trainer from cached state"
            )
            # Restore all attributes from cached state
            self.config = kt_cached_state["config"]
            self.model = kt_cached_state["model"]
            self.tokenizer = kt_cached_state["tokenizer"]
            self.optimizer = kt_cached_state["optimizer"]
            self.scheduler = kt_cached_state["scheduler"]
            self.attn_mask = kt_cached_state["attn_mask"]
            self.device = kt_cached_state["device"]
            self.steps = kt_cached_state["steps"]
            self.checkpoint_version = kt_cached_state["checkpoint_version"]
            self.global_rank = kt_cached_state["global_rank"]
            self.local_rank = kt_cached_state["local_rank"]
            self.world_size = kt_cached_state["world_size"]

            return
            

        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config
            config_file = os.path.join(_current_dir, "config.yaml")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError("Must provide either config_path or config")

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.attn_mask = None
        self.device = None
        self.steps = 0
        self.checkpoint_version = 0

        # Distributed setup will be done in setup()
        self.global_rank = None
        self.local_rank = None
        self.world_size = None

    def setup(self, skip = False):
        """Initialize model, optimizer, and distributed training."""
        if not skip: 
            self.setup_distributed()

            if self.global_rank == 0:
                print(f"Setting up LLaDA2 GRPO Trainer with {self.world_size} GPUs")
                os.makedirs(self.config.get("output_dir", "./checkpoints"), exist_ok=True)

            self.setup_tokenizer()
            self.setup_model()
            self.setup_optimizer()
            self.setup_attention_mask()

            if self.global_rank == 0:
                print("GRPO Trainer setup complete")

    def __kt_cached_state__(self) -> Dict[str, Any]:
        """Return state to be cached by Kubetorch across reloads."""
        # Preserve model, optimizer, and training state across reloads
        return {
            "config": self.config,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "attn_mask": self.attn_mask,
            "device": self.device,
            "steps": self.steps,
            "checkpoint_version": self.checkpoint_version,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
        }

    def setup_distributed(self):
        """Setup distributed training with NCCL backend and FSDP process groups."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        # Barrier to ensure all ranks have initialized before creating custom groups
        # dist.new_group() is a collective operation - all ranks must participate together
        dist.barrier()

        # Setup FSDP sharding groups
        # fsdp_sharding_group_size: number of GPUs that share one model replica
        # Example: world_size=8, fsdp_sharding_group_size=2 -> 4 model replicas, each sharded across 2 GPUs
        self.fsdp_sharding_group_size = self.config.get(
            "fsdp_sharding_group_size", self.world_size
        )

        if self.fsdp_sharding_group_size > 1 and self.world_size > 1:
            # Create FSDP process groups for sharding
            assert (
                self.world_size % self.fsdp_sharding_group_size == 0
            ), f"world_size ({self.world_size}) must be divisible by fsdp_sharding_group_size ({self.fsdp_sharding_group_size})"

            self.num_model_replicas = self.world_size // self.fsdp_sharding_group_size
            self.replica_id = self.global_rank // self.fsdp_sharding_group_size
            self.rank_in_replica = self.global_rank % self.fsdp_sharding_group_size

            # Create process groups - each replica has its own group
            self.fsdp_process_groups = []
            for i in range(self.num_model_replicas):
                replica_ranks = list(
                    range(
                        i * self.fsdp_sharding_group_size,
                        (i + 1) * self.fsdp_sharding_group_size,
                    )
                )
                group = dist.new_group(ranks=replica_ranks)
                self.fsdp_process_groups.append(group)

            self.fsdp_process_group = self.fsdp_process_groups[self.replica_id]

            if self.global_rank == 0:
                print("Initialized FSDP training:")
                print(f"  World size: {self.world_size} GPUs")
                print(
                    f"  FSDP sharding group size: {self.fsdp_sharding_group_size} GPUs/replica"
                )
                print(f"  Number of model replicas: {self.num_model_replicas}")
                print(f"  Data parallelism across {self.num_model_replicas} replicas")
        else:
            # No FSDP sharding - each GPU has full model
            self.num_model_replicas = self.world_size if self.world_size > 0 else 1
            self.replica_id = self.global_rank
            self.rank_in_replica = 0
            self.fsdp_process_group = None

            if self.global_rank == 0:
                print(
                    f"Distributed setup: rank={self.global_rank}, world_size={self.world_size} (no FSDP sharding)"
                )

    def setup_tokenizer(self):
        """Setup tokenizer."""
        if self.global_rank == 0:
            print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _ensure_model_downloaded(self, model_path: str):
        """Download model if needed.

        All ranks try to download if path doesn't exist. Filesystem handles
        the race condition - first rank creates the directory, others wait.
        """
        from pathlib import Path

        # Path("inclusionAI/LLaDA2.0-mini-preview").mkdir(parents=True, exist_ok=True)

        # kt.get(key="py-sglang/inclusionAI/LLaDA2.0-mini-preview/LLaDA2.0-mini-preview/", seed_data=False, dest="inclusionAI/LLaDA2.0-mini-preview", contents=True, force =True)


        if not os.path.exists(model_path):
            print(
                f"[RANK {self.global_rank}] Model not found at {model_path}, downloading from HuggingFace..."
            )
            from huggingface_hub import snapshot_download

            try:
                snapshot_download(repo_id=model_path, local_dir=model_path)
                print(f"[RANK {self.global_rank}] Model downloaded successfully")
            except Exception as e:
                print(f"[RANK {self.global_rank}] Download attempt: {e}")

        # Wait for all ranks to reach this point
        dist.barrier()
        print(f"[RANK {self.global_rank}] Model ready at {model_path}")

    def setup_model(self):
        """Setup LLaDA2 model with FSDP."""
        if self.global_rank == 0:
            print("Loading LLaDA2 model...")

        model_path = self.config["model_path"]

        # Download model if needed
        self._ensure_model_downloaded(model_path)

        # Load model config
        model_config = LLaDA2Config.from_pretrained(model_path)
        model_config._attn_implementation = self.config.get(
            "attn_implementation", "sdpa"
        )

        # Create model on meta device
        with torch.device("meta"):
            model = LLaDA2ForCausalLM(model_config)

        # FSDP auto wrap policy
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LLaDA2DecoderLayer},
        )

        # Mixed precision policy
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Wrap with FSDP
        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=self.local_rank,
            sync_module_states=True,
            param_init_fn=lambda module: module.to_empty(
                device=self.device, recurse=False
            ),
            process_group=self.fsdp_process_group,
        )

        # Load pretrained weights
        self.load_model_weights(model_path)

        # Enable gradient checkpointing if configured
        if self.config.get("use_gradient_checkpointing", True):
            if self.global_rank == 0:
                print("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()

    def load_model_weights(self, model_path: str):
        """Load pretrained weights into FSDP-wrapped model.

        Args:
            model_path: Path to model checkpoint
        """
        if self.global_rank == 0:
            print("Loading model weights...")

        from glob import glob

        from safetensors.torch import safe_open

        index_file = os.path.join(model_path, "model.safetensors.index.json")
        single_file = os.path.join(model_path, "model.safetensors")

        # All ranks load state_dict from disk
        state_dict = {}
        if os.path.exists(index_file):
            safetensor_files = sorted(
                glob(os.path.join(model_path, "model-*.safetensors"))
            )
            for shard_file in safetensor_files:
                with safe_open(shard_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        elif os.path.exists(single_file):
            with safe_open(single_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            if self.global_rank == 0:
                print(f"Warning: No safetensors found in {model_path}")
            return

        # All ranks call load_state_dict
        self.model.load_state_dict(state_dict, strict=False)
        del state_dict

        if self.global_rank == 0:
            print("Model weights loaded successfully")

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        if HAS_BITSANDBYTES:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-6),
                weight_decay=self.config.get("weight_decay", 0.01),
                betas=(0.9, 0.999),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-6),
                weight_decay=self.config.get("weight_decay", 0.01),
                betas=(0.9, 0.999),
                fused=True,
            )

        # Note: scheduler will be setup with total_steps once known
        # For now, we'll update it per-batch
        if self.global_rank == 0:
            print(f"Optimizer setup with lr={self.config.get('learning_rate', 1e-6)}")

    def setup_attention_mask(self):
        """Setup block diffusion attention mask."""
        self.attn_mask = create_block_diffusion_mask(
            self.config["max_seq_len"],
            self.config["block_size"],
            self.device,
        )

        if self.global_rank == 0:
            print(
                f"Attention mask created: max_seq_len={self.config['max_seq_len']}, "
                f"block_size={self.config['block_size']}"
            )

    def train_batch(
        self,
        prompts: List[str],
        completions: List[str],
        token_ids: List[List[int]],
        rewards: List[float],
        num_generations: int,
    ) -> Dict[str, Any]:
        """Train on a batch using GRPO with diffusion format.

        Args:
            prompts: List of prompt strings (expanded: each prompt repeated K times)
            completions: List of completion strings
            token_ids: List of token ID lists for completions
            rewards: List of rewards for each completion
            num_generations: Number of generations per prompt (K)

        Returns:
            Dictionary of metrics
        """
        start_time = time.time()
        timings = {}

        # Debug: Check inputs before distribution
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(
            f"[WORKER {rank}] train_batch called with {len(prompts)} prompts, {len(completions)} completions, {len(token_ids)} token_ids"
        )

        # Debug: Show token_ids lengths and rewards
        token_lengths = [len(ids) for ids in token_ids]
        print(
            f"[WORKER {rank}] Token lengths: min={min(token_lengths) if token_lengths else 0}, max={max(token_lengths) if token_lengths else 0}, mean={sum(token_lengths)/len(token_lengths) if token_lengths else 0:.1f}"
        )
        print(f"[WORKER {rank}] Rewards: {rewards}")
        print(
            f"[WORKER {rank}] Sample completion texts (first 3): {completions[:3] if len(completions) >= 3 else completions}"
        )

        if not self.model:
            self.setup()

        self.model.train()
        self.optimizer.zero_grad()

        # Split batch across MODEL REPLICAS at PROMPT level for GRPO correctness
        # With FSDP sharding, multiple GPUs share one model replica, so we split by replica not by GPU
        if self.num_model_replicas > 1:
            replica_id = self.replica_id
            num_replicas = self.num_model_replicas

            total_samples = len(prompts)
            assert (
                total_samples % num_generations == 0
            ), f"Total samples {total_samples} must be divisible by num_generations {num_generations}"

            total_prompts = total_samples // num_generations
            prompts_per_replica = total_prompts // num_replicas
            remainder = total_prompts % num_replicas

            # Handle uneven splits
            if replica_id < remainder:
                start_prompt = replica_id * (prompts_per_replica + 1)
                end_prompt = start_prompt + prompts_per_replica + 1
            else:
                start_prompt = (
                    remainder * (prompts_per_replica + 1)
                    + (replica_id - remainder) * prompts_per_replica
                )
                end_prompt = start_prompt + prompts_per_replica

            # Convert prompt indices to sample indices
            start_idx = start_prompt * num_generations
            end_idx = end_prompt * num_generations

            # Split data for this replica (all GPUs in replica get same data)
            prompts = prompts[start_idx:end_idx]
            completions = completions[start_idx:end_idx]
            token_ids = token_ids[start_idx:end_idx]
            rewards = rewards[start_idx:end_idx]

            rank = self.global_rank
        else:
            rank = self.global_rank

        num_samples = len(prompts)
        # Get micro_batch_size from config
        micro_batch_size = self.config.get("micro_batch_size", num_samples)
        print(
            f"[WORKER {rank}] Processing {num_samples} samples with micro_batch_size={micro_batch_size}"
        )

        # Tokenize prompts
        tokenize_start = time.time()
        prompt_encoding = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config["max_seq_len"] // 2,  # Half because dual-sequence
            return_tensors="pt",
        )
        prompt_ids = prompt_encoding.input_ids.to(self.device)
        timings["tokenize"] = time.time() - tokenize_start

        # Pad completions
        pad_start = time.time()
        max_len = min(
            max(len(ids) for ids in token_ids), self.config["max_seq_len"] // 2
        )
        padded_completion_ids = []
        completion_masks = []

        pad_id = self.tokenizer.pad_token_id
        for ids in token_ids:
            padded = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))
            mask = [1.0] * min(len(ids), max_len) + [0.0] * max(0, max_len - len(ids))
            padded_completion_ids.append(padded)
            completion_masks.append(mask)

        completion_ids = torch.tensor(padded_completion_ids, dtype=torch.long).to(
            self.device
        )
        completion_mask = torch.tensor(completion_masks, dtype=torch.float).to(
            self.device
        )
        timings["data_prep"] = time.time() - pad_start

        # Debug: Check if we have valid completion data
        print(
            f"[WORKER {rank}] max_len={max_len}, completion_mask.sum()={completion_mask.sum().item():.1f}"
        )
        print(
            f"[WORKER {rank}] completion_ids shape: {completion_ids.shape}, first sample: {completion_ids[0][:10].tolist()}"
        )

        # Calculate GRPO advantages (group-relative)
        adv_start = time.time()
        rewards_tensor = torch.tensor(rewards).view(-1, num_generations)
        advantages = (rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)) / (
            rewards_tensor.std(dim=1, keepdim=True) + 1e-8
        )
        advantages = advantages.view(-1).to(self.device)
        print(
            f"[WORKER {rank}] Advantages: min={advantages.min().item():.3f}, max={advantages.max().item():.3f}, mean={advantages.mean().item():.3f}"
        )
        print(
            f"[WORKER {rank}] Advantages per sample: {[f'{a:.3f}' for a in advantages.cpu().tolist()]}"
        )
        timings["advantages"] = time.time() - adv_start

        # Apply noise transition (diffusion-specific)
        noise_start = time.time()
        noise_range = tuple(self.config["noise_range"])
        mask_token_id = self.config["mask_token_id"]

        # Apply noise to both prompts and completions
        batch_size = prompt_ids.shape[0]
        noisy_prompt_ids = torch.zeros_like(prompt_ids)
        noisy_completion_ids = torch.zeros_like(completion_ids)

        for i in range(batch_size):
            noisy_prompt_ids[i] = sft_noise_transition(
                prompt_ids[i].unsqueeze(0),
                None,  # labels not needed for prompts
                noise_range,
                mask_token_id,
            )[0].squeeze(0)

            noisy_completion_ids[i] = sft_noise_transition(
                completion_ids[i].unsqueeze(0), None, noise_range, mask_token_id
            )[0].squeeze(0)

        timings["noise_transition"] = time.time() - noise_start

        # Build dual-sequence diffusion format: [noisy_prompt|clean_prompt|noisy_completion|clean_completion]
        format_start = time.time()
        full_input_ids = torch.cat(
            [noisy_prompt_ids, prompt_ids, noisy_completion_ids, completion_ids], dim=1
        )

        # Create repeated position IDs
        prompt_len = prompt_ids.shape[1]
        completion_len = completion_ids.shape[1]
        seq_len = prompt_len + completion_len

        position_ids = (
            torch.cat(
                [
                    torch.arange(seq_len, device=self.device),
                    torch.arange(seq_len, device=self.device),
                ]
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Prepare attention mask - slice to actual sequence length
        actual_dual_seq_len = seq_len * 2  # Dual sequence: noisy + clean
        if (
            FLEX_ATTENTION_AVAILABLE
            and self.config.get("attn_implementation") == "flex_attention"
        ):
            batch_attn_mask = self.attn_mask[
                :, :, :actual_dual_seq_len, :actual_dual_seq_len
            ]
        else:
            # Slice the pre-created mask to actual sequence length and expand for batch
            batch_attn_mask = self.attn_mask[
                :, :, :actual_dual_seq_len, :actual_dual_seq_len
            ].expand(batch_size, -1, -1, -1)

        timings["diffusion_format"] = time.time() - format_start

        # Micro-batching loop to reduce memory usage
        total_loss = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        for micro_batch_idx, micro_idx in enumerate(
            range(0, batch_size, micro_batch_size)
        ):
            micro_end = min(micro_idx + micro_batch_size, batch_size)

            print(
                f"[WORKER {rank}] Processing micro-batch {micro_batch_idx + 1}/{num_micro_batches} (samples {micro_idx}-{micro_end-1})"
            )

            # Slice batch for this micro-batch
            micro_input_ids = full_input_ids[micro_idx:micro_end]
            micro_completion_ids = completion_ids[micro_idx:micro_end]
            micro_advantages = advantages[micro_idx:micro_end]
            micro_completion_mask = completion_mask[micro_idx:micro_end]
            micro_position_ids = position_ids[micro_idx:micro_end]

            # Slice attention mask for micro-batch
            if (
                FLEX_ATTENTION_AVAILABLE
                and self.config.get("attn_implementation") == "flex_attention"
            ):
                micro_attn_mask = batch_attn_mask  # Shared for flex attention
            else:
                micro_attn_mask = batch_attn_mask[micro_idx:micro_end]

            # Forward pass
            forward_start = time.time()
            outputs = self.model(
                input_ids=micro_input_ids,
                attention_mask=micro_attn_mask,
                position_ids=micro_position_ids,
                use_cache=False,
            )

            # Extract logits from noisy completion positions
            # Input structure: [noisy_prompt | clean_prompt | noisy_completion | clean_completion]
            # We want logits from noisy_completion to predict clean_completion
            noisy_completion_start = 2 * prompt_len
            noisy_completion_end = noisy_completion_start + completion_len
            logits = outputs.logits[:, noisy_completion_start:noisy_completion_end, :]
            total_forward_time += time.time() - forward_start

            # Compute DrGRPO loss for this micro-batch
            vocab_size = logits.size(-1)
            flat_logits = logits.reshape(-1, vocab_size)
            flat_targets = micro_completion_ids.reshape(-1)

            token_losses = F.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            ).reshape(micro_completion_ids.shape)

            # Weight by advantages (DrGRPO)
            masked_token_losses = (
                token_losses * micro_advantages.unsqueeze(-1) * micro_completion_mask
            )
            micro_mask_sum = micro_completion_mask.sum()

            if micro_batch_idx == 0 and rank == 0:  # Debug first micro-batch only
                print(
                    f"[WORKER {rank}] Micro-batch {micro_batch_idx}: token_losses mean={token_losses.mean().item():.4f}, masked_sum={masked_token_losses.sum().item():.4f}, mask_sum={micro_mask_sum.item():.1f}"
                )

            micro_weighted_loss = masked_token_losses.sum() / (micro_mask_sum + 1e-8)

            # Scale loss for gradient accumulation across micro-batches
            scaled_loss = micro_weighted_loss / (
                (batch_size + micro_batch_size - 1) // micro_batch_size
            )
            total_loss += micro_weighted_loss.item()

            # Backward (accumulates gradients)
            backward_start = time.time()
            scaled_loss.backward()
            backward_time = time.time() - backward_start
            total_backward_time += backward_time

            print(
                f"[WORKER {rank}] Completed micro-batch {micro_batch_idx + 1}/{num_micro_batches}: loss={micro_weighted_loss.item():.4f}, forward={time.time() - forward_start:.2f}s, backward={backward_time:.2f}s"
            )

        # Clip gradients and update after all micro-batches
        print(
            f"[WORKER {rank}] All micro-batches complete, clipping gradients and updating weights..."
        )
        optimizer_start = time.time()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.get("max_grad_norm", 1.0)
        )
        self.optimizer.step()
        optimizer_time = time.time() - optimizer_start

        timings["forward"] = total_forward_time
        timings["backward"] = total_backward_time
        timings["optimizer"] = optimizer_time

        # Calculate average loss across micro-batches
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size
        avg_loss = total_loss / num_micro_batches
        timings["loss_compute"] = avg_loss

        self.steps += 1
        timings["total"] = time.time() - start_time

        print(
            f"[WORKER {rank}] Training step complete: avg_loss={avg_loss:.4f}, total_time={timings['total']:.2f}s (forward={total_forward_time:.2f}s, backward={total_backward_time:.2f}s, optimizer={optimizer_time:.2f}s)"
        )

        # Collect metrics
        rewards_tensor = torch.tensor(rewards)
        response_lengths = [len(ids) for ids in token_ids]

        local_metrics = {
            "loss": avg_loss,
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "reward_max": rewards_tensor.max().item(),
            "reward_min": rewards_tensor.min().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "response_length_mean": np.mean(response_lengths),
            "num_samples": len(rewards),
        }

        # Aggregate metrics across workers
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            aggregated_metrics = {}
            for metric_name, metric_value in local_metrics.items():
                metric_tensor = torch.tensor(metric_value, device=self.device)

                if metric_name in ["reward_max"]:
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.MAX)
                elif metric_name in ["reward_min"]:
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.MIN)
                elif metric_name in ["num_samples"]:
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                else:
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                    metric_tensor /= world_size

                aggregated_metrics[metric_name] = metric_tensor.item()

            # Barrier to ensure all workers finish training before returning
            # Without this, rank 0 returns early while other ranks are still in backward/optimizer
            dist.barrier()

            if rank == 0:
                return {
                    "metrics": aggregated_metrics,
                    "timings": timings,
                    "step": self.steps,
                }
            else:
                return {"metrics": {}, "timings": {}, "step": self.steps}
        else:
            return {"metrics": local_metrics, "timings": timings, "step": self.steps}

    def save_checkpoint(self) -> Tuple[str, int, str]:
        """Save model checkpoint.

        With FSDP sharding, this gathers the full state dict on rank 0 only.

        Returns:
            Tuple of (key, checkpoint_version, checkpoint_subdir)
        """
        self.checkpoint_version += 1
        checkpoint_path = (
            Path(self.config.get("output_dir", "./checkpoints"))
            / f"checkpoint-v{self.checkpoint_version}-step{self.steps}"
        )

        if self.global_rank == 0:
            print(f"Saving checkpoint v{self.checkpoint_version} to {checkpoint_path}")
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model with FSDP - gather full state dict on rank 0
        # For FSDP with multiple replicas, we only save from the first replica (ranks 0 to sharding_group_size-1)
        # and within that replica, only rank 0 gets the full state dict
        save_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,  # Only rank 0 in each FSDP group gets full state dict
        )

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=save_policy,
        ):
            state_dict = self.model.state_dict()

            # Only global rank 0 saves (not replica rank 0)
            if self.global_rank == 0:
                torch.save(state_dict, checkpoint_path / "model.pt")
                print(
                    f"Checkpoint saved successfully (gathered from {self.fsdp_sharding_group_size} GPUs)"
                )

        dist.barrier()
        key = f"model_v{self.checkpoint_version}"
        if self.global_rank == 0:
            print(f'Putting {checkpoint_path} to {key}')
            kt.vput(key=key, src = str(checkpoint_path))

        return key, self.checkpoint_version, checkpoint_path.name


if __name__ == "__main__": 
    import os

    import yaml

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    train_img = (
        kt.Image(image_id="nvcr.io/nvidia/ai-workbench/python-cuda126:1.0.2")
        .run_bash(
            "uv pip install --system --break-system-packages torch torchvision torchaudio triton datasets transformers wandb diffusers tiktoken torchdata psutil timm einops safetensors pyyaml"
        )
        .pip_install(["bitsandbytes", "liger-kernel"])
        .set_env_vars(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "HF_TOKEN": os.environ["HF_TOKEN"]
            }
        )
    )

    num_workers = 3
    gpus_per_worker = 1
    train_compute = kt.Compute(
        gpus=gpus_per_worker,
        image=train_img,
        launch_timeout=1200,
        allowed_serialization=["json", "pickle"],
    ).distribute("pytorch", workers=num_workers)

    train_service = kt.cls(GRPOTrainer).to(
        train_compute, init_args={"config": config}, get_if_exists=True
    )
    # train_service.setup() 
    print(train_service.save_checkpoint())

