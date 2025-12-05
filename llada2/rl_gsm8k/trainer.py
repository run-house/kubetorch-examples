"""GRPO Trainer for LLaDA2 Diffusion Language Model.

Adapts Group Relative Policy Optimization (GRPO) for LLaDA2's
diffusion-based language model architecture.
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import kubetorch as kt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

try:
    from torch.nn.attention.flex_attention import flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

from llada2.data import sft_noise_transition
from llada2.trainers.base import BaseTrainer


class GRPOTrainer(BaseTrainer):
    """GRPO Trainer for LLaDA2 diffusion model.

    Extends BaseTrainer with GRPO-specific:
    - Group-relative advantage computation
    - Dual-sequence diffusion format for training
    - Checkpoint versioning for hot-swap with inference
    """

    def __init__(self, config_path: str = None, config: Dict = None, kt_cached_state=None):
        """Initialize GRPO trainer.

        Args:
            config_path: Path to YAML config file
            config: Config dictionary (takes precedence over config_path)
            kt_cached_state: Cached state from kubetorch
        """
        # Load config from file if provided
        if config is None and config_path is not None:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif config is None:
            config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
            if os.path.exists(config_file):
                import yaml
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError("Must provide either config_path or config")

        # Handle cached state restoration
        if kt_cached_state:
            super().__init__(config, kt_cached_state)
            self.checkpoint_version = kt_cached_state.get("checkpoint_version", 0)
            return

        super().__init__(config, kt_cached_state=None)
        self.checkpoint_version = 0

    def __kt_cached_state__(self) -> Dict[str, Any]:
        """Extend base cached state with GRPO-specific state."""
        state = super().__kt_cached_state__()
        state["checkpoint_version"] = self.checkpoint_version
        return state

    def train(self, *args, **kwargs):
        """Not used - GRPO uses train_batch() for async training."""
        raise NotImplementedError("GRPOTrainer uses train_batch() for async training")

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
            prompts: Prompt strings (expanded: each repeated K times)
            completions: Completion strings
            token_ids: Token ID lists for completions
            rewards: Rewards for each completion
            num_generations: Number of generations per prompt (K)

        Returns:
            Dictionary with metrics, timings, and step count
        """
        start_time = time.time()
        rank = dist.get_rank() if dist.is_initialized() else 0

        if not self.model:
            self.setup()

        self.model.train()
        self.optimizer.zero_grad()

        # Split batch across model replicas for data parallelism
        prompts, completions, token_ids, rewards = self._split_for_replicas(
            prompts, completions, token_ids, rewards, num_generations
        )

        num_samples = len(prompts)
        micro_batch_size = self.config.get("micro_batch_size", num_samples)

        # Tokenize prompts
        prompt_encoding = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config["max_seq_len"] // 2,
            return_tensors="pt",
        )
        prompt_ids = prompt_encoding.input_ids.to(self.device)

        # Pad completions
        completion_ids, completion_mask = self._pad_completions(token_ids)

        # Calculate GRPO advantages (group-relative)
        advantages = self._compute_advantages(rewards, num_generations)

        # Apply diffusion noise
        noisy_prompt_ids, noisy_completion_ids = self._apply_noise(prompt_ids, completion_ids)

        # Build dual-sequence format
        full_input_ids, position_ids, batch_attn_mask = self._build_dual_sequence(
            noisy_prompt_ids, prompt_ids, noisy_completion_ids, completion_ids
        )

        # Micro-batch training loop
        total_loss, timings = self._train_micro_batches(
            full_input_ids, completion_ids, advantages, completion_mask,
            position_ids, batch_attn_mask, prompt_ids.shape[1], micro_batch_size
        )

        self.steps += 1
        timings["total"] = time.time() - start_time

        # Collect and aggregate metrics
        return self._collect_metrics(rewards, token_ids, advantages, total_loss, timings)

    def _split_for_replicas(
        self, prompts, completions, token_ids, rewards, num_generations
    ):
        """Split batch across model replicas at prompt level for GRPO correctness."""
        if self.num_model_replicas <= 1:
            return prompts, completions, token_ids, rewards

        total_samples = len(prompts)
        total_prompts = total_samples // num_generations
        prompts_per_replica = total_prompts // self.num_model_replicas
        remainder = total_prompts % self.num_model_replicas

        if self.replica_id < remainder:
            start_prompt = self.replica_id * (prompts_per_replica + 1)
            end_prompt = start_prompt + prompts_per_replica + 1
        else:
            start_prompt = (
                remainder * (prompts_per_replica + 1)
                + (self.replica_id - remainder) * prompts_per_replica
            )
            end_prompt = start_prompt + prompts_per_replica

        start_idx = start_prompt * num_generations
        end_idx = end_prompt * num_generations

        return (
            prompts[start_idx:end_idx],
            completions[start_idx:end_idx],
            token_ids[start_idx:end_idx],
            rewards[start_idx:end_idx],
        )

    def _pad_completions(self, token_ids: List[List[int]]):
        """Pad completion token IDs to uniform length."""
        max_len = min(
            max(len(ids) for ids in token_ids), self.config["max_seq_len"] // 2
        )
        pad_id = self.tokenizer.pad_token_id

        padded_ids = []
        masks = []
        for ids in token_ids:
            padded = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))
            mask = [1.0] * min(len(ids), max_len) + [0.0] * max(0, max_len - len(ids))
            padded_ids.append(padded)
            masks.append(mask)

        completion_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        completion_mask = torch.tensor(masks, dtype=torch.float, device=self.device)
        return completion_ids, completion_mask

    def _compute_advantages(self, rewards: List[float], num_generations: int):
        """Compute GRPO group-relative advantages."""
        rewards_tensor = torch.tensor(rewards).view(-1, num_generations)
        advantages = (rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)) / (
            rewards_tensor.std(dim=1, keepdim=True) + 1e-8
        )
        return advantages.view(-1).to(self.device)

    def _apply_noise(self, prompt_ids, completion_ids):
        """Apply diffusion noise transition to prompts and completions."""
        noise_range = tuple(self.config["noise_range"])
        mask_token_id = self.config["mask_token_id"]
        batch_size = prompt_ids.shape[0]

        noisy_prompt_ids = torch.zeros_like(prompt_ids)
        noisy_completion_ids = torch.zeros_like(completion_ids)

        for i in range(batch_size):
            noisy_prompt_ids[i] = sft_noise_transition(
                prompt_ids[i].unsqueeze(0), None, noise_range, mask_token_id
            )[0].squeeze(0)
            noisy_completion_ids[i] = sft_noise_transition(
                completion_ids[i].unsqueeze(0), None, noise_range, mask_token_id
            )[0].squeeze(0)

        return noisy_prompt_ids, noisy_completion_ids

    def _build_dual_sequence(
        self, noisy_prompt_ids, prompt_ids, noisy_completion_ids, completion_ids
    ):
        """Build dual-sequence diffusion format: [noisy_prompt|clean_prompt|noisy_completion|clean_completion]."""
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        completion_len = completion_ids.shape[1]
        seq_len = prompt_len + completion_len

        full_input_ids = torch.cat(
            [noisy_prompt_ids, prompt_ids, noisy_completion_ids, completion_ids], dim=1
        )

        position_ids = (
            torch.cat([
                torch.arange(seq_len, device=self.device),
                torch.arange(seq_len, device=self.device),
            ])
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        actual_dual_seq_len = seq_len * 2
        if FLEX_ATTENTION_AVAILABLE and self.config.get("attn_implementation") == "flex_attention":
            batch_attn_mask = self.attn_mask[:, :, :actual_dual_seq_len, :actual_dual_seq_len]
        else:
            batch_attn_mask = self.attn_mask[
                :, :, :actual_dual_seq_len, :actual_dual_seq_len
            ].expand(batch_size, -1, -1, -1)

        return full_input_ids, position_ids, batch_attn_mask

    def _train_micro_batches(
        self, full_input_ids, completion_ids, advantages, completion_mask,
        position_ids, batch_attn_mask, prompt_len, micro_batch_size
    ):
        """Run micro-batch training loop with gradient accumulation."""
        batch_size = full_input_ids.shape[0]
        completion_len = completion_ids.shape[1]
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        total_loss = 0.0
        timings = {"forward": 0.0, "backward": 0.0}

        for micro_batch_idx, micro_idx in enumerate(range(0, batch_size, micro_batch_size)):
            micro_end = min(micro_idx + micro_batch_size, batch_size)
            print(f"[WORKER {self.global_rank}] Micro-batch {micro_batch_idx + 1}/{num_micro_batches}")

            micro_input_ids = full_input_ids[micro_idx:micro_end]
            micro_completion_ids = completion_ids[micro_idx:micro_end]
            micro_advantages = advantages[micro_idx:micro_end]
            micro_completion_mask = completion_mask[micro_idx:micro_end]
            micro_position_ids = position_ids[micro_idx:micro_end]

            if FLEX_ATTENTION_AVAILABLE and self.config.get("attn_implementation") == "flex_attention":
                micro_attn_mask = batch_attn_mask
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
            noisy_completion_start = 2 * prompt_len
            noisy_completion_end = noisy_completion_start + completion_len
            logits = outputs.logits[:, noisy_completion_start:noisy_completion_end, :]
            timings["forward"] += time.time() - forward_start

            # Compute DrGRPO loss
            vocab_size = logits.size(-1)
            flat_logits = logits.reshape(-1, vocab_size)
            flat_targets = micro_completion_ids.reshape(-1)

            token_losses = F.cross_entropy(
                flat_logits, flat_targets, reduction="none"
            ).reshape(micro_completion_ids.shape)

            # Optional: drop low-entropy tokens (model already confident)
            entropy_mask = micro_completion_mask
            if self.config.get("entropy_dropout", False):
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    # Entropy: H = -sum(p * log(p)), normalized by log(vocab_size)
                    log_probs = torch.log(probs + 1e-10)
                    entropy = -torch.sum(probs * log_probs, dim=-1) / torch.log(torch.tensor(vocab_size, dtype=torch.float, device=self.device))
                    # Keep tokens with entropy above threshold (uncertain tokens)
                    threshold = self.config.get("entropy_threshold", 0.3)
                    high_entropy_mask = (entropy > threshold).float()
                    entropy_mask = micro_completion_mask * high_entropy_mask
                    kept_ratio = entropy_mask.sum() / (micro_completion_mask.sum() + 1e-8)
                    entropy_mean = (entropy * micro_completion_mask).sum() / (micro_completion_mask.sum() + 1e-8)
                    print(f"[WORKER {self.global_rank}] Entropy: mean={entropy_mean.item():.4f}, threshold={threshold}, keeping {kept_ratio.item()*100:.1f}%")

            masked_token_losses = token_losses * micro_advantages.unsqueeze(-1) * entropy_mask
            micro_mask_sum = entropy_mask.sum()
            micro_weighted_loss = masked_token_losses.sum() / (micro_mask_sum + 1e-8)

            scaled_loss = micro_weighted_loss / num_micro_batches
            total_loss += micro_weighted_loss.item()

            # Backward
            backward_start = time.time()
            scaled_loss.backward()
            backward_time = time.time() - backward_start
            timings["backward"] += backward_time

            print(f"[WORKER {self.global_rank}] loss={micro_weighted_loss.item():.4f} fwd={timings['forward']:.2f}s bwd={backward_time:.2f}s")

        # Optimizer step
        optimizer_start = time.time()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.get("max_grad_norm", 1.0)
        )
        self.optimizer.step()
        timings["optimizer"] = time.time() - optimizer_start

        avg_loss = total_loss / num_micro_batches
        print(f"[WORKER {self.global_rank}] Step done: avg_loss={avg_loss:.4f} opt={timings['optimizer']:.2f}s")
        return avg_loss, timings

    def _collect_metrics(self, rewards, token_ids, advantages, loss, timings):
        """Collect and aggregate metrics across workers."""
        rewards_tensor = torch.tensor(rewards)
        response_lengths = [len(ids) for ids in token_ids]

        local_metrics = {
            "loss": loss,
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "reward_max": rewards_tensor.max().item(),
            "reward_min": rewards_tensor.min().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "response_length_mean": np.mean(response_lengths),
            "num_samples": len(rewards),
        }

        if not dist.is_initialized():
            return {"metrics": local_metrics, "timings": timings, "step": self.steps}

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        aggregated = {}
        for name, value in local_metrics.items():
            tensor = torch.tensor(value, device=self.device)
            if name == "reward_max":
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            elif name == "reward_min":
                dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            elif name == "num_samples":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= world_size
            aggregated[name] = tensor.item()

        dist.barrier()

        if rank == 0:
            return {"metrics": aggregated, "timings": timings, "step": self.steps}
        return {"metrics": {}, "timings": {}, "step": self.steps}

    def save_checkpoint(self) -> Tuple[str, int, str]:
        """Save model checkpoint with FSDP state gathering.

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

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            state_dict = self.model.state_dict()
            if self.global_rank == 0:
                torch.save(state_dict, checkpoint_path / "model.pt")
                print(f"Checkpoint saved (gathered from {self.fsdp_sharding_group_size} GPUs)")

        dist.barrier()

        key = f"model_v{self.checkpoint_version}"
        if self.global_rank == 0:
            kt.vput(key=key, src=str(checkpoint_path))

        return key, self.checkpoint_version, checkpoint_path.name
