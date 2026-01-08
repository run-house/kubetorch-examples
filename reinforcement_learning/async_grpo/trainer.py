import gc
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class GRPOTrainer:
    """GRPO trainer with LoRA for memory-efficient training."""

    def __init__(
        self,
        model_id,
        trainer_config=None,
    ):
        self.model_id = model_id
        trainer_config = trainer_config or {}
        self.learning_rate = trainer_config.get("learning_rate", 1e-5)
        self.lora_r = trainer_config.get("lora_r", 16)
        self.lora_alpha = trainer_config.get("lora_alpha", 32)
        self.lora_dropout = trainer_config.get("lora_dropout", 0.1)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = None
        self.steps = 0
        self.checkpoint_version = 0
        self.pad_token_id = None
        self.microbatch_size = 4

    def setup(self):
        """Initialize model with LoRA and memory optimizations."""
        from peft import get_peft_model, LoraConfig, TaskType
        from torch.nn.parallel import DistributedDataParallel as DDP
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=TARGET_MODULES,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        self.model = self.model.to(self.device)

        self.model = DDP(
            self.model, device_ids=[self.device], find_unused_parameters=True
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate, weight_decay=0.01
        )

        print(f"Trainer setup complete on {self.device} with LoRA training")

    def compute_token_level_loss(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        advantages: torch.Tensor,
    ):
        """Compute DrGRPO token-level loss."""
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, prompt_ids.size(1) - 1 : -1, :]

        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = completion_ids.reshape(-1)

        token_losses = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).reshape(completion_ids.shape)

        masked_losses = token_losses * completion_mask
        token_advantages = advantages.unsqueeze(-1).expand_as(masked_losses)

        weighted_token_loss = (
            masked_losses * token_advantages * completion_mask
        ).sum() / completion_mask.sum()

        sequence_loss = masked_losses.sum(dim=1).mean()

        return weighted_token_loss, sequence_loss

    def train_batch(
        self,
        prompts: List[str],
        completions: List[str],
        completion_ids: List[List[int]],
        rewards: List[float],
        num_generations: int,
    ) -> Dict:
        """Train on a batch using DrGRPO with gradient accumulation."""
        from torch.amp import autocast

        start_time = time.time()
        timings = {}

        if not self.model:
            self.setup()
        self.model.train()
        self.optimizer.zero_grad()

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            total_samples = len(prompts)
            assert total_samples % num_generations == 0
            total_prompts = total_samples // num_generations
            prompts_per_worker = total_prompts // world_size
            remainder = total_prompts % world_size

            if rank < remainder:
                start_prompt = rank * (prompts_per_worker + 1)
                end_prompt = start_prompt + prompts_per_worker + 1
            else:
                start_prompt = (
                    remainder * (prompts_per_worker + 1)
                    + (rank - remainder) * prompts_per_worker
                )
                end_prompt = start_prompt + prompts_per_worker

            start_idx = start_prompt * num_generations
            end_idx = end_prompt * num_generations

            prompts = prompts[start_idx:end_idx]
            completions = completions[start_idx:end_idx]
            completion_ids = completion_ids[start_idx:end_idx]
            rewards = rewards[start_idx:end_idx]

        tokenize_start = time.time()
        prompt_encoding = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        prompt_ids_tensor = prompt_encoding.input_ids.to(self.device)
        timings["tokenize"] = time.time() - tokenize_start

        adv_start = time.time()
        rewards_array = torch.tensor(rewards).view(-1, num_generations)
        mean_rewards = rewards_array.mean(dim=1, keepdim=True)
        std_rewards = rewards_array.std(dim=1, keepdim=True)
        advantages = (rewards_array - mean_rewards) / (std_rewards + 1e-8)
        advantages_tensor = advantages.view(-1).to(self.device)
        timings["advantages"] = time.time() - adv_start

        pad_start = time.time()
        max_len = min(max(len(ids) for ids in completion_ids), 512)
        padded_completion_ids = []
        completion_masks = []

        for ids in completion_ids:
            padded = ids[:max_len] + [self.pad_token_id] * max(0, max_len - len(ids))
            mask = [1.0] * min(len(ids), max_len) + [0.0] * max(0, max_len - len(ids))
            padded_completion_ids.append(padded)
            completion_masks.append(mask)

        completion_ids_tensor = torch.tensor(
            padded_completion_ids, dtype=torch.long
        ).to(self.device)
        completion_mask_tensor = torch.tensor(completion_masks, dtype=torch.float).to(
            self.device
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        timings["data_prep"] = time.time() - pad_start

        forward_start = time.time()
        batch_size = prompt_ids_tensor.size(0)
        total_loss = 0
        num_microbatches = max(1, batch_size // self.microbatch_size)

        for i in range(0, batch_size, self.microbatch_size):
            end_idx = min(i + self.microbatch_size, batch_size)

            micro_prompt_ids = prompt_ids_tensor[i:end_idx]
            micro_completion_ids = completion_ids_tensor[i:end_idx]
            micro_completion_mask = completion_mask_tensor[i:end_idx]
            micro_advantages = advantages_tensor[i:end_idx]

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                token_loss, seq_loss = self.compute_token_level_loss(
                    micro_prompt_ids,
                    micro_completion_ids,
                    micro_completion_mask,
                    micro_advantages,
                )
                loss = token_loss / num_microbatches

            loss.backward()
            total_loss += token_loss.item()
        timings["forward"] = time.time() - forward_start

        backward_start = time.time()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        timings["backward"] = time.time() - backward_start

        if self.steps % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        self.steps += 1
        timings["total"] = time.time() - start_time

        response_lengths = [len(ids) for ids in completion_ids]
        prompt_lengths = [len(p.split()) for p in prompts]
        prompt_tokens = prompt_ids_tensor.numel()
        completion_tokens = sum(response_lengths)

        local_metrics = {
            "loss": total_loss / num_microbatches,
            "token_loss": total_loss / num_microbatches,
            "seq_loss": seq_loss.item() if "seq_loss" in locals() else 0,
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "reward_max": rewards_tensor.max().item(),
            "reward_min": rewards_tensor.min().item(),
            "advantages_mean": advantages_tensor.mean().item(),
            "advantages_std": advantages_tensor.std().item(),
            "response_length_mean": np.mean(response_lengths),
            "prompt_length_mean": np.mean(prompt_lengths),
            "num_samples": len(rewards),
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # Define reduction operations per metric
            max_metrics = {"reward_max"}
            min_metrics = {"reward_min"}
            sum_metrics = {
                "num_samples",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
            }
            # All others use SUM then divide by world_size (average)

            aggregated_metrics = local_metrics.copy()
            for name in [
                "reward_mean",
                "reward_std",
                "reward_max",
                "reward_min",
                "advantages_mean",
                "advantages_std",
                "response_length_mean",
                "prompt_length_mean",
                "num_samples",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
            ]:
                t = torch.tensor(local_metrics[name]).to(self.device)
                if name in max_metrics:
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
                elif name in min_metrics:
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
                else:
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                    if name not in sum_metrics:
                        t /= world_size
                aggregated_metrics[name] = t.item()

            if rank == 0:
                return {
                    "metrics": aggregated_metrics,
                    "timings": timings,
                    "step": self.steps,
                }
            return {"metrics": {}, "timings": {}, "step": self.steps}

        return {"metrics": local_metrics, "timings": timings, "step": self.steps}

    def get_lora_state_dict(self):
        """Extract LoRA parameters as CUDA tensors for GPU-to-GPU transfer."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        return {
            name: param.data
            for name, param in model.named_parameters()
            if "lora_" in name.lower()
        }

    def get_peft_config(self):
        """Get PEFT config dict for TensorLoRARequest."""
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": TARGET_MODULES,
            "bias": "none",
        }

    def get_lora_metadata(self):
        """Get LoRA tensor metadata (names, shapes, dtypes) for polling setup."""
        if self.model is None:
            self.setup()
        state = self.get_lora_state_dict()
        return {
            name: {"shape": list(t.shape), "dtype": str(t.dtype)}
            for name, t in state.items()
        }

    def publish_lora_metadata(self):
        """Publish LoRA metadata as JSON file for inference workers to discover."""
        import json

        import kubetorch as kt

        metadata = self.get_lora_metadata()
        meta_path = "/tmp/lora_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        kt.put(key="lora/metadata", src=meta_path)
        print(f"[PUBLISH] Published LoRA metadata ({len(metadata)} tensors)")

    def publish_lora_weights(self, key: str):
        """Publish LoRA weights to data store."""
        import kubetorch as kt

        self.checkpoint_version += 1
        lora_state = self.get_lora_state_dict()
        self._published_lora_state = lora_state

        total_params = sum(t.numel() for t in lora_state.values())
        total_bytes = sum(t.numel() * t.element_size() for t in lora_state.values())
        print(
            f"[PUBLISH] {len(lora_state)} tensors, {total_params:,} params, {total_bytes/1e6:.2f} MB"
        )

        t0 = time.time()
        kt.put(key=key, src=lora_state, verbose=True)
        elapsed = time.time() - t0
        throughput = total_bytes / elapsed / 1e6
        print(f"[PUBLISH] kt.put() took {elapsed:.3f}s ({throughput:.1f} MB/s)")

        metadata = {
            name: {"shape": list(t.shape), "dtype": str(t.dtype)}
            for name, t in lora_state.items()
        }

        return key, self.checkpoint_version, metadata
