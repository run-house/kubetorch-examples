"""SFT Trainer for LLaDA2 Diffusion Language Model.

Supervised fine-tuning trainer with dual-sequence diffusion format.
"""
import os
from typing import Dict

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import trange

try:
    from torch.nn.attention.flex_attention import flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

from llada2.data import collate_fn, LLaDA2Dataset
from llada2.trainers.base import BaseTrainer
from llada2.utils import get_cosine_schedule_with_warmup


class SFTTrainer(BaseTrainer):
    """SFT Trainer for LLaDA2 diffusion model.

    Extends BaseTrainer with SFT-specific:
    - DataLoader with DistributedSampler
    - Epoch-based training loop
    - Gradient accumulation
    """

    def __init__(self, config: Dict):
        """Initialize SFT trainer.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # SFT-specific state
        self.dataloader = None
        self.sampler = None
        self.gradient_accumulation_steps = 1
        self.steps_per_epoch = 0
        self.global_step = 0

        # Run full setup immediately (SFT is not async)
        self._run_setup()

    def _run_setup(self):
        """Run full setup including data loading."""
        self.setup()
        self.setup_data()
        self._setup_scheduler()

    def setup_data(self):
        """Setup DataLoader with DistributedSampler."""
        if self.global_rank == 0:
            print("Loading dataset...")

        dataset = LLaDA2Dataset(
            self.config["train_data_path"],
            self.tokenizer,
            self.config["max_seq_len"],
            self.config["noise_range"],
            self.config["mask_token_id"],
        )

        self.sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True,
        )

        self.gradient_accumulation_steps = max(
            1,
            self.config["global_batch_size"]
            // (self.config["micro_batch_size"] * self.world_size),
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config["micro_batch_size"],
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        self.steps_per_epoch = len(self.dataloader) // self.gradient_accumulation_steps

    def _setup_scheduler(self):
        """Setup learning rate scheduler after data is loaded."""
        total_steps = self.steps_per_epoch * self.config["num_epochs"]
        warmup_steps = int(total_steps * self.config["warmup_ratio"])

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

    def train_step(self, batch) -> float:
        """Execute single training step.

        Args:
            batch: Batch dictionary with noisy_input_ids, input_ids, labels

        Returns:
            Loss value
        """
        noisy_input_ids = batch["noisy_input_ids"].to(self.device)
        clean_input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        batch_size = noisy_input_ids.shape[0]
        seq_len = noisy_input_ids.shape[1]

        # Build dual-sequence input
        full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
        position_ids = (
            torch.cat([
                torch.arange(seq_len, device=self.device),
                torch.arange(seq_len, device=self.device),
            ])
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Get attention mask
        if FLEX_ATTENTION_AVAILABLE and self.config["attn_implementation"] == "flex_attention":
            batch_attn_mask = self.attn_mask
        else:
            batch_attn_mask = self.attn_mask.expand(batch_size, -1, -1, -1)

        # Forward pass
        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=batch_attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        # Compute loss on noisy sequence positions
        logits = outputs.logits[:, :seq_len, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return loss

    def train(self):
        """Run the full training loop."""
        self.model.train()

        for epoch in range(self.config["num_epochs"]):
            self.sampler.set_epoch(epoch)

            if self.global_rank == 0:
                pbar = trange(self.steps_per_epoch, desc=f"Epoch {epoch + 1}")
            else:
                pbar = range(self.steps_per_epoch)

            data_iter = iter(self.dataloader)
            self.optimizer.zero_grad()

            for step in pbar:
                total_loss = 0.0

                for _ in range(self.gradient_accumulation_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break

                    loss = self.train_step(batch)
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()
                    total_loss += loss.item()

                self.model.clip_grad_norm_(self.config["max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                if self.global_rank == 0 and self.global_step % self.config["log_steps"] == 0:
                    avg_loss = total_loss / self.gradient_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

                if self.global_step % self.config["save_steps"] == 0:
                    self.save_checkpoint(self.global_step)

        self._save_final_checkpoint()
        self._cleanup()

    def save_checkpoint(self, step: int):
        """Save checkpoint at given step."""
        if self.global_rank == 0:
            print(f"\nSaving checkpoint at step {step}...")

        save_path = os.path.join(self.config["output_dir"], f"checkpoint-{step}")

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
            if self.global_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save(state_dict, os.path.join(save_path, "model.pt"))

        dist.barrier()

    def _save_final_checkpoint(self):
        """Save final checkpoint after training."""
        if self.global_rank == 0:
            print("\nSaving final checkpoint...")

        save_path = os.path.join(self.config["output_dir"], "final")

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
            if self.global_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save(state_dict, os.path.join(save_path, "model.pt"))

    def _cleanup(self):
        """Cleanup distributed process group."""
        dist.destroy_process_group()
        if self.global_rank == 0:
            print("Training complete!")


# Backward compatibility alias
LLaDA2Trainer = SFTTrainer
