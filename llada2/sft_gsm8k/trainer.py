import os
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import trange
from transformers import AutoTokenizer

try:
    import bitsandbytes as bnb

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not available. Using standard optimizer.")

try:
    from torch.nn.attention.flex_attention import (  # noqa
        create_block_mask,
        flex_attention,
    )

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print(
        "Warning: FlexAttention not available. Install PyTorch 2.5+ for better performance."
    )

from llada2.data import collate_fn, LLaDA2Dataset
from llada2.llada2_moe_vanilla import LLaDA2Config, LLaDA2DecoderLayer, LLaDA2ForCausalLM
from llada2.utils import create_block_diffusion_mask, get_cosine_schedule_with_warmup


class LLaDA2Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_distributed()

        if self.global_rank == 0:
            print(f"Training with {self.world_size} GPUs")
            os.makedirs(self.config["output_dir"], exist_ok=True)

        self.setup_tokenizer()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_attention_mask()

        self.global_step = 0

    def setup_distributed(self):
        dist.init_process_group(backend="nccl")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_model(self):
        if self.global_rank == 0:
            print("Loading model...")

        model_config = LLaDA2Config.from_pretrained(self.config["model_path"])
        model_config._attn_implementation = self.config["attn_implementation"]

        with torch.device("meta"):
            model = LLaDA2ForCausalLM(model_config)

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LLaDA2DecoderLayer},
        )

        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

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
        )

        self.load_model_weights()

        if self.config["use_gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()

    def load_model_weights(self):
        if self.global_rank == 0:
            print("Loading model weights...")

        from glob import glob

        from safetensors.torch import safe_open

        model_path = self.config["model_path"]
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        single_file = os.path.join(model_path, "model.safetensors")

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
            raise FileNotFoundError(f"No model weights found in {model_path}")

        self.model.load_state_dict(state_dict, strict=False)
        del state_dict

    def setup_data(self):
        if self.global_rank == 0:
            print("Loading dataset...")

        dataset = LLaDA2Dataset(
            self.config["train_data_path"],
            self.tokenizer,
            self.config["max_seq_len"],
            self.config["noise_range"],
            self.config["mask_token_id"],
        )

        sampler = DistributedSampler(
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
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        self.sampler = sampler
        self.steps_per_epoch = len(self.dataloader) // self.gradient_accumulation_steps

    def setup_optimizer(self):
        total_steps = self.steps_per_epoch * self.config["num_epochs"]
        warmup_steps = int(total_steps * self.config["warmup_ratio"])

        if HAS_BITSANDBYTES:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                betas=(0.9, 0.999),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                betas=(0.9, 0.999),
                fused=True,
            )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

    def setup_attention_mask(self):
        self.attn_mask = create_block_diffusion_mask(
            self.config["max_seq_len"],
            self.config["block_size"],
            self.device,
        )

    def train_step(self, batch):
        noisy_input_ids = batch["noisy_input_ids"].to(self.device)
        clean_input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        batch_size = noisy_input_ids.shape[0]
        full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
        seq_len = noisy_input_ids.shape[1]
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

        if (
            FLEX_ATTENTION_AVAILABLE
            and self.config["attn_implementation"] == "flex_attention"
        ):
            batch_attn_mask = self.attn_mask
        else:
            batch_attn_mask = self.attn_mask.expand(batch_size, -1, -1, -1)

        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=batch_attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        logits = outputs.logits[:, :seq_len, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return loss

    def save_checkpoint(self, step):
        if self.global_rank == 0:
            print(f"\nSaving checkpoint at step {step}...")

        save_path = os.path.join(self.config["output_dir"], f"checkpoint-{step}")

        with FSDP.state_dict_type(
            self.model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT
        ):
            state_dict = self.model.state_dict()
            if self.global_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save(state_dict, os.path.join(save_path, "model.pt"))

        dist.barrier()

    def train(self):
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

                for accum_step in range(self.gradient_accumulation_steps):
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

                if (
                    self.global_rank == 0
                    and self.global_step % self.config["log_steps"] == 0
                ):
                    avg_loss = total_loss / self.gradient_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

                if self.global_step % self.config["save_steps"] == 0:
                    self.save_checkpoint(self.global_step)

        self.save_final_checkpoint()
        self.cleanup()

    def save_final_checkpoint(self):
        if self.global_rank == 0:
            print("\nSaving final checkpoint...")

        save_path = os.path.join(self.config["output_dir"], "final")
        with FSDP.state_dict_type(
            self.model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT
        ):
            state_dict = self.model.state_dict()
            if self.global_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save(state_dict, os.path.join(save_path, "model.pt"))

    def cleanup(self):
        dist.destroy_process_group()
        if self.global_rank == 0:
            print("Training complete!")


if __name__ == "__main__":
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
    trainer = LLaDA2Trainer(CONFIG)
    trainer.train()
