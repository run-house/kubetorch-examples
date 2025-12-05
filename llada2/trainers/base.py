"""Base Trainer for LLaDA2 models.

Provides shared FSDP, model loading, optimizer, and attention mask setup
for both SFT and GRPO training.
"""
import os
from abc import ABC, abstractmethod
from functools import partial
from glob import glob
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

from llada2.llada2_moe_vanilla import LLaDA2Config, LLaDA2DecoderLayer, LLaDA2ForCausalLM
from llada2.utils import create_block_diffusion_mask


class BaseTrainer(ABC):
    """Base trainer with shared FSDP, model, and optimizer setup.

    Subclasses must implement the training loop logic.
    """

    def __init__(self, config: Dict, kt_cached_state: Optional[Dict] = None):
        """Initialize trainer with config.

        Args:
            config: Configuration dictionary
            kt_cached_state: Optional cached state from kubetorch for hot-reload
        """
        if kt_cached_state:
            self._restore_from_cache(kt_cached_state)
            return

        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.attn_mask = None
        self.device = None
        self.steps = 0

        # Distributed state
        self.global_rank = None
        self.local_rank = None
        self.world_size = None

        # FSDP group state (set in setup_distributed)
        self.fsdp_process_group = None
        self.fsdp_sharding_group_size = None
        self.num_model_replicas = None
        self.replica_id = None
        self.rank_in_replica = None

    def _restore_from_cache(self, state: Dict):
        """Restore trainer state from kubetorch cache."""
        print("Reusing existing trainer from cached state")
        self.config = state["config"]
        self.model = state["model"]
        self.tokenizer = state["tokenizer"]
        self.optimizer = state["optimizer"]
        self.scheduler = state["scheduler"]
        self.attn_mask = state["attn_mask"]
        self.device = state["device"]
        self.steps = state["steps"]
        self.global_rank = state["global_rank"]
        self.local_rank = state["local_rank"]
        self.world_size = state["world_size"]
        # Restore FSDP state if present
        self.fsdp_process_group = state.get("fsdp_process_group")
        self.fsdp_sharding_group_size = state.get("fsdp_sharding_group_size")
        self.num_model_replicas = state.get("num_model_replicas")
        self.replica_id = state.get("replica_id")
        self.rank_in_replica = state.get("rank_in_replica")

    def __kt_cached_state__(self) -> Dict[str, Any]:
        """Return state to be cached by kubetorch across reloads."""
        return {
            "config": self.config,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "attn_mask": self.attn_mask,
            "device": self.device,
            "steps": self.steps,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "fsdp_process_group": self.fsdp_process_group,
            "fsdp_sharding_group_size": self.fsdp_sharding_group_size,
            "num_model_replicas": self.num_model_replicas,
            "replica_id": self.replica_id,
            "rank_in_replica": self.rank_in_replica,
        }

    def setup(self):
        """Initialize model, optimizer, and distributed training."""
        self.setup_distributed()

        if self.global_rank == 0:
            print(f"Setting up trainer with {self.world_size} GPUs")
            os.makedirs(self.config.get("output_dir", "./checkpoints"), exist_ok=True)

        self.setup_tokenizer()
        self.setup_model()
        self.setup_optimizer()
        self.setup_attention_mask()

        if self.global_rank == 0:
            print("Trainer setup complete")

    def setup_distributed(self):
        """Setup distributed training with NCCL backend."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        dist.barrier()
        self._setup_fsdp_groups()

    def _setup_fsdp_groups(self):
        """Setup FSDP sharding groups for multi-replica training."""
        self.fsdp_sharding_group_size = self.config.get(
            "fsdp_sharding_group_size", self.world_size
        )

        if self.fsdp_sharding_group_size > 1 and self.world_size > 1:
            assert (
                self.world_size % self.fsdp_sharding_group_size == 0
            ), f"world_size ({self.world_size}) must be divisible by fsdp_sharding_group_size ({self.fsdp_sharding_group_size})"

            self.num_model_replicas = self.world_size // self.fsdp_sharding_group_size
            self.replica_id = self.global_rank // self.fsdp_sharding_group_size
            self.rank_in_replica = self.global_rank % self.fsdp_sharding_group_size

            # Create process groups - each replica has its own group
            fsdp_process_groups = []
            for i in range(self.num_model_replicas):
                replica_ranks = list(
                    range(
                        i * self.fsdp_sharding_group_size,
                        (i + 1) * self.fsdp_sharding_group_size,
                    )
                )
                group = dist.new_group(ranks=replica_ranks)
                fsdp_process_groups.append(group)

            self.fsdp_process_group = fsdp_process_groups[self.replica_id]

            if self.global_rank == 0:
                print(f"FSDP: {self.world_size} GPUs, {self.fsdp_sharding_group_size} GPUs/replica, {self.num_model_replicas} replicas")
        else:
            self.num_model_replicas = self.world_size if self.world_size > 0 else 1
            self.replica_id = self.global_rank
            self.rank_in_replica = 0
            self.fsdp_process_group = None

    def setup_tokenizer(self):
        """Setup tokenizer."""
        if self.global_rank == 0:
            print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_model(self):
        """Setup LLaDA2 model with FSDP wrapping."""
        if self.global_rank == 0:
            print("Loading model...")

        model_path = self.config["model_path"]
        self._ensure_model_downloaded(model_path)

        model_config = LLaDA2Config.from_pretrained(model_path)
        model_config._attn_implementation = self.config.get("attn_implementation", "sdpa")

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
            param_init_fn=lambda module: module.to_empty(device=self.device, recurse=False),
            process_group=self.fsdp_process_group,
        )

        self.load_model_weights(model_path)

        if self.config.get("use_gradient_checkpointing", True):
            if self.global_rank == 0:
                print("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()

    def _ensure_model_downloaded(self, model_path: str):
        """Download model from HuggingFace if not present locally."""
        if not os.path.exists(model_path):
            print(f"[RANK {self.global_rank}] Downloading model from HuggingFace: {model_path}")
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=model_path, local_dir=model_path)
            except Exception as e:
                print(f"[RANK {self.global_rank}] Download error: {e}")

        dist.barrier()

    def load_model_weights(self, model_path: str):
        """Load pretrained weights into FSDP-wrapped model."""
        if self.global_rank == 0:
            print("Loading model weights...")

        from safetensors.torch import safe_open

        index_file = os.path.join(model_path, "model.safetensors.index.json")
        single_file = os.path.join(model_path, "model.safetensors")

        state_dict = {}
        if os.path.exists(index_file):
            safetensor_files = sorted(glob(os.path.join(model_path, "model-*.safetensors")))
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

        self.model.load_state_dict(state_dict, strict=False)
        del state_dict

        if self.global_rank == 0:
            print("Model weights loaded")

    def setup_optimizer(self):
        """Setup optimizer (AdamW8bit if available, else AdamW)."""
        lr = self.config.get("learning_rate", self.config.get("lr", 1e-6))
        weight_decay = self.config.get("weight_decay", 0.01)

        if HAS_BITSANDBYTES:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                fused=True,
            )

        if self.global_rank == 0:
            print(f"Optimizer: lr={lr}")

    def setup_attention_mask(self):
        """Setup block diffusion attention mask."""
        self.attn_mask = create_block_diffusion_mask(
            self.config["max_seq_len"],
            self.config["block_size"],
            self.device,
        )

        if self.global_rank == 0:
            print(f"Attention mask: max_seq_len={self.config['max_seq_len']}, block_size={self.config['block_size']}")

    @abstractmethod
    def train(self, *args, **kwargs):
        """Run the training loop. Must be implemented by subclasses."""
        pass
