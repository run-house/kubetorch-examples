# # PyTorch Multi-Node Distributed Training with Preemption Recovery
# This example demonstrates how Kubetorch handles preemptions natively by training a BERT model
# for text classification with automatic checkpoint saving and recovery. The training can be
# interrupted at any time and will resume from the last checkpoint.

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Dict

os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "10"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"  # Disable watchdog killing process
os.environ["NCCL_ABORT_ON_ERROR"] = "1"

import kubetorch as kt
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BERTTrainer:
    """
    A concise BERT fine-tuning trainer that handles distributed training and preemption recovery.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        checkpoint_dir: str = "/tmp/bert_checkpoints",
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.rank = None
        self.world_size = None
        self.device = None
        self.dataloader = None

        # Load model, tokenizer, and dataset (heavy operations which should block "ready" state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.dataset = self._load_dataset()
        self.epoch = 0
        self.latest_loss = None  # Track latest loss for checkpointing

    def setup(self):
        """Initialize distributed training, model, and optimizer with restart recovery."""
        if torch.distributed.is_initialized() and self.dataloader is not None:
            print("Setup already complete for this instance, skipping.")
            return

        if not torch.distributed.is_initialized():
            print("Connecting to process group...")
            torch.distributed.init_process_group(backend="nccl")
        else:
            print(
                "Process group already initialized, completing setup for this instance."
            )

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        print(f"Rank {self.rank}/{self.world_size} initialized on {self.device}")

        self.model = self.model.to(self.device)

        # Handle restart recovery BEFORE DDP to avoid NCCL errors
        self._sync_after_restart()

        self.model = DDP(self.model, device_ids=[self.device])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        # Create distributed sampler for multi-GPU training
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,  # Drop last batch if incomplete for DDP
        )

    def _get_checkpoint_path(self) -> Path:
        """Get the path for the latest checkpoint."""
        return self.checkpoint_dir / "checkpoint_latest.pt"

    def save_checkpoint(self):
        """Save training checkpoint for preemption recovery."""
        # We only need one rank to save the checkpoint per worker
        # No need to save a checkpoint if no losses have been recorded yet, i.e. no batches processed
        if os.environ.get("LOCAL_RANK", "0") != "0" or self.latest_loss is None:
            return

        print(f"Rank {self.rank}: Saving checkpoint at epoch {self.epoch}")

        try:
            model_state = (
                self.model.module.state_dict()
                if hasattr(self.model, "module")
                else self.model.state_dict()
            )
            print(f"Rank {self.rank}: Model state extracted")

            checkpoint = {
                "epoch": self.epoch,
                "model_state_dict": model_state,
                "loss": self.latest_loss,
            }
            checkpoint_path = self._get_checkpoint_path()
            temp_path = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temp_path)
            temp_path.rename(checkpoint_path)

            print(f"Checkpoint saved at epoch {self.epoch} to {checkpoint_path}")

        except Exception as e:
            print(f"Rank {self.rank}: Failed to save checkpoint: {e}")
            # Don't re-raise - checkpoint save failure shouldn't crash training

    def _sync_after_restart(self):
        """Initial sync when starting up (simplified version)."""
        checkpoint_path = self._get_checkpoint_path()
        has_checkpoint = checkpoint_path.exists()

        if has_checkpoint:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.epoch = checkpoint["epoch"] + 1
            print(
                f"Rank {self.rank}: Restored from local checkpoint at epoch {checkpoint['epoch']}"
            )
        else:
            print(f"Rank {self.rank}: No local checkpoint found")
            self.epoch = 0

        # Get max epoch across all ranks
        epoch_tensor = torch.tensor([self.epoch], dtype=torch.long, device=self.device)
        torch.distributed.all_reduce(epoch_tensor, op=torch.distributed.ReduceOp.MAX)
        max_epoch = epoch_tensor[0].item()

        # Find a rank that has the max epoch (use min rank for determinism and to only select one)
        has_max_epoch = torch.tensor(
            [self.rank if self.epoch == max_epoch else self.world_size],
            dtype=torch.long,
            device=self.device,
        )
        torch.distributed.all_reduce(has_max_epoch, op=torch.distributed.ReduceOp.MIN)
        source_rank = has_max_epoch[0].item()

        self.epoch = max_epoch

        # All ranks participate in weight sync (both sender and receivers)
        # Don't sync if all ranks have epoch 0, i.e. no checkpoint found on any rank
        if self.epoch > 0:
            if self.rank == source_rank:
                print(
                    f"Rank {self.rank}: Broadcasting weights to other ranks (epoch {max_epoch})"
                )
            else:
                print(
                    f"Rank {self.rank}: Receiving weights from rank {source_rank} (epoch {max_epoch})"
                )

            for param in self.model.parameters():
                torch.distributed.broadcast(param.data, src=source_rank)

        torch.distributed.barrier()

    def _load_dataset(self):
        """Load IMDB dataset from HuggingFace for sentiment classification with caching."""
        import pickle

        from datasets import load_dataset

        # Create cache path for tokenized dataset
        cache_path = self.checkpoint_dir / "tokenized_imdb_dataset.pkl"

        # Try to load cached tokenized dataset
        if cache_path.exists():
            print(f"Loading cached tokenized dataset from {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    tokenized_dataset = pickle.load(f)
                print("Successfully loaded cached dataset")
                return tokenized_dataset
            except Exception as e:
                print(f"Failed to load cached dataset: {e}, reprocessing...")

        print("Loading and tokenizing IMDB dataset.")

        # Load IMDB dataset
        dataset = load_dataset("imdb", split="train")

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )
        tokenized_dataset.set_format("torch")

        # Save tokenized dataset to cache
        try:
            print(f"Saving tokenized dataset to cache at {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(tokenized_dataset, f)
            print("Dataset cached successfully")
        except Exception as e:
            print(f"Failed to cache dataset: {e}")

        return tokenized_dataset

    def train(self, epochs: int = 3) -> Dict:
        """
        Main training loop with checkpoint saving for preemption recovery.
        """
        self.setup()

        for epoch in range(self.epoch, self.epoch + epochs):
            print(f"Training epoch {epoch}")
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            # Set epoch for distributed sampler
            self.dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.dataloader):
                start_time = time.time()
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs.logits, labels)

                # Backward pass (this is where DDP hangs if a rank is dead)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                end_time = time.time()
                if batch_idx % 5 == 0:
                    print(
                        f"Rank {self.rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Time/Batch: {end_time - start_time:.2f}s"
                    )
                self.latest_loss = loss.item()  # Update latest loss for checkpointing

            self.epoch += 1  # Update last completed epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.latest_loss = avg_epoch_loss
            print(
                f"Rank {self.rank}: Epoch {epoch} complete, Avg Loss: {avg_epoch_loss:.4f}"
            )

        return {
            "rank": self.rank,
            "loss": self.latest_loss,
            "epochs_completed": self.epoch,
        }


def main():
    """
    Main entry point that sets up distributed training and handles preemption recovery.
    The training can be interrupted at any point and will automatically resume from
    the last checkpoint when restarted.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Training with Preemption Recovery"
    )
    parser.add_argument(
        "--epochs", type=int, default=6, help="number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model to fine-tune (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of distributed workers (default: 2)",
    )
    args = parser.parse_args()

    # Define compute environment with GPU support
    gpus = kt.Compute(
        gpus=1,  # 1 GPU per worker
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install(
            ["transformers==4.36.0", "datasets"]
        ),
        launch_timeout=600,
        inactivity_ttl="1h",
    ).distribute("pytorch", workers=args.workers)

    # Create and dispatch the trainer to remote compute
    trainer_args = {
        "model_name": args.model,
        "checkpoint_dir": "/tmp/bert_checkpoints",
        "batch_size": args.batch_size,
    }
    trainer = kt.cls(BERTTrainer).to(gpus, init_args=trainer_args)

    # Run training - will automatically recover from checkpoints if interrupted
    completed_epochs = 0
    retries = 0

    def cache_worker():
        while completed_epochs < args.epochs:
            try:
                trainer.save_checkpoint(workers="ready")
            except Exception as e:
                print(f"Cache worker encountered exception: {e}")
            time.sleep(30)  # Update cache every 30 seconds

    # Start background caching
    cache_thread = threading.Thread(target=cache_worker, daemon=True)
    cache_thread.start()

    results = None
    while completed_epochs < args.epochs:
        try:
            results = trainer.train(epochs=1)
            completed_epochs += 1
        except Exception as e:
            if retries >= 3:
                print(f"Training failed after {retries} retries. Exiting.")
                raise e
            print(f"Training interrupted with exception: {e}. Retrying...")
            retries += 1
            time.sleep(20)
            trainer.setup(restart_procs=True)

    print("\nTraining completed on all ranks:")
    for rank_result in results:
        print(
            f"  Rank {rank_result['rank']}: Final Loss = {rank_result['loss']:.4f}, "
            f"Epochs Completed = {rank_result['epochs_completed']}"
        )

    print(
        "\nNote: If training was interrupted, it automatically resumed from the last checkpoint!"
    )


if __name__ == "__main__":
    main()
