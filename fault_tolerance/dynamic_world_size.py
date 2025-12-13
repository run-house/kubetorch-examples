# # PyTorch Distributed Training with Dynamic World Size
# This example demonstrates how Kubetorch handles dynamic scaling of distributed training jobs.
# Unlike traditional preemption recovery which assumes pods fail and restart, this example shows
# how to adapt training when pods are added or removed from the cluster, allowing elastic scaling
# of distributed workloads.
#
# ::youtube[Fault Tolerance]{url="https://www.youtube.com/watch?v=k1olO4P_1WY"}
#
# We begin by importing the necessary libraries and setting up the environment.

import argparse
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Dict

import kubetorch as kt
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

# Configure NCCL to be more resilient to failures

os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "10"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"  # Disable watchdog killing process
os.environ["NCCL_ABORT_ON_ERROR"] = "1"

# ## BERT Trainer Class
# A BERT fine-tuning trainer that handles distributed training with dynamic world size changes.
# The trainer can adapt to workers joining or leaving the training process without losing progress.
class BERTTrainer:
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
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
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
            print("Process group already initialized, completing setup for this instance.")

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        print(f"Rank {self.rank}/{self.world_size} initialized on {self.device}")

        self.model = self.model.to(self.device)

        # Handle restart recovery BEFORE DDP to avoid NCCL errors
        self._sync_after_restart()

        self.model = DDP(self.model, device_ids=[self.device])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # ## Collate Function for On-the-Fly Tokenization
        # Tokenize batches as they're loaded to avoid upfront processing delays
        def collate_fn(examples):
            texts = [ex["text"] for ex in examples]
            labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)

            # Tokenize the batch
            encoding = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "label": labels,
            }

        # ## Distributed Sharding for Streaming Dataset
        # For streaming datasets, we shard by skipping examples rather than using DistributedSampler
        # Each rank only processes every Nth example where N is the world size
        from torch.utils.data import IterableDataset

        class DistributedStreamingWrapper(IterableDataset):
            def __init__(self, dataset, rank, world_size, epoch=0):
                self.dataset = dataset
                self.rank = rank
                self.world_size = world_size
                self.epoch = epoch

            def __iter__(self):
                # Shuffle with fixed seed for this instance
                dataset_iter = iter(self.dataset.shuffle(seed=self.epoch, buffer_size=1000))
                # Each rank processes every world_size-th example
                for i, example in enumerate(dataset_iter):
                    if i % self.world_size == self.rank:
                        yield example

        # Wrap the streaming dataset for distributed training with fixed epoch
        distributed_dataset = DistributedStreamingWrapper(self.dataset, self.rank, self.world_size, epoch=self.epoch)

        self.dataloader = DataLoader(
            distributed_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            drop_last=True,  # Drop last batch if incomplete for DDP
        )

    def _get_checkpoint_path(self) -> Path:
        """Get the path for the latest checkpoint."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / "checkpoint_latest.pt"

    def save_checkpoint(self):
        """Save training checkpoint for preemption recovery."""
        # We only need one rank to save the checkpoint per worker
        # No need to save a checkpoint if no losses have been recorded yet, i.e. no batches processed
        if os.environ.get("LOCAL_RANK", "0") != "0" or self.latest_loss is None:
            return

        try:
            model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

            checkpoint = {
                "epoch": self.epoch,
                "model_state_dict": model_state,
                "loss": self.latest_loss,
            }
            checkpoint_path = self._get_checkpoint_path()
            temp_path = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temp_path)
            temp_path.rename(checkpoint_path)

            print(f"Rank {self.rank}: Checkpoint saved at epoch {self.epoch} to {checkpoint_path}")

        except Exception as e:
            print(f"Rank {self.rank}: Failed to save checkpoint: {e}")
            # Don't re-raise - checkpoint save failure shouldn't crash training

    def _sync_after_restart(self):
        """Synchronizes model weights across ranks after world size changes.

        When the world size changes (workers added/removed), this method ensures all ranks
        have consistent model weights by finding the rank with the most recent checkpoint
        and broadcasting its weights to all other ranks.
        """
        checkpoint_path = self._get_checkpoint_path()
        has_checkpoint = checkpoint_path.exists()

        if has_checkpoint:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.epoch = checkpoint["epoch"] + 1
            print(f"Rank {self.rank}: Restored from local checkpoint at epoch {checkpoint['epoch']}")
        else:
            print(f"Rank {self.rank}: No local checkpoint found")
            self.epoch = 0

        # ## Finding the Most Recent State
        # Get max epoch across all ranks to identify who has the latest checkpoint
        epoch_tensor = torch.tensor([self.epoch], dtype=torch.long, device=self.device)
        torch.distributed.all_reduce(epoch_tensor, op=torch.distributed.ReduceOp.MAX)
        max_epoch = epoch_tensor[0].item()

        # ## Selecting Source Rank
        # Find the rank with the max epoch (use min rank for determinism when multiple have same epoch)
        has_max_epoch = torch.tensor(
            [self.rank if self.epoch == max_epoch else self.world_size],
            dtype=torch.long,
            device=self.device,
        )
        torch.distributed.all_reduce(has_max_epoch, op=torch.distributed.ReduceOp.MIN)
        source_rank = has_max_epoch[0].item()

        self.epoch = max_epoch

        # ## Broadcasting Weights
        # All ranks participate in weight sync to ensure consistency after world size change.
        # This is crucial for maintaining training stability when workers are added/removed
        if self.epoch > 0:
            if self.rank == source_rank:
                print(f"Rank {self.rank}: Broadcasting weights to other ranks (epoch {max_epoch})")
            else:
                print(f"Rank {self.rank}: Receiving weights from rank {source_rank} (epoch {max_epoch})")

            for param in self.model.parameters():
                torch.distributed.broadcast(param.data, src=source_rank)

        torch.distributed.barrier()

    def _load_dataset(self):
        """Load IMDB dataset from HuggingFace using streaming for faster startup."""
        from datasets import load_dataset

        print("Loading IMDB dataset with streaming enabled...")

        # ## Streaming Dataset for Fast Startup
        # Using streaming=True avoids downloading/processing the entire dataset upfront.
        # This is crucial for elastic training where workers may join/leave frequently.
        # Tokenization happens on-the-fly in the collate function, eliminating startup delays.
        return load_dataset("imdb", split="train", streaming=True)

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
            print(f"Rank {self.rank}: Epoch {epoch} complete, Avg Loss: {avg_epoch_loss:.4f}")

        return {
            "rank": self.rank,
            "loss": self.latest_loss,
            "epochs_completed": self.epoch,
        }


# ## Main Training Loop with Dynamic World Size
# Main entry point that sets up distributed training and handles dynamic world size changes.
# The training adapts to workers being added or removed, redistributing the workload
# and synchronizing model state across the new set of workers.
def main():
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training with Preemption Recovery")
    parser.add_argument("--epochs", type=int, default=6, help="number of epochs to train (default: 5)")
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
        default=4,
        help="number of distributed workers (default: 2)",
    )
    args = parser.parse_args()

    # Define compute environment with GPU support
    gpus = kt.Compute(
        gpus=1,  # 1 GPU per worker
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install(["transformers==4.36.0", "datasets"]),
        launch_timeout=600,
    ).distribute("pytorch", workers=args.workers, port=12345)

    # Create and dispatch the trainer to remote compute
    trainer_args = {
        "model_name": args.model,
        "batch_size": args.batch_size,
    }
    trainer = kt.cls(BERTTrainer).to(gpus, init_args=trainer_args)

    # ## Training with Dynamic World Size Handling
    # The training loop handles WorkerMembershipChanged exceptions which occur when
    # pods are added or removed from the distributed training cluster
    completed_epochs = 0
    retries = 0

    def cache_worker():
        """Background thread that periodically saves checkpoints.

        This ensures we have recent checkpoints available when world size changes,
        allowing new or restarted workers to synchronize with the latest training state.
        """
        while completed_epochs < args.epochs:
            try:
                trainer.save_checkpoint(workers="ready")
            except Exception as e:
                print(f"Cache worker encountered exception: {e}")
            time.sleep(30)  # Update cache every 30 seconds

    results = None
    max_retries = 10
    while completed_epochs < args.epochs:
        try:
            # ## Checkpoint Caching During Training
            # Start background caching inside the try block to handle membership changes
            # that might occur during checkpoint operations. This ensures checkpoints are
            # available for weight synchronization when world size changes.
            cache_thread = threading.Thread(target=cache_worker, daemon=True)
            cache_thread.start()

            results = trainer.train(epochs=1)
            completed_epochs += 1
        except (kt.WorkerMembershipChanged, kt.PodTerminatedError) as e:
            # ## Handling World Size Changes
            # When workers are added or removed, Kubetorch raises WorkerMembershipChanged.
            # We respond by:
            # 1. Updating the distributed configuration with the new worker count
            # 2. Re-initializing the trainer on the new set of workers
            # 3. The trainer's _sync_after_restart() ensures all workers have consistent weights
            if retries >= max_retries:
                print(f"Training failed after {retries} retries. Exiting.")
                raise e
            retries += 1
            new_worker_num = len(gpus.pod_names())
            print(f"World size changed, continuing with {new_worker_num} workers")
            # Reconfigure distributed training with new world size
            gpus.distribute("pytorch", workers=new_worker_num, port=12345 + retries)
            # Re-initialize trainer, which will sync weights across new worker set
            # Increment the port because sometimes PyTorch can take a few seconds to free it
            trainer = kt.cls(BERTTrainer).to(gpus, init_args=trainer_args)

    print("\nTraining completed on all ranks:")
    for rank_result in results:
        print(
            f"  Rank {rank_result['rank']}: Final Loss = {rank_result['loss']:.4f}, "
            f"Epochs Completed = {rank_result['epochs_completed']}"
        )

    print("\nNote: Training automatically adapted to world size changes and synchronized weights!")


if __name__ == "__main__":
    main()
