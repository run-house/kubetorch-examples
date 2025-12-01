# # PyTorch Distributed Training with Dynamic World Size
# This example demonstrates how Kubetorch handles dynamic scaling of distributed training jobs.
# Unlike traditional preemption recovery which assumes pods fail and restart, this example shows
# how to adapt training when pods are added or removed from the cluster, allowing elastic scaling
# of distributed workloads.

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict

# Suppress specific warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

import kubetorch as kt
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        self.distributed_shuffled_dataset = None
        # Use tensors for epoch and loss so they're backed by server memory
        self.epoch_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
        self.loss_tensor = torch.tensor([0.0], dtype=torch.float32, device="cuda")

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

        # Persist model state to object store for fault tolerance
        # The returned dict contains tensors backed by the server's memory
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

        # Store everything as tensors - epoch and loss will auto-update in server memory
        checkpoint_to_store = {
            "model_state_dict": model_state,
            "epoch": self.epoch_tensor,
            "loss": self.loss_tensor,
        }

        # Get back server-backed versions
        persisted_checkpoint = kt.vput("berttrain", checkpoint_to_store)

        # Load the server-backed state dict into the model
        # This ensures the model uses tensors owned by the server process
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(persisted_checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(persisted_checkpoint["model_state_dict"])

        # Update our local references to use server-backed tensors
        self.epoch_tensor = persisted_checkpoint["epoch"]
        self.loss_tensor = persisted_checkpoint["loss"]

        print(f"Rank {self.rank}: Model state persisted to object store")
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

            def set_epoch(self, epoch):
                """Update epoch for shuffling."""
                self.epoch = epoch

            def __iter__(self):
                # Shuffle with epoch-based seed for consistent shuffling across ranks
                dataset_iter = iter(self.dataset.shuffle(seed=self.epoch, buffer_size=1000))
                # Each rank processes every world_size-th example
                for i, example in enumerate(dataset_iter):
                    if i % self.world_size == self.rank:
                        yield example

        # Wrap the streaming dataset for distributed training
        # Use current epoch from tensor for initial shuffling
        current_epoch = self.epoch_tensor.item() if hasattr(self, "epoch_tensor") else 0
        self.distributed_shuffled_dataset = DistributedStreamingWrapper(
            self.dataset, self.rank, self.world_size, epoch=current_epoch
        )

        self.dataloader = DataLoader(
            self.distributed_shuffled_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            drop_last=True,  # Drop last batch if incomplete for DDP
        )

    def _sync_after_restart(self):
        """Synchronizes model weights across ranks after world size changes.

        When the world size changes (workers added/removed), this method ensures all ranks
        have consistent model weights by finding the rank with the most recent state
        from the object store and broadcasting its weights to all other ranks.
        """
        # Pop from object store to get owned tensors (all ranks do this)
        # This transfers ownership from store to this process, enabling broadcast
        checkpoint = kt.pop("berttrain", default=None)

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # These are now regular tensors owned by this process
            self.epoch_tensor = checkpoint["epoch"]
            self.loss_tensor = checkpoint["loss"]

            # Read the current epoch value from the tensor
            current_epoch = self.epoch_tensor.item()
            print(f"Rank {self.rank}: Restored from object store at epoch {current_epoch}")
        else:
            print(f"Rank {self.rank}: No state found in object store")
            # Initialize tensors if not found
            self.epoch_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
            self.loss_tensor = torch.tensor([0.0], dtype=torch.float32, device="cuda")

        # ## Finding the Most Recent State
        # Get max epoch across all ranks to identify who has the latest checkpoint
        # Use the epoch from our tensor
        current_epoch = self.epoch_tensor.item() if hasattr(self, "epoch_tensor") else 0
        epoch_for_sync = torch.tensor([current_epoch], dtype=torch.long, device=self.device)
        torch.distributed.all_reduce(epoch_for_sync, op=torch.distributed.ReduceOp.MAX)
        max_epoch = epoch_for_sync[0].item()

        # ## Selecting Source Rank
        # Find the rank with the max epoch (use min rank for determinism when multiple have same epoch)
        has_max_epoch = torch.tensor(
            [self.rank if current_epoch == max_epoch else self.world_size],
            dtype=torch.long,
            device=self.device,
        )
        torch.distributed.all_reduce(has_max_epoch, op=torch.distributed.ReduceOp.MIN)
        source_rank = has_max_epoch[0].item()

        # Update epoch tensor if needed
        if hasattr(self, "epoch_tensor"):
            self.epoch_tensor[0] = max_epoch
        else:
            self.epoch_tensor = torch.tensor([max_epoch], dtype=torch.long, device="cuda")

        # ## Broadcasting Weights
        # All ranks participate in weight sync to ensure consistency after world size change.
        # This is crucial for maintaining training stability when workers are added/removed
        if max_epoch > 0:
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

        # Get starting epoch from tensor
        start_epoch = self.epoch_tensor.item()

        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"Training epoch {epoch}")
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            # Set epoch for distributed dataset shuffling
            if hasattr(self, "distributed_dataset"):
                self.distributed_shuffled_dataset.set_epoch(epoch)

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
                # Update loss tensor in-place (automatically updates in server memory)
                self.loss_tensor[0] = loss.item()

            # Increment epoch tensor in-place
            self.epoch_tensor[0] = epoch + 1
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.loss_tensor[0] = avg_epoch_loss
            print(f"Rank {self.rank}: Epoch {self.epoch_tensor.item()} complete, Avg Loss: {avg_epoch_loss:.4f}")

        return {
            "rank": self.rank,
            "loss": self.loss_tensor.item(),
            "epochs_completed": self.epoch_tensor.item(),
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
    max_retries = 10

    results = None
    while completed_epochs < args.epochs:
        try:
            # Train for one epoch at a time to handle membership changes
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
            new_worker_num = len(trainer.compute.pod_names())
            print(f"World size changed, continuing with {new_worker_num} workers")
            # time.sleep(5)  # Give DNS a moment to update.
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
