"""VHR10 Classifier Training with DINOv3 ViT SAT-493M

Trains a classifier on VHR10 satellite imagery using DINOv3 models pretrained on SAT-493M.

After training, you would want to take your
"""

import argparse
import os
from pathlib import Path

import kubetorch as kt
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

DINOV3_MODELS = {
    "vitl16": {
        "model_id": "facebook/dinov3-vitl16-pretrain-sat493m",
        "embed_dim": 1024,
    },
    "vit7b16": {
        "model_id": "facebook/dinov3-vit7b16-pretrain-sat493m",
        "embed_dim": 4096,
    },
}


def vhr10_collate_fn(batch):
    """Collate function for VHR10, used in Dataloader"""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    all_labels = [item["all_labels"] for item in batch]
    return {"image": images, "label": labels, "all_labels": all_labels}


class VHR10ClassificationWrapper(torch.utils.data.Dataset):
    """Converts VHR10 detection to classification.

    Training: Uses most common label for loss computation
    Accuracy: Checks if prediction matches any label in the image
    """

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        labels = sample.get("labels", torch.tensor([]))  # VHR10 uses 'labels' (plural)

        # Convert tensor to PIL
        if isinstance(image, torch.Tensor):
            import numpy as np

            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            image_np = (
                image.cpu().numpy()
                if isinstance(image, torch.Tensor)
                else np.array(image)
            )
            if image_np.dtype != np.uint8:
                image_np = (
                    (image_np * 255).astype(np.uint8)
                    if image_np.max() <= 1.0
                    else image_np.astype(np.uint8)
                )
            image = Image.fromarray(image_np)

        # Preprocess with DINOv3
        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        # Get all labels for accuracy computation
        if isinstance(labels, torch.Tensor) and labels.numel() > 0:
            all_labels = labels.tolist()
        elif isinstance(labels, (list, tuple)) and len(labels) > 0:
            all_labels = list(labels)
        else:
            all_labels = [0]

        # For training: use most common label
        from collections import Counter

        label = Counter(all_labels).most_common(1)[0][0]

        return {"image": pixel_values, "label": label, "all_labels": all_labels}


class DINOv3ViT(nn.Module):
    """DINOv3 ViT for feature extraction from HuggingFace.

    This backbone is completely independent and can be used for inference separately
    from the classifier head. It handles its own preprocessing via the processor.
    """

    def __init__(self, model_name: str = "vitl16", pretrained: bool = True):
        super().__init__()
        if model_name not in DINOV3_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Choose from {list(DINOV3_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_info = DINOV3_MODELS[model_name]
        self.embed_dim = self.model_info["embed_dim"]
        self.model, self.processor = self._load_model(pretrained)

    def _load_model(self, pretrained):
        from transformers import AutoImageProcessor, AutoModel

        model_id = self.model_info["model_id"]
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        if pretrained:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_config(config)

        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor

    def forward(self, x):
        """Extract features from preprocessed pixel values.

        Args:
            x: Preprocessed pixel values tensor [batch, channels, height, width]

        Returns:
            Feature embeddings tensor [batch, embed_dim]
        """
        outputs = self.model(pixel_values=x)
        return (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0]
        )

    def extract_features(self, images, device=None):
        """Standalone inference method that handles preprocessing and feature extraction.

        Args:
            images: PIL Image, list of PIL Images, or preprocessed tensor
            device: Optional device to use (cuda/cpu). If None, uses current device.

        Returns:
            Feature embeddings tensor [batch, embed_dim]
        """
        # Preprocess if not already a tensor
        if not isinstance(images, torch.Tensor):
            processed = self.processor(images=images, return_tensors="pt")
            pixel_values = processed["pixel_values"]
        else:
            pixel_values = images

        if device is not None:
            pixel_values = pixel_values.to(device)
            self.to(device)

        self.eval()
        with torch.no_grad():
            features = self.forward(pixel_values)

        return features


class ClassifierHead(nn.Module):
    """Classification head that operates on pre-computed embeddings.

    This class is independent of the backbone and can be used for inference
    by loading just the classifier weights and passing in pre-computed features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings):
        """Forward pass using pre-computed embeddings.

        Args:
            embeddings: Pre-computed feature embeddings [batch, embed_dim]

        Returns:
            Class logits [batch, num_classes]
        """
        return self.classifier(embeddings)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device=None):
        """Load classifier head from checkpoint.

        Args:
            checkpoint_path: Path to saved checkpoint
            device: Optional device to load to

        Returns:
            Initialized ClassifierHead with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

        # Create classifier with saved dimensions
        classifier = cls(
            embed_dim=checkpoint["embed_dim"],
            num_classes=checkpoint["num_classes"],
        )

        # Load weights - handle both prefixed and non-prefixed keys
        state_dict = checkpoint["state_dict"]
        classifier_state = {}
        for k, v in state_dict.items():
            # Remove 'classifier.' prefix if present
            key = k.replace("classifier.", "") if k.startswith("classifier.") else k
            classifier_state[f"classifier.{key}"] = v

        classifier.load_state_dict(classifier_state)

        if device is not None:
            classifier = classifier.to(device)

        print(
            f"Loaded classifier from {checkpoint_path} (val_acc: {checkpoint.get('val_acc', 'N/A')})"
        )
        return classifier


class VHR10Classifier(nn.Module):
    """VHR10 classifier with DINOv3 backbone and classification head.

    This wrapper class combines the backbone and classifier for training purposes.
    For inference, you can use the backbone and classifier head separately.
    """

    def __init__(
        self, num_classes=10, model_name="vitl16", pretrained=True, freeze_backbone=True
    ):
        super().__init__()
        self.backbone = DINOv3ViT(model_name=model_name, pretrained=pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = ClassifierHead(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        """Forward pass through backbone and classifier.

        Args:
            x: Preprocessed pixel values [batch, channels, height, width]

        Returns:
            Class logits [batch, num_classes]
        """
        embeddings = self.get_features(x)
        return self.classifier(embeddings)

    def get_features(self, x):
        """Extract features from backbone.

        Args:
            x: Preprocessed pixel values [batch, channels, height, width]

        Returns:
            Feature embeddings [batch, embed_dim]
        """
        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.backbone.parameters())
        ):
            return self.backbone(x)


class VHR10Trainer:
    """Distributed trainer for VHR10 classification with DINOv3."""

    def __init__(self, data_root="./data", checkpoint_dir="./checkpoints"):
        self.data_root = data_root
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rank = None
        self.world_size = None
        self.device_id = None
        self.device = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.best_acc = 0.0
        self.model_name = None

    def init_comms(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        self.device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.device_id}")
        torch.cuda.set_device(self.device)

    def init_model(
        self,
        num_classes=10,
        model_name="vitl16",
        freeze_backbone=True,
        lr=1e-4,
        weight_decay=0.01,
    ):
        self.model_name = model_name
        model = VHR10Classifier(num_classes, model_name, True, freeze_backbone).to(
            self.device
        )
        self.model = (
            DDP(model, device_ids=[self.device_id]) if self.world_size > 1 else model
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def load_data(self, batch_size: int = 32, num_workers: int = 4):
        """Create train and validation dataloaders.

        Note: VHR10 is an object detection dataset. We adapt it for classification
        by using the primary (first) label from each image.
        """
        print(f"Loading VHR10 dataset from {self.data_root}")

        # Get the image processor from the model
        if self.model is None:
            raise RuntimeError(
                "Model must be initialized before loading data. Call init_model() first."
            )

        # Get the DINOv3 backbone to access the processor
        if isinstance(self.model, DDP):
            backbone = self.model.module.backbone
        else:
            backbone = self.model.backbone

        processor = backbone.processor
        from torchgeo.datasets import VHR10

        # Load VHR10 dataset (only "positive" split has labels)
        # Note: VHR10 is detection, returns: image, labels, boxes, masks
        # Labels are 1-indexed (1-10) for 10 object classes
        full_dataset = VHR10(
            root=self.data_root,
            split="positive",  # Only positive images have labels
            download=True,
        )

        # Split into train/val (80/20)
        from torch.utils.data import random_split

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(42)  # For reproducibility
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Wrap datasets with processor for proper preprocessing
        train_dataset = VHR10ClassificationWrapper(train_dataset, processor)
        val_dataset = VHR10ClassificationWrapper(val_dataset, processor)

        # Create samplers for distributed training
        train_sampler = (
            DistributedSampler(train_dataset) if self.world_size > 1 else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False)
            if self.world_size > 1
            else None
        )

        # Create dataloaders with module-level collate function
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=vhr10_collate_fn,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=vhr10_collate_fn,
        )

        print(
            f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples"
        )

    def train_epoch(self, epoch: int):
        """Train for one epoch.

        The training explicitly separates the backbone and classifier calls:
        1. Extract features from backbone
        2. Pass features through classifier head

        Loss: Computed using most common label
        Accuracy: Prediction is correct if it matches ANY label in the image
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        num_batches = len(self.train_loader)
        print_interval = max(1, num_batches // 10)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            all_labels = batch["all_labels"]

            # Forward pass - explicitly separate backbone and classifier
            self.optimizer.zero_grad()

            # Step 1: Extract features from backbone
            model = self.model.module if isinstance(self.model, DDP) else self.model
            embeddings = model.get_features(images)

            # Step 2: Pass features through classifier head
            outputs = model.classifier(embeddings)

            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            # Accuracy: check if prediction is in any of the labels for each image
            for i, pred in enumerate(predicted):
                if pred.item() in all_labels[i]:
                    correct += 1

            if (batch_idx + 1) % print_interval == 0:
                print(
                    f"Rank {self.rank}: Epoch {epoch}, Batch {batch_idx+1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%"
                )

        epoch_loss = running_loss / num_batches
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch.

        Also uses explicit separation of backbone and classifier for consistency.
        Accuracy: Prediction is correct if it matches ANY label in the image

        In distributed mode, aggregates metrics across all ranks for accurate validation.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                all_labels = batch["all_labels"]

                # Explicitly separate backbone and classifier
                model = self.model.module if isinstance(self.model, DDP) else self.model
                embeddings = model.get_features(images)
                outputs = model.classifier(embeddings)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)

                # Accuracy: check if prediction is in any of the labels for each image
                for i, pred in enumerate(predicted):
                    if pred.item() in all_labels[i]:
                        correct += 1

        if self.world_size > 1:
            metrics = torch.tensor(
                [running_loss, correct, total], dtype=torch.float32, device=self.device
            )
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            running_loss, correct, total = metrics.tolist()
            total_batches = len(self.val_loader) * self.world_size
            val_loss = running_loss / total_batches
        else:
            val_loss = running_loss / len(self.val_loader)

        val_acc = 100.0 * correct / total

        if self.rank == 0:
            print(
                f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% (aggregated across {self.world_size} rank{'s' if self.world_size > 1 else ''})"
            )

        return val_loss, val_acc

    def train(
        self,
        num_epochs: int,
        batch_size: int = 32,
        lr: float = 1e-4,
        model_name: str = "vitl16",
        freeze_backbone: bool = True,
        num_classes: int = 11,  # VHR10 has labels 1-10, so need 11 classes (0-10)
    ):
        # Initialize distributed communications
        self.init_comms()
        print("Distributed communications initialized")

        # Initialize model
        self.init_model(
            num_classes=num_classes,
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            lr=lr,
        )
        print("Model initialized")

        # Load data
        self.load_data(batch_size=batch_size)
        print("Data loaded")

        # Setup learning rate scheduler - reduces LR when validation plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.65,
            patience=3,
        )

        # Training loop
        for epoch in range(num_epochs):
            # Set epoch for distributed sampler
            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)

            print(f"Starting epoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = self.train_epoch(epoch)
            print("Trained epoch")

            val_loss, val_acc = self.validate_epoch()
            print("Validated epoch")

            # Step scheduler based on validation accuracy
            self.scheduler.step(val_acc)

            # Save best model (only on rank 0)
            if self.rank == 0 and val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(
                    f"vhr10_dinov3_{model_name}_best.pth", epoch, val_acc
                )

            print(
                f"Rank {self.rank}: Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

        print(f"Training complete! Best accuracy: {self.best_acc:.2f}%")

    def save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save classifier head checkpoint (backbone is frozen, no need to save it)."""
        # Get model state dict (unwrap DDP if needed)
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        # Extract only classifier weights (backbone is frozen/pretrained)
        # Keys are like 'classifier.classifier.0.weight', strip to 'classifier.0.weight'
        classifier_state = {
            k.replace("classifier.classifier.", "classifier."): v
            for k, v in model_state.items()
            if k.startswith("classifier.")
        }

        # Get dimensions from the classifier head
        # After stripping, keys are 'classifier.1.weight' (Linear layer)
        embed_dim = classifier_state["classifier.1.weight"].shape[1]
        num_classes = classifier_state["classifier.4.weight"].shape[0]

        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": epoch,
            "state_dict": classifier_state,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "model_name": self.model_name,
        }

        torch.save(checkpoint, checkpoint_path)
        file_size = checkpoint_path.stat().st_size / 1024
        print(
            f"Saved classifier head to {checkpoint_path} ({file_size:.1f} KB, val_acc: {val_acc:.2f}%)"
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load classifier head from checkpoint (backbone loads from HuggingFace)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get the model (unwrap DDP if needed)
        model = self.model.module if isinstance(self.model, DDP) else self.model

        # Load only classifier weights (backbone is already loaded from HuggingFace)
        classifier_state = {
            f"classifier.{k}": v for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(classifier_state, strict=False)

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(
            f"Loaded classifier head from {checkpoint_path} (val_acc: {checkpoint.get('val_acc', 'N/A')})"
        )
        return checkpoint

    def extract_features(self, images):
        """Extract features from images using the backbone.
        This method can be used for inference-only feature extraction.
        Args:
            images: Can be PIL Images, list of PIL Images, or preprocessed tensor
        Returns:
            Feature tensor of shape [batch, embed_dim]
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() first.")

        # Get the processor from the backbone
        if isinstance(self.model, DDP):
            processor = self.model.module.backbone.processor
        else:
            processor = self.model.backbone.processor

        # Preprocess if not already a tensor
        if not isinstance(images, torch.Tensor):
            # Assume PIL Images or list of PIL Images
            processed = processor(images=images, return_tensors="pt")
            images = processed["pixel_values"].to(self.device)
        else:
            images = images.to(self.device)

        self.model.eval()

        with torch.no_grad():
            # Get the model (unwrap DDP if needed)
            model = self.model.module if isinstance(self.model, DDP) else self.model
            features = model.get_features(images)

        return features

    def predict(self, image):
        """Predict class for a single image."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() first.")

        # Get the processor from the backbone
        if isinstance(self.model, DDP):
            processor = self.model.module.backbone.processor
        else:
            processor = self.model.backbone.processor

        # Use the processor for proper preprocessing
        processed = processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(pixel_values)
            _, predicted = torch.max(output, 1)
            return predicted.item()


# ## Run Distributed Training
# The following demonstrates how to launch compute and run the distributed training pipeline.
# - We define compute with GPUs and call .distribute('pytorch') to setup distributed training
# - Then we dispatch the trainer class to the remote compute using kt.cls
# - We create an instance of the trainer class on remote, which is now running distributed
# - Multiple methods can be called on the remote trainer instance
def main():
    parser = argparse.ArgumentParser(description="VHR10 Classification with DINOv3")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--data-root", type=str, default="./data", help="data root directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="checkpoint directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vitl16",
        choices=["vitl16", "vit7b16"],
        help="DINOv3 model variant (vitl16: distilled, vit7b16: original)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="freeze backbone weights",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="number of distributed workers"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=11,
        help="number of classes (VHR10 uses 11 for labels 1-10)",
    )

    args = parser.parse_args()

    # Define compute configuration
    img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install(
        [
            "torchgeo",
            "transformers",
            "torchvision",
            "pillow",
            "soxr",  # Required by transformers for audio_utils
        ]
    )

    gpu_compute = kt.Compute(
        gpus=1,
        image=img,
        launch_timeout=600,
        inactivity_ttl="2h",
        secrets=["huggingface"],
    ).distribute("pytorch", workers=args.workers)

    # Initialize trainer arguments
    init_args = dict(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Dispatch trainer class to remote GPUs
    remote_trainer = kt.cls(VHR10Trainer).to(gpu_compute, init_args=init_args)

    # Run distributed training
    print("Starting distributed training...")
    remote_trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
