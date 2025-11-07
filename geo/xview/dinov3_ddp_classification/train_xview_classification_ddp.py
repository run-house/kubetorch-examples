"""Multi-label classification on XView with DINOv3 + PyTorch DDP"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from tqdm import tqdm

import kubetorch as kt

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

# XView type_id to class_id mapping (11-94 -> 0-59)
XVIEW_TYPE_TO_CLASS = {
    11: 0, 12: 1, 13: 2, 15: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8,
    23: 9, 24: 10, 25: 11, 26: 12, 27: 13, 28: 14, 29: 15, 32: 16,
    33: 17, 34: 18, 35: 19, 36: 20, 37: 21, 38: 22, 40: 23, 41: 24,
    42: 25, 44: 26, 45: 27, 47: 28, 49: 29, 50: 30, 51: 31, 52: 32,
    53: 33, 54: 34, 55: 35, 56: 36, 57: 37, 59: 38, 60: 39, 61: 40,
    62: 41, 63: 42, 64: 43, 65: 44, 66: 45, 71: 46, 72: 47, 73: 48,
    74: 49, 76: 50, 77: 51, 79: 52, 83: 53, 84: 54, 86: 55, 89: 56,
    91: 57, 93: 58, 94: 59
}


class XViewClassificationDataset(Dataset):
    """Streams XView tiled images from GCS for multi-label classification."""

    def __init__(self, gcs_bucket: str, gcs_prefix: str, creds_path: str,
                 processor, num_classes: int = 60):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.creds_path = creds_path
        self.processor = processor
        self.num_classes = num_classes
        self._client = None
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        self.image_files = self._list_gcs_images()

    @property
    def client(self):
        """Lazy-initialize GCS client per worker process."""
        if self._client is None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.creds_path
            from google.cloud import storage
            self._client = storage.Client()
        return self._client

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_client'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _list_gcs_images(self) -> List[str]:
        """List images from GCS, filtering out empty labels."""
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)

        image_blobs = list(bucket.list_blobs(prefix=f"{self.gcs_prefix}/images/"))
        image_files = [b.name for b in image_blobs if b.name.endswith('.png')]

        label_blobs = list(bucket.list_blobs(prefix=f"{self.gcs_prefix}/labels/"))
        label_sizes = {b.name: b.size for b in label_blobs if not b.name.endswith('/')}

        filtered = [
            img for img in image_files
            if label_sizes.get(img.replace('/images/', '/labels/').replace('.png', '.txt'), 0) > 0
        ]

        print(f"Found {len(filtered)}/{len(image_files)} images with labels")
        client.close()
        return filtered

    def _download_to_memory(self, blob_name: str) -> bytes:
        """Download blob to memory."""
        bucket = self.client.bucket(self.gcs_bucket)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()

    def _load_labels(self, label_bytes: bytes, exclusion_classes=[48, 5]) -> torch.Tensor:
        """Convert YOLO labels to multi-label binary vector."""
        label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        if not label_bytes:
            return label_vector

        for line in label_bytes.decode('utf-8').split('\n'):
            parts = line.strip().split()
            if parts:
                try:
                    type_id = int(parts[0])
                    if type_id in XVIEW_TYPE_TO_CLASS:
                        class_id = XVIEW_TYPE_TO_CLASS[type_id]
                        if class_id not in exclusion_classes:
                            label_vector[class_id] = 1.0
                except ValueError:
                    pass

        return label_vector

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stream image and labels from GCS."""
        img_blob = self.image_files[idx]
        label_blob = img_blob.replace('/images/', '/labels/').replace('.png', '.txt')

        import io
        image = Image.open(io.BytesIO(self._download_to_memory(img_blob))).convert('RGB')
        target = self._load_labels(self._download_to_memory(label_blob))
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, target


class FocalLoss(nn.Module):
    """Focal Loss: FL = -alpha * (1 - p_t)^gamma * log(p_t)"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce_loss = -(targets * torch.log(probs + 1e-8) + (1 - targets) * torch.log(1 - probs + 1e-8))
        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum() if self.reduction == 'sum' else focal_loss


class DINOv3ViT(nn.Module):
    """DINOv3 ViT for feature extraction."""

    def __init__(self, model_name: str = "vitl16", pretrained: bool = True):
        super().__init__()
        assert model_name in DINOV3_MODELS, f"Unsupported model: {model_name}"
        self.model_name = model_name
        self.model_info = DINOV3_MODELS[model_name]
        self.embed_dim = self.model_info["embed_dim"]
        self.model, self.processor = self._load_model(pretrained)

    def _load_model(self, pretrained):
        from transformers import AutoImageProcessor, AutoModel, AutoConfig
        model_id = self.model_info["model_id"]
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        if pretrained:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_config(config)

        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]


class ClassifierHead(nn.Module):
    """Simple MLP classifier head."""

    def __init__(self, embed_dim: int, num_classes: int = 60, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(features)


class XViewClassificationTrainer:
    """DDP trainer for XView multi-label classification."""

    def __init__(self, gcs_bucket: str, gcs_prefix: str, creds_path: str,
                 num_classes: int = 60, model_name: str = "vitl16",
                 freeze_backbone: bool = False, unfreeze_last_n_layers: int = 0):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.creds_path = creds_path
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_layers = unfreeze_last_n_layers

        self.rank = None
        self.world_size = None
        self.device = None
        self.backbone = None
        self.classifier = None
        self.processor = None
        self.train_loader = None
        self.val_loader = None

    def init_comms(self):
        """Initialize DDP."""
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.world_size > 1:
            dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if self.rank == 0:
            print(f"DDP initialized: {self.world_size} workers, device={self.device}")

    def init_model(self):
        """Load and configure model."""
        if self.rank == 0:
            print(f"Loading {self.model_name}...")

        self.backbone = DINOv3ViT(model_name=self.model_name, pretrained=True)
        self.processor = self.backbone.processor

        # Freeze all backbone params
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Optionally unfreeze last N layers for fine-tuning
        if self.unfreeze_last_n_layers > 0:
            hf_model = self.backbone.model
            if hasattr(hf_model, 'layer'):
                total = len(hf_model.layer)
                n = min(self.unfreeze_last_n_layers, total)
                for i in range(total - n, total):
                    for param in hf_model.layer[i].parameters():
                        param.requires_grad = True
                if hasattr(hf_model, 'norm'):
                    for param in hf_model.norm.parameters():
                        param.requires_grad = True
                if self.rank == 0:
                    print(f"Unfroze last {n}/{total} layers + norm")
            self.backbone.train()
        else:
            self.backbone.eval()

        self.backbone = self.backbone.to(self.device)

        # Classifier head
        embed_dim = DINOV3_MODELS[self.model_name]["embed_dim"]
        self.classifier = ClassifierHead(embed_dim, self.num_classes, 512, 0.1).to(self.device)

        # Wrap in DDP
        if self.world_size > 1:
            if self.unfreeze_last_n_layers > 0:
                self.backbone = DDP(self.backbone, device_ids=[self.device.index])
            self.classifier = DDP(self.classifier, device_ids=[self.device.index])

    def load_data(self, batch_size: int = 16, num_workers: int = 8, prefetch_factor: int = 4):
        """Stream data from GCS with distributed sampling."""
        dataset = XViewClassificationDataset(
            self.gcs_bucket, self.gcs_prefix, self.creds_path, self.processor, self.num_classes
        )

        # 80/20 train/val split
        train_size = int(0.8 * len(dataset))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_sampler = DistributedSampler(train_ds, self.world_size, self.rank,
                                           shuffle=True) if self.world_size > 1 else None
        val_sampler = DistributedSampler(val_ds, self.world_size, self.rank,
                                         shuffle=False) if self.world_size > 1 else None

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': num_workers > 0,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        }

        self.train_loader = DataLoader(train_ds, sampler=train_sampler,
                                       shuffle=(train_sampler is None), **loader_kwargs)
        self.val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kwargs)

        if self.rank == 0:
            print(f"Data: {len(train_ds)} train, {len(val_ds)} val | {num_workers} workers")

    def _compute_metrics(self, all_preds: torch.Tensor, all_targets: torch.Tensor,
                         total_loss: float, num_batches: int, epoch: int, mode: str = "Val"):
        """Compute and aggregate metrics across workers."""
        tp = (all_preds * all_targets).sum(dim=0)
        fp = (all_preds * (1 - all_targets)).sum(dim=0)
        fn = ((1 - all_preds) * all_targets).sum(dim=0)
        support = all_targets.sum(dim=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        accuracy = (all_preds == all_targets).float().mean().item()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss, accuracy], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss, accuracy = loss_tensor.tolist()
            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(support, op=dist.ReduceOp.SUM)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        if self.rank == 0:
            valid = support > 0
            num_valid = valid.sum().item()

            if num_valid > 0:
                # Macro averages (unweighted)
                macro_p = precision[valid].mean().item()
                macro_r = recall[valid].mean().item()
                macro_f1 = f1[valid].mean().item()

                # Weighted averages (by support)
                total_support = support.sum()
                weighted_p = (precision * support).sum() / (total_support + 1e-8)
                weighted_r = (recall * support).sum() / (total_support + 1e-8)
                weighted_f1 = (f1 * support).sum() / (total_support + 1e-8)
            else:
                macro_p = macro_r = macro_f1 = 0.0
                weighted_p = weighted_r = weighted_f1 = 0.0

            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} - {mode} Results")
            print(f"{'='*60}")
            print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
            print(f"\nMacro Avg    - P: {macro_p:.3f}  R: {macro_r:.3f}  F1: {macro_f1:.3f}")
            print(
                f"Weighted Avg - P: {weighted_p.item():.3f}  R: {weighted_r.item():.3f}  F1: {weighted_f1.item():.3f}")
            print(f"Classes: {num_valid}/60 present")

            # Show top 5 and bottom 5 classes by F1
            if num_valid > 0:
                valid_f1 = f1.clone()
                valid_f1[~valid] = -1
                sorted_f1, sorted_idx = valid_f1.sort(descending=True)

                # Top 5 best performing classes
                print("\nTop 5 Classes:")
                for i in range(min(5, num_valid)):
                    if sorted_f1[i] >= 0:
                        idx = sorted_idx[i].item()
                        print(
                            f"  Class {idx:2d}: F1={f1[idx].item():.3f} P={precision[idx].item():.3f} R={recall[idx].item():.3f} (n={int(support[idx].item())})")

                # Bottom 5 worst performing classes
                if num_valid > 5:
                    print("\nBottom 5 Classes:")
                    for i in range(max(0, num_valid-5), num_valid):
                        if sorted_f1[i] >= 0:
                            idx = sorted_idx[i].item()
                            print(
                                f"  Class {idx:2d}: F1={f1[idx].item():.3f} P={precision[idx].item():.3f} R={recall[idx].item():.3f} (n={int(support[idx].item())})")

            print(f"{'='*60}\n")
            return avg_loss, weighted_f1.item() if num_valid > 0 else 0.0
        return avg_loss, 0.0

    def train_epoch(self, epoch: int, optimizer, criterion, scaler):
        """Train one epoch."""
        self.classifier.train()
        self.backbone.train() if self.unfreeze_last_n_layers > 0 else self.backbone.eval()
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        all_preds, all_targets = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=self.rank != 0)
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                if self.unfreeze_last_n_layers > 0:
                    features = self.backbone(images)
                else:
                    with torch.no_grad():
                        features = self.backbone(images)
                logits = self.classifier(features)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.detach())
            all_targets.append(targets)
            total_loss += loss.item()

            if self.rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss, _ = self._compute_metrics(
            torch.cat(all_preds), torch.cat(all_targets), total_loss, len(pbar), epoch, "Train"
        )
        return avg_loss

    def validate_epoch(self, epoch: int, criterion):
        """Validate one epoch."""
        self.classifier.eval()
        self.backbone.eval()

        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val {epoch}", disable=self.rank != 0)
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                features = self.backbone(images)
                logits = self.classifier(features)
                loss = criterion(logits, targets)

                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds)
                all_targets.append(targets)
                total_loss += loss.item()

        avg_loss, weighted_f1 = self._compute_metrics(
            torch.cat(all_preds), torch.cat(all_targets), total_loss, len(pbar), epoch, "Val"
        )
        return avg_loss, weighted_f1

    def save_checkpoint(self, epoch: int, optimizer, loss: float, f1_score: float,
                        checkpoint_dir: str = "./checkpoints"):
        """Save checkpoint on rank 0."""
        if self.rank != 0:
            return

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        classifier = self.classifier.module if isinstance(self.classifier, DDP) else self.classifier

        checkpoint = {
            'epoch': epoch,
            'state_dict': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': loss,
            'val_f1': f1_score,
        }

        path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Saved {path}")

    def run(self, epochs: int = 50, batch_size: int = 32, lr: float = 1e-4,
            save_every: int = 5, num_workers: int = 8, prefetch_factor: int = 4):
        """Run distributed training."""
        self.init_comms()
        self.init_model()
        self.load_data(batch_size, num_workers, prefetch_factor)

        # Build optimizer with differential LR for fine-tuning
        params = []
        classifier = self.classifier.module if self.world_size > 1 else self.classifier
        params.append({'params': classifier.parameters(), 'lr': lr})

        if self.unfreeze_last_n_layers > 0:
            backbone = self.backbone.module if (
                self.world_size > 1 and isinstance(self.backbone, DDP)) else self.backbone
            backbone_params = [p for p in backbone.parameters() if p.requires_grad]
            params.append({'params': backbone_params, 'lr': lr * 0.1})
            if self.rank == 0:
                print(f"LR: classifier={lr}, backbone={lr * 0.1}")

        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = FocalLoss(alpha=0.65, gamma=2.0)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        best_f1 = 0.0
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch, optimizer, criterion, scaler)
            val_loss, val_f1 = self.validate_epoch(epoch, criterion)
            scheduler.step()

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, optimizer, val_loss, val_f1)

            if val_f1 > best_f1 and self.rank == 0:
                best_f1 = val_f1
                self.save_checkpoint(epoch, optimizer, val_loss, val_f1, "./checkpoints/best")

        if self.rank == 0:
            print(f"Done! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    gcs_bucket = "rh_demo_image_data"
    gcs_prefix = "xview_tiled/train"
    local_creds_path = "/Users/paulyang/Downloads/runhouse-test-8ad14c2b4edf.json"

    img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install([
        "google-cloud-storage", "tqdm", "pillow", "transformers", "soxr"
    ]).run_bash("uv pip install -U torch --system").set_env_vars({"HF_TOKEN": os.environ["HF_TOKEN"]})

    gpu_compute = kt.Compute(
        gpus=1,
        image=img,
        secrets=[kt.Secret(name="gcp-dataaccess", path=local_creds_path), "huggingface"],
        launch_timeout=1200,
    ).distribute("pytorch", workers=2)

    init_args = dict(
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix,
        creds_path=local_creds_path,
        num_classes=60,
        model_name="vitl16",
        unfreeze_last_n_layers=3,  # 0=frozen, 2-4=partial, 12+=full fine-tuning
    )

    remote_trainer = kt.cls(XViewClassificationTrainer).to(gpu_compute, init_args=init_args)
    remote_trainer.run(epochs=50, batch_size=32, lr=1e-4, save_every=5)
