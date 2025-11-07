# XView Multi-Label Classification with DINOv3

Trains a DINOv3 vision transformer for multi-label object classification on XView satellite imagery using PyTorch DDP.

## File Structure

**`train_xview_classification_ddp.py`** - Main training script containing:

- **`XVIEW_TYPE_TO_CLASS`** - Mapping from XView type IDs (11-94) to sequential class IDs (0-59)
- **`XViewClassificationDataset`** - PyTorch dataset that streams tiled images directly from GCS
  - Lazy GCS client initialization per worker
  - Filters images with empty labels
  - Converts YOLO labels to multi-label binary vectors
- **`FocalLoss`** - Custom loss function for handling class imbalance (α=0.65, γ=2.0)
- **`DINOv3ViT`** - Vision transformer backbone from HuggingFace
  - Supports vitl16 (1024-dim) and vit7b16 (4096-dim) pretrained on satellite imagery
- **`ClassifierHead`** - 2-layer MLP (embed_dim → 512 → 60 classes)
- **`XViewClassificationTrainer`** - Main trainer class with:
  - `init_comms()` - Initialize PyTorch DDP
  - `init_model()` - Load backbone and classifier, optionally unfreeze last N layers
  - `load_data()` - Setup distributed data loaders with prefetching
  - `_compute_metrics()` - Calculate precision, recall, F1 (macro & weighted), show top/bottom 5 classes
  - `train_epoch()` / `validate_epoch()` - Training/validation loops with mixed precision
  - `save_checkpoint()` - Save model checkpoints
  - `run()` - Main training loop with cosine LR scheduler

## Quick Start
```bash
python train_xview_classification_ddp.py
```

## Architecture
- **Backbone:** DINOv3-ViT (vitl16 or vit7b16) pretrained on satellite imagery
- **Head:** 2-layer MLP classifier (1024→512→60 classes)
- **Training:** PyTorch DDP with mixed precision (AMP)

## Features
- **GCS Streaming:** No local disk required, streams images directly from Google Cloud Storage
- **Focal Loss:** Handles extreme class imbalance (α=0.65, γ=2.0)
- **Partial Fine-Tuning:** Optionally unfreeze last N transformer layers for domain adaptation
  - `unfreeze_last_n_layers=0`: Frozen backbone (fastest)
  - `unfreeze_last_n_layers=3`: Partial fine-tuning (recommended)
  - `unfreeze_last_n_layers=12+`: Full fine-tuning (slowest)
- **Distributed Training:** Multi-GPU via PyTorch DDP with DistributedSampler
- **Optimized Data Loading:** Persistent workers + prefetching for high throughput

## Metrics
Tracks per-class and aggregate metrics:
- Precision, Recall, F1 (macro & weighted)
- Accuracy
- Loss

Saves best model by weighted F1 score.
