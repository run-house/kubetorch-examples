# XView Object Detection with YOLOv8

Trains YOLOv8 for object detection on XView satellite imagery with 60 object classes.

## Quick Start
```bash
python train_xview_yolo.py
```

## Features
- **GCS Download:** Parallel download from Google Cloud Storage
- **Auto Label Remapping:** Converts XView type_id (11-94) to class_id (0-59)
- **Auto Train/Val Split:** 90/10 split with random seed
- **Multi-GPU Training:** Automatic detection and use of all GPUs
- **Model Sizes:** n/s/m/l/x (nano to xlarge)

## Workflow
1. Downloads tiled dataset from GCS
2. Remaps labels to sequential class IDs
3. Splits into train/val (90/10)
4. Creates YOLO dataset config
5. Trains YOLOv8 with early stopping
6. Runs validation and shows sample predictions

## Metrics
- mAP50, mAP50-95
- Precision, Recall
- Per-class detection counts
