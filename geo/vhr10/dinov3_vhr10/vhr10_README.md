# VHR10 DINOv2 Satellite Image Classification

Train a classifier on VHR10 satellite imagery using DINOv2 models pretrained on SAT-493M.

## Architecture

**DINOv2ViT** - Standalone backbone for feature extraction
**ClassifierHead** - Lightweight head that operates on embeddings (choose your own adventure, could even be classical ML)
**VHR10Classifier** - Wrapper combining backbone + head for training

## Training Details

- **Dataset**: VHR10 object detection dataset adapted for classification, replace with yours
- **Accuracy**: Prediction correct if it matches ANY label in the image
- **Backbone**: Frozen during training (feature extractor only)

## Program Flow
1. `VHR10Trainer.init_comms()` - Initialize distributed training
2. `VHR10Trainer.init_model()` - Load frozen DINOv2 backbone + trainable classifier head
3. `VHR10Trainer.load_data()` - Load VHR10 dataset with preprocessing
4. `VHR10Trainer.train_epoch()` - **Backbone extracts features → Classifier predicts → Backprop through classifier only**
5. `VHR10Trainer.validate_epoch()` - Same pipeline without gradients
6. `VHR10Trainer.save_checkpoint()` - Save only classifier weights (~2KB), backbone loads from HuggingFace

## Usage

```bash
python vhr10_dinov3_classifier.py --epochs 20 --batch-size 32 --model-name vitl16 --workers 2
```

## Inference

The architecture allows separate backbone and classifier usage.
See `inference_example.py` for deployment patterns.
