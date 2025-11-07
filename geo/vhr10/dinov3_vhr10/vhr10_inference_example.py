"""Example deployment patterns for VHR10 DINOv3 classifier

This file demonstrates how to use the refactored backbone and classifier
separately for inference, making it easy to:
1. Deploy backbone as a persistent embedding service
2. Run lightweight classifier locally or as separate service
3. Cache embeddings for multiple downstream tasks
4. Use different classifiers on the same features

Pattern 1: Separate backbone (remote) and classifier (local)
- Deploy DINOv3 backbone as a persistent embedding service
- Run lightweight classifier locally or as separate service
- Best for: Multiple classifiers sharing same backbone

Pattern 2: Local inference with pre-computed embeddings
- Extract embeddings once, cache them
- Run multiple classifiers on cached embeddings
- Best for: Batch processing, experimentation
"""

import kubetorch as kt
import torch
from PIL import Image

# Import the refactored classes from the main module
from vhr10_dinov3_classifier import ClassifierHead, DINOv3ViT


# ============================================================================
# Pattern 1: Separate Backbone Service + Local Classifier
# ============================================================================
class DINOv3EmbeddingService:
    """Persistent embedding service for DINOv3 backbone.

    Deploy this on GPU with KT, keeps model loaded for fast inference.
    Multiple classifiers / users can query this by name for embeddings.

    This now uses the refactored DINOv3ViT class which is fully independent.
    """

    def __init__(self, model_name="vitl16"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = DINOv3ViT(model_name=model_name, pretrained=True).to(self.device)
        self.backbone.eval()
        self.processor = self.backbone.processor

        # Freeze all parameters for inference-only mode
        for param in self.backbone.parameters():
            param.requires_grad = False

    def embed(self, images):
        """Extract embeddings for images using the backbone's inference method.

        Args:
            images: PIL Image, list of PIL Images, or preprocessed tensor

        Returns:
            Embeddings tensor of shape [batch, embed_dim]
        """
        # Use the backbone's built-in extract_features method
        return self.backbone.extract_features(images, device=self.device)

    def embed_batch(self, image_paths, batch_size=32):
        """Batch process multiple images efficiently."""
        from torch.utils.data import DataLoader, Dataset

        class ImageDataset(Dataset):
            def __init__(self, paths, processor):
                self.paths = paths
                self.processor = processor

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        dataset = ImageDataset(image_paths, self.processor)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

        all_embeddings = []
        with torch.no_grad():
            for batch in loader:
                embeddings = self.backbone(batch.to(self.device))
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


# ============================================================================
# Usage Example
# ============================================================================


def deploy_separate_services():
    """Example 1: Deploy backbone as service, classifier runs locally."""

    # Step 1: Deploy embedding service on GPU with Kubetorch with autoscaling
    gpu_compute = kt.Compute(
        gpus=1,
        image=kt.Image("nvcr.io/nvidia/pytorch:23.10-py3").pip_install(["transformers", "pillow", "soxr"]),
    ).autoscale(min_scale=0, max_scale=10, concurrency=50, metric="concurrency")

    embedding_service = kt.cls(DINOv3EmbeddingService).to(gpu_compute, init_args={"model_name": "vitl16"})

    # Step 2: Load lightweight classifier locally using ClassifierHead
    checkpoint_path = "./checkpoints/vhr10_dinov3_vitl16_best.pth"
    classifier = ClassifierHead.load_from_checkpoint(checkpoint_path, device="cpu")

    # Step 3: Inference pipeline
    def predict_image(image_path):
        img = Image.open(image_path).convert("RGB")
        embeddings = embedding_service.embed(img)  # Remote call

        classifier.eval()
        with torch.no_grad():
            logits = classifier(embeddings)
            pred = torch.argmax(logits, dim=1)
            prob = torch.softmax(logits, dim=1)

        return pred.item(), prob[0, pred].item()

    # Use it
    pred, confidence = predict_image("image.jpg")
    print(f"Predicted class: {pred}, confidence: {confidence:.2%}")

    # Batch processing
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    embeddings = embedding_service.embed_batch(image_paths, batch_size=32)

    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings.cpu())
        predictions = torch.argmax(logits, dim=1)

    print(f"Batch predictions: {predictions}")
