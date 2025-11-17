"""VHR10 Sklearn Baseline with DINOv2 Features

This script demonstrates using the frozen DINOv2 backbone with sklearn classifiers.
It extracts embeddings once, then trains lightweight sklearn models (LogisticRegression, SVM, etc.)
"""

import argparse

import kubetorch as kt
import torch
from PIL import Image


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


class DINOv3ViT(torch.nn.Module):
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
        import os

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


class BaselineTrainer:
    """Trains sklearn classifiers on frozen DINOv2 features."""

    def __init__(self, data_root="./data", model_name="vitl16"):
        self.data_root = data_root
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = None
        self.processor = None
        self.train_loader = None
        self.val_loader = None

    def init_backbone(self):
        """Load the frozen DINOv2 backbone."""
        print(f"Loading DINOv2 backbone ({self.model_name})...")
        self.backbone = DINOv3ViT(model_name=self.model_name, pretrained=True)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        self.processor = self.backbone.processor

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        print(f"Backbone loaded: {self.backbone.embed_dim}-dim embeddings")

    def load_data(self):
        """Load VHR10 dataset - simple version without DataLoader."""
        from torch.utils.data import random_split
        from torchgeo.datasets import VHR10

        print(f"Loading VHR10 dataset from {self.data_root}")

        # Load VHR10 dataset - labels are 1-indexed (1-10) for 10 object classes
        full_dataset = VHR10(root=self.data_root, split="positive", download=True)

        # Split into train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Wrap datasets with processor
        self.train_dataset = VHR10ClassificationWrapper(
            self.train_dataset, self.processor
        )
        self.val_dataset = VHR10ClassificationWrapper(self.val_dataset, self.processor)

        print(
            f"Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples"
        )

    def extract_embeddings(self):
        """Extract embeddings from train and val sets."""
        import numpy as np

        def _extract(dataset, name):
            embeddings, labels = [], []
            with torch.no_grad():
                for idx in range(len(dataset)):
                    sample = dataset[idx]
                    image = sample["image"].unsqueeze(0).to(self.device)
                    embedding = self.backbone(image)
                    embeddings.append(embedding.cpu().numpy())
                    labels.append(sample["label"] - 1)  # Convert 1-10 to 0-9
            return np.vstack(embeddings), np.array(labels)

        print(
            f"\nExtracting embeddings from {len(self.train_dataset)} train, {len(self.val_dataset)} val samples..."
        )
        train_emb, train_labels = _extract(self.train_dataset, "train")
        val_emb, val_labels = _extract(self.val_dataset, "val")
        print(f"Done: {train_emb.shape}, {val_emb.shape}")

        return train_emb, train_labels, val_emb, val_labels

    def train_classifier(self, name, clf, train_emb, train_labels, val_emb, val_labels):
        """Train and evaluate any sklearn-compatible classifier."""
        from sklearn.metrics import accuracy_score

        clf.fit(train_emb, train_labels)
        train_acc = accuracy_score(train_labels, clf.predict(train_emb))
        val_acc = accuracy_score(val_labels, clf.predict(val_emb))

        print(f"{name:20s} - Train: {train_acc*100:.1f}%, Val: {val_acc*100:.1f}%")
        return {"train_acc": float(train_acc), "val_acc": float(val_acc)}

    def run(self):
        """Run the full baseline pipeline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from xgboost import XGBClassifier

        self.init_backbone()
        self.load_data()
        train_emb, train_labels, val_emb, val_labels = self.extract_embeddings()

        print("\nTraining classifiers on frozen DINOv2 features:")
        results = {
            "LogisticRegression": self.train_classifier(
                "LogisticRegression",
                LogisticRegression(max_iter=1000, random_state=42),
                train_emb,
                train_labels,
                val_emb,
                val_labels,
            ),
            "LinearSVM": self.train_classifier(
                "LinearSVM",
                LinearSVC(max_iter=1000, random_state=42),
                train_emb,
                train_labels,
                val_emb,
                val_labels,
            ),
            "XGBoost": self.train_classifier(
                "XGBoost",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    eval_metric="mlogloss",
                ),
                train_emb,
                train_labels,
                val_emb,
                val_labels,
            ),
        }

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, result in results.items():
            print(f"{name:20s} - Val: {result['val_acc']*100:.2f}%")
        print("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(description="VHR10 Sklearn Baseline with DINOv2")
    parser.add_argument(
        "--data-root", type=str, default="./data", help="data root directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vitl16",
        choices=["vitl16", "vit7b16"],
        help="DINOv2 model variant",
    )

    args = parser.parse_args()

    # Run locally or on remote compute
    img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install(
        [
            "torchgeo",
            "transformers",
            "scikit-learn",
            "xgboost",
            "pillow",
            "soxr",
        ]
    )

    gpu_compute = kt.Compute(
        gpus=1,
        image=img,
        secrets=["huggingface"],
        launch_timeout=600,
    )

    # Dispatch to remote
    remote_trainer = kt.cls(BaselineTrainer).to(
        gpu_compute,
        init_args={"data_root": args.data_root, "model_name": args.model_name},
    )

    # Run baseline
    results = remote_trainer.run()

    print("Baseline training complete!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
