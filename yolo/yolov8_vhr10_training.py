# # YOLOv8 Object Detection with VHR-10 Dataset
# In this example, we define a `YOLOTrainer` class that encapsulates YOLOv8 training using
# the VHR-10 dataset from TorchGeo. The VHR-10 dataset contains very high resolution
# satellite images with 10 object classes for detection tasks.
# We dispatch this class to remote compute with GPU and train the model.

from pathlib import Path

import kubetorch as kt


# ## YOLO Trainer Class
# We will send this training class to a remote instance with a GPU with Kubetorch
class YOLOTrainer:
    def __init__(self):
        self.model = None
        self.data_dir = Path("./vhr10_data")
        self.dataset_yaml = None

    def prepare_dataset(self, train_split=0.8):
        """Download and prepare VHR-10 dataset in YOLO format"""
        import yaml
        from torchgeo.datasets import VHR10

        print("Downloading and preparing VHR-10 dataset...")

        # Create directory structure for YOLO format
        for split in ["train", "val"]:
            (self.data_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Load VHR-10 dataset from TorchGeo
        # VHR-10 has 'positive' (with objects) and 'negative' (without objects) splits
        # We use the positive split and manually split it into train/val
        positive_dataset = VHR10(root=str(self.data_dir / "raw"), split="positive", download=True)

        # VHR-10 class names (10 classes)
        class_names = [
            "airplane",
            "ship",
            "storage_tank",
            "baseball_diamond",
            "tennis_court",
            "basketball_court",
            "ground_track_field",
            "harbor",
            "bridge",
            "vehicle",
        ]

        total_samples = len(positive_dataset)
        train_size = int(train_split * total_samples)

        print(f"Total samples: {total_samples}")
        print(f"Train samples: {train_size}, Val samples: {total_samples - train_size}")

        self._process_split_subset(positive_dataset, "train", 0, train_size)
        self._process_split_subset(positive_dataset, "val", train_size, total_samples)

        # Create dataset YAML file for YOLO
        # VHR10 uses 1-indexed labels (1-10), so nc=11 (class 0 unused)
        dataset_config = {
            "path": str(self.data_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": 11,  # VHR10 labels are 1-10, so need 11 classes (0-10, where 0 is unused)
            "names": ["unused"] + class_names,  # Pad with "unused" at index 0
        }

        self.dataset_yaml = str(self.data_dir / "vhr10.yaml")
        with open(self.dataset_yaml, "w") as f:
            yaml.dump(dataset_config, f)

        print(f"Dataset prepared successfully at {self.data_dir}")
        print(f"Dataset config saved to {self.dataset_yaml}")

    def _process_split_subset(self, dataset, split_name, start_idx, end_idx):
        """Convert TorchGeo dataset subset to YOLO format"""
        import numpy as np
        from PIL import Image

        for idx in range(start_idx, end_idx):
            # Use a different index for saving files (0-based for each split)
            save_idx = idx - start_idx
            sample = dataset[idx]
            image = sample["image"]  # CxHxW tensor
            boxes = sample["boxes"]  # Nx4 tensor with xyxy format
            labels = sample["labels"]  # N tensor with class indices

            # Convert image tensor to PIL Image and save
            # Normalize to [0, 255] range if needed
            img_tensor = image.cpu()
            if img_tensor.max() <= 1.0:
                img_tensor = img_tensor * 255
            img_tensor = img_tensor.clamp(0, 255)

            # Convert to RGB format (YOLO expects RGB images)
            if img_tensor.shape[0] == 3:  # Already RGB
                img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            elif img_tensor.shape[0] == 1:  # Grayscale - convert to RGB
                img_np = img_tensor[0].numpy().astype(np.uint8)
                img_np = np.stack([img_np, img_np, img_np], axis=-1)
            else:  # Multi-channel but not RGB - take first 3 channels
                img_np = img_tensor[:3].permute(1, 2, 0).numpy().astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            img_w, img_h = img_pil.size  # PIL.Image.size returns (width, height)
            img_path = self.data_dir / split_name / "images" / f"{save_idx:06d}.jpg"
            img_pil.save(img_path)

            # Convert boxes to YOLO format (normalized xywh)
            label_path = self.data_dir / split_name / "labels" / f"{save_idx:06d}.txt"
            with open(label_path, "w") as f:
                for box, label in zip(boxes, labels):
                    # Convert from xyxy to xywh
                    x1, y1, x2, y2 = box.tolist()
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    # YOLO format: class_id x_center y_center width height (normalized)
                    f.write(f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def train_model(self, model_size="n", epochs=100, imgsz=640, batch_size=-1):
        """Train YOLOv8 model on VHR-10 dataset
        Args:
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            epochs: Number of training epochs
            imgsz: Input image size
        """
        import torch
        from ultralytics import YOLO

        if self.dataset_yaml is None:
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")

        # Load YOLOv8 model (n=nano, s=small, m=medium, l=large, x=xlarge)
        self.model = YOLO(f"yolov8{model_size}.pt")

        # Train the model
        self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name="vhr10_training",
            patience=10,
            save=True,
            plots=True,
            verbose=True,
            device=list(range(torch.cuda.device_count())),
        )  # Results are all side effects ond isk

        print("\nTraining completed!")

        # Run validation to get metrics in memory
        # Ultralytics YOLO saves metrics to disk, but we can get them via model.val()
        metrics_result = self.model.val()
        metrics = metrics_result.results_dict if hasattr(metrics_result, "results_dict") else {}

        if metrics:
            # Extract common metrics (keys may vary by YOLO version)
            get_m = lambda k: metrics.get(f"metrics/{k}(B)", metrics.get(f"metrics/{k}", 0))
            m50, m50_95, prec, rec = (
                get_m("mAP50"),
                get_m("mAP50-95"),
                get_m("precision"),
                get_m("recall"),
            )
            if m50 or m50_95:
                print(f"  mAP50: {m50:.4f}, mAP50-95: {m50_95:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
            else:
                print("  (Metrics saved to runs/detect/vhr10_training/results.json)")
        else:
            print("  (Metrics saved to runs/detect/vhr10_training/results.json)")

        return metrics if metrics else None

    def show_predictions(self, num_samples=5):
        """Show model predictions on sample validation images"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        val_images = list((self.data_dir / "val" / "images").glob("*.jpg"))[:num_samples]
        if not val_images:
            raise ValueError("No validation images found.")

        class_counts = {}
        total = 0

        print(f"\nPredictions on {len(val_images)} samples:")
        for img_path in val_images:
            results = self.model(str(img_path), verbose=False)
            for r in results:
                dets = [(r.names[int(b.cls[0])], float(b.conf[0])) for b in r.boxes]
                total += len(dets)
                for cls_name, _ in dets:
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                print(f"{img_path.name}: {', '.join([f'{c}({conf:.2f})' for c, conf in dets]) if dets else 'none'}")

        avg = total / len(val_images) if val_images else 0
        print(
            f"\n{total} total ({avg:.1f} avg) - "
            + ", ".join([f"{n}: {c}" for n, c in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)])
        )

        return {
            "total_detections": total,
            "num_images": len(val_images),
            "avg_per_image": avg,
            "class_counts": class_counts,
        }

    def get_best_model_path(self):
        """Get the path to the best model from training"""
        # YOLOv8 saves best model in runs/detect/vhr10_training/weights/best.pt
        best_model = Path("runs/detect/vhr10_training/weights/best.pt")
        if best_model.exists():
            return str(best_model)
        return None

    def save_best_model(self, destination="vhr10_best.pt"):
        # best_model_path = self.get_best_model_path()
        # Implement your own logic to get to blob storage
        return None

    def export_model(self, format="onnx"):
        """Export model to different formats"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print(f"Exporting model to {format} format...")
        self.model.export(format=format)


# ## Launch Compute and Execute Training
#
# Now, we define the main function that will run locally when we run this script and set up
# our Kubetorch module on a remote cluster. First, we create a cluster with the desired instance type and provider.
if __name__ == "__main__":
    img = (
        kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3")
        .pip_install(
            [
                "ultralytics",
                "PyYAML",
                "pillow",
                "torchgeo",
            ]
        )
        .run_bash("pip uninstall opencv-python opencv opencv-python-headless -y")  # Remove existing
        .run_bash("pip install opencv-python-headless==4.5.5.64 --break-system-packages")  # Pin this version
        .run_bash('pip install "numpy<2.0"')  # Set numpy to be compatible with yolo
    )

    # Configure cluster with GPUs
    # The GPUs will be automatically visible via CUDA_VISIBLE_DEVICES
    # Change gpus="4" to "1", "8", etc. based on your needs
    cluster = kt.Compute(
        gpus="4",  # All 4 GPUs will be used for training automatically
        image=img,
        launch_timeout=1200,
    )

    # Send training class to remote cluster
    remote_trainer = kt.cls(YOLOTrainer).to(cluster)

    # Prepare dataset (downloads VHR-10 and converts to YOLO format)
    remote_trainer.prepare_dataset()

    # Train model (automatically validates each epoch and saves best checkpoint)
    # Options: n=nano, s=small, m=medium, l=large, x=xlarge
    metrics = remote_trainer.train_model(
        model_size="m",
        epochs=50,
        imgsz=640,
        batch_size=16,
    )

    # Show predictions on sample images (optional qualitative check)
    # Prints detections and returns summary dict
    pred_summary = remote_trainer.show_predictions(num_samples=10)

    # Save best model
    model_path = remote_trainer.save_best_model("vhr10_yolov8_best.pt")

    print("\nDone!", metrics)
