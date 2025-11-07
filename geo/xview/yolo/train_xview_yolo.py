"""YOLOv8 training on XView from GCS"""

import os
import yaml
from pathlib import Path
import kubetorch as kt

# XView type_id to class_id (11-94 -> 0-59)
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


class XViewYOLOTrainer:
    """YOLOv8 trainer for XView from GCS."""

    def __init__(self, gcs_bucket, gcs_prefix, creds_path):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.creds_path = creds_path
        self.model = None
        self.data_dir = Path("./xview_local")
        self.dataset_yaml = None

    def download_from_gcs(self):
        """Download dataset from GCS with parallel downloads."""
        import sys
        for mod in list(sys.modules.keys()):
            if mod.startswith('google'):
                del sys.modules[mod]

        import importlib
        import google.cloud.storage
        importlib.reload(google.cloud.storage)

        from google.cloud import storage
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        if (self.data_dir / "train").exists():
            print("Dataset exists, skipping download")
            return

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.creds_path
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)

        prefix = f"{self.gcs_prefix}/train/"
        files = [(blob, self.data_dir / "train" / blob.name[len(prefix):])
                 for blob in bucket.list_blobs(prefix=prefix) if blob.name[len(prefix):]]

        print(f"Downloading {len(files)} files...")

        def download(file_info):
            blob, path = file_info
            path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(path))

        with ThreadPoolExecutor(max_workers=24) as executor:
            list(tqdm(executor.map(download, files), total=len(files)))

        print(f"Downloaded to {self.data_dir}")
        self._remap_labels()

    def prepare_dataset(self):
        """Download and split dataset."""
        self.download_from_gcs()
        self._split_train_val()

    def _remap_labels(self):
        """Remap type_id (11-94) to class_id (0-59)."""
        labels_dir = self.data_dir / "train" / "labels"
        if not labels_dir.exists():
            return

        for label_file in labels_dir.glob("*.txt"):
            remapped = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        type_id = int(parts[0])
                        if type_id in XVIEW_TYPE_TO_CLASS:
                            class_id = XVIEW_TYPE_TO_CLASS[type_id]
                            remapped.append(f"{class_id} {' '.join(parts[1:])}\n")

            with open(label_file, 'w') as f:
                f.writelines(remapped)

    def _split_train_val(self, val_split=0.1):
        """Split into train/val (10% validation)."""
        import shutil
        import random

        val_dir = self.data_dir / "val"
        if (val_dir / "images").exists():
            print("Val split exists, skipping")
            return

        images = list((self.data_dir / "train" / "images").glob("*.png"))
        random.seed(42)
        random.shuffle(images)
        val_images = images[:int(len(images) * val_split)]

        (val_dir / "images").mkdir(parents=True)
        (val_dir / "labels").mkdir(parents=True)

        for img in val_images:
            shutil.move(str(img), str(val_dir / "images" / img.name))
            lbl = self.data_dir / "train" / "labels" / img.with_suffix('.txt').name
            if lbl.exists():
                shutil.move(str(lbl), str(val_dir / "labels" / lbl.name))

        print(f"Split: {len(images) - len(val_images)} train, {len(val_images)} val")

    def create_dataset_yaml(self):
        """Create YOLO dataset config."""
        class_names = [
            "Fixed-wing Aircraft", "Small Aircraft", "Cargo Plane", "Helicopter",
            "Passenger Vehicle", "Small Car", "Bus", "Pickup Truck", "Utility Truck",
            "Truck", "Cargo Truck", "Truck w/Box", "Truck Tractor", "Trailer",
            "Truck w/Flatbed", "Truck w/Liquid", "Crane Truck", "Railway Vehicle",
            "Passenger Car", "Cargo Car", "Flat Car", "Tank car", "Locomotive",
            "Maritime Vessel", "Motorboat", "Sailboat", "Tugboat", "Barge",
            "Fishing Vessel", "Ferry", "Yacht", "Container Ship", "Oil Tanker",
            "Engineering Vehicle", "Tower crane", "Container Crane", "Reach Stacker",
            "Straddle Carrier", "Mobile Crane", "Dump Truck", "Haul Truck",
            "Scraper/Tractor", "Front loader/Bulldozer", "Excavator", "Cement Mixer",
            "Ground Grader", "Hut/Tent", "Shed", "Building", "Aircraft Hangar",
            "Damaged Building", "Facility", "Construction Site", "Vehicle Lot",
            "Helipad", "Storage Tank", "Shipping container lot", "Shipping Container",
            "Pylon", "Tower"
        ]

        config = {
            "path": str(self.data_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(class_names),
            "names": class_names,
        }

        self.dataset_yaml = str(self.data_dir / "xview.yaml")
        with open(self.dataset_yaml, "w") as f:
            yaml.dump(config, f)

        print(f"Config saved to {self.dataset_yaml}")

    def train_model(self, model_size="n", epochs=100, imgsz=768, batch_size=16):
        """Train YOLOv8 (n/s/m/l/x)."""
        from ultralytics import YOLO
        import torch

        if self.dataset_yaml is None:
            raise ValueError("Call create_dataset_yaml() first")

        print(f"Loading YOLOv8{model_size}...")
        self.model = YOLO(f"yolov8{model_size}.pt")

        print("Training...")
        self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name="xview_training",
            patience=10,
            save=True,
            plots=True,
            verbose=True,
            device=list(range(torch.cuda.device_count())),
        )

        metrics_result = self.model.val()
        metrics = metrics_result.results_dict if hasattr(metrics_result, "results_dict") else {}

        if metrics:
            def get_m(k): return metrics.get(f"metrics/{k}(B)", metrics.get(f"metrics/{k}", 0))
            m50, m50_95, prec, rec = get_m("mAP50"), get_m("mAP50-95"), get_m("precision"), get_m("recall")
            if m50 or m50_95:
                print(f"mAP50: {m50:.4f}, mAP50-95: {m50_95:.4f}, P: {prec:.4f}, R: {rec:.4f}")

        return metrics if metrics else None

    def show_predictions(self, num_samples=5):
        """Run inference on sample images."""
        if self.model is None:
            raise ValueError("Call train_model() first")

        val_images = list((self.data_dir / "train" / "images").glob("*.png"))[:num_samples]
        if not val_images:
            raise ValueError("No images found")

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
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_summary = ", ".join([f"{n}: {c}" for n, c in sorted_counts])
        print(f"\n{total} total ({avg:.1f} avg) - {class_summary}")

        return {"total": total, "num_images": len(val_images), "avg": avg, "class_counts": class_counts}


if __name__ == "__main__":
    gcs_bucket = "rh_demo_image_data"
    gcs_prefix = "xview_tiled"
    local_creds_path = "/Users/paulyang/Downloads/runhouse-test-8ad14c2b4edf.json"

    img = (kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3")
           .pip_install(["ultralytics", "PyYAML", "tqdm"])
           .run_bash("uv pip install --system google-cloud-storage google-auth google-api-core")
           .run_bash("pip uninstall opencv-python opencv opencv-python-headless -y")
           .run_bash("pip install opencv-python-headless==4.5.5.64 --break-system-packages")
           .run_bash('pip install "numpy<2.0"'))

    compute = kt.Compute(
        gpus="4",
        image=img,
        secrets=[kt.Secret(name="gcp-dataaccess", path=local_creds_path)],
        launch_timeout=1200,
    )

    remote_trainer = kt.cls(XViewYOLOTrainer).to(compute, init_args={
        "gcs_bucket": gcs_bucket,
        "gcs_prefix": gcs_prefix,
        "creds_path": local_creds_path,
    })

    remote_trainer.prepare_dataset()
    remote_trainer.create_dataset_yaml()

    metrics = remote_trainer.train_model(model_size="m", epochs=50, imgsz=768, batch_size=16)
    pred_summary = remote_trainer.show_predictions(num_samples=10)

    print("\nDone!", metrics)
