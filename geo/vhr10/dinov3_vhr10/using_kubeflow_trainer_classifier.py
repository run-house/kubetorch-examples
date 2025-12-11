import argparse
import os
import time

import kubetorch as kt

from vhr10_dinov3_classifier import VHR10Trainer

container = {
    "name": "pytorch-container",
    "image": "pytorch/pytorch:latest",
    "resources": {
        "requests": {
            "cpu": "0.5",
            "memory": "1Gi",
            "nvidia.com/gpu": 1,
        },
        "limits": {
            "nvidia.com/gpu": 1,
        },
    },
}

PYTORCHJOB_MANIFEST = {
    "apiVersion": "kubeflow.org/v1",
    "kind": "PyTorchJob",
    "metadata": {
        "name": "",
        "namespace": "default",
    },
    "spec": {
        "pytorchReplicaSpecs": {
            "Master": {
                "replicas": 1,
                "restartPolicy": "OnFailure",
                "template": {"spec": {"containers": [container]}},
            },
            "Worker": {
                "replicas": 2,
                "restartPolicy": "OnFailure",
                "template": {"spec": {"containers": [container]}},
            },
        },
    },
}


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="VHR10 Classification with DINOv3")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
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
        "--workers", type=int, default=3, help="number of distributed workers"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="number of classes (VHR10 uses labels 1-10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="probability to use for binary classification",
    )

    args = parser.parse_args()

    # Set worker replicas (total = 1 Master + N-1 Workers)
    PYTORCHJOB_MANIFEST["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] = (
        args.workers - 1
    )

    # Define image with dependencies
    img = (
        kt.Image(image_id="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime")
        .pip_install(
            [
                "torchgeo[datasets,models]",
                "transformers",
                "torchvision",
                "pillow",
                "soxr",  # Required by transformers for audio_utils
            ]
        )
        .set_env_vars({"HF_TOKEN": os.environ["HF_TOKEN"]})
    )

    # Create compute from PyTorchJob manifest
    gpu_compute = kt.Compute.from_manifest(PYTORCHJOB_MANIFEST)
    gpu_compute.gpus = 1
    gpu_compute.image = img
    gpu_compute.launch_timeout = 600
    gpu_compute.inactivity_ttl = "2h"
    gpu_compute.distributed_config = {"quorum_workers": args.workers}

    # Initialize trainer arguments
    init_args = dict(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Dispatch trainer class to remote GPUs
    remote_trainer = kt.cls(VHR10Trainer).to(
        gpu_compute, init_args=init_args, stream_logs=True
    )
    print("Time to first activity:", time.time() - start_time)

    # Run distributed training
    remote_trainer.setup(
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    print("Time to setup:", time.time() - start_time)

    data_start = time.time()
    remote_trainer.load_data(args.batch_size)
    print("Time to load data:", time.time() - data_start)
    print("Time to start training:", time.time() - start_time)  # 19 seconds after warm

    remote_trainer.train(num_epochs=args.epochs, threshold=args.threshold)
    print(
        "Training complete, total time:", time.time() - start_time
    )  # 160 s after first run


if __name__ == "__main__":
    main()
