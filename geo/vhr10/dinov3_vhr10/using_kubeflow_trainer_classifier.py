# # Kubeflow Trainer + Kubetorch
# In this example, we show how to use the Kubetorch `from_manifest()` to
# launch TrainJob / PyTorchJob CRD based distributed training workloads.
# You can interactively override the base manifest with Kubetorch as well
# for easy iteration over a base configuration.
#
# With Kubetorch, you get a significantly better interface into compute
# and development experience.
# * Instantly redeploy local code changes, without having to rebuild Docker images
# * Persisting the environment across iteration loops, without having to reload data
# or artifacts or re-pip install any libraries.
# * No need to requeue for resources
# * Directly run inference and evaluations after model completes on the same service
# without having to deploy it separately
# * Allow for multi-threaded parallel calls into the same deployed training class

import argparse
import os
import time

import kubetorch as kt

# We import the underlying trainer class, which is a regular Python class with methods
# like train(), load_data(), predict(), etc.
from vhr10_dinov3_classifier import VHR10Trainer

# A toy example of a PyTorchJob + TrainJob manifest, hardcoded for convenience
container = {
    "name": "kubetorch",
    "image": "pytorch/pytorch:latest",
    "resources": {
        "requests": {
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


TRAINJOB_MANIFEST = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainJob",
    "metadata": {
        "name": "",
        "namespace": "default",
    },
    "spec": {
        "runtimeRef": {"name": "torch-distributed"},
        "trainer": {"numNodes": 3},
        "template": {"spec": {"containers": [container]}},
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

    # Override worker replicas based on passed args
    PYTORCHJOB_MANIFEST["spec"]["pytorchReplicaSpecs"]["Worker"]["replicas"] = (
        args.workers - 1
    )
    TRAINJOB_MANIFEST["spec"]["trainer"]["numNodes"] = args.workers

    # Define image with dependencies, which overrides the base container
    img = (
        kt.Image(image_id="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime")
        .pip_install(
            [
                "torchgeo[datasets,models]",
                "transformers",
                "torchvision",
                "pillow",
                "soxr",
            ]
        )
        .set_env_vars({"HF_TOKEN": os.environ["HF_TOKEN"]})
    )

    # Create compute from PyTorchJob manifest, and then update with fields for Kubetorch
    gpu_compute = kt.Compute.from_manifest(TRAINJOB_MANIFEST)
    gpu_compute.gpus = 1
    gpu_compute.image = img
    gpu_compute.launch_timeout = 600
    gpu_compute.inactivity_ttl = "2h"
    gpu_compute.distributed_config = {"quorum_workers": args.workers}

    # Dispatch trainer class to remote compute, launching as TrainJob/PTJ
    init_args = dict(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
    )

    remote_trainer = kt.cls(VHR10Trainer).to(gpu_compute, init_args=init_args)
    print("Time to first activity:", time.time() - start_time)

    # Run distributed training, calling methods on remote
    remote_trainer.setup(
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    print("Time to setup:", time.time() - start_time)

    remote_trainer.load_data(args.batch_size, num_workers=0)
    print(
        "Time to start training:", time.time() - start_time
    )  # 19 seconds after up, 50 seconds from warm node

    remote_trainer.train(num_epochs=args.epochs, threshold=args.threshold)
    print("Training complete, total time:", time.time() - start_time)


if __name__ == "__main__":
    main()
