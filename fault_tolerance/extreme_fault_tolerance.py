# TODO [SB]: add comment that explains the example.
# points:
#    * we train a basic Torch image classification model with the MNIST Dataset


# We use the very popular MNIST dataset, which includes a large number
# of handwritten digits, and create a neural network that accurately identifies
# what digit is in an image.
#
# ## Setting up a model class


import argparse
import math
import time
import io

import kubetorch as kt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime, timedelta


# Let's define a function that downloads the data. You can imagine this as a generic function to access data.
def download_data(path="./data"):
    datasets.MNIST(path, train=True, download=True)
    datasets.MNIST(path, train=False, download=True)
    print("Done with data download")


def preprocess_data(path):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (28, 28), interpolation=Image.BILINEAR
            ),  # Resize to 28x28 using bilinear interpolation
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize with mean=0.5, std=0.5 for general purposes
        ]
    )

    train = datasets.MNIST(path, train=False, download=False, transform=transform)
    test = datasets.MNIST(path, train=False, download=False, transform=transform)
    print("Done with data preprocessing")
    print(f"Number of training samples: {len(train)}")
    print(f"Number of test samples: {len(test)}")


# Next, we define a model class. We define a very basic feedforward neural network with three fully connected layers.
class TorchExampleBasic(nn.Module):
    def __init__(self):
        super(TorchExampleBasic, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# We also define a trainer class. The trainer class has methods to load data,
# train the model for one epoch, test the model on the test data, and then finally to save
# the model to S3. We'll also save snapshots of the training stages in S3, that will be easily accessed and loaded in
# case we would like to continue the model training after it was terminated for any reason (exceptions etc).
class SimpleTrainer:
    def __init__(self, model=None, bucket_name=None, checkpoint_path=None, region_name = "us-east-1"):

        super(SimpleTrainer, self).__init__()

        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=10))
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.rank = rank
        self.world_size = world_size
        print(f"Rank {rank} of {world_size} initialized")

        device_id = rank % torch.cuda.device_count()

        if model:
            model = model().to(device_id)
        else:
            model = TorchExampleBasic().to(device_id)

        ddp_model = DDP(model, device_ids=[device_id])

        self.device_id = device_id
        self.model = ddp_model
        self.device = torch.device(f"cuda:{device_id}")

        if bucket_name and checkpoint_path:
            self.load_from_checkpoint(bucket_name=bucket_name, checkpoint_path=checkpoint_path, region_name=region_name)

        self.model.to(self.device)

        self.epoch = 0

        self.train_loader = None
        self.test_loader = None

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.accuracy = None
        self.test_loss = None

    def load_train(self, path, batch_size, is_distributed=False, rank=0, world_size=1):
        rank = self.rank or rank
        world_size = self.world_size or world_size

        dataset = datasets.MNIST(
            path, train=True, download=True, transform=self.transform
        )

        if is_distributed:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            self.train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def load_test(self, path, batch_size):
        data = datasets.MNIST(
            path, train=False, download=False, transform=self.transform
        )
        self.test_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    def train_model(self, learning_rate=0.001):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return round(running_loss / 100, 3)

    def test_model(self):

        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_loss = test_loss
        self.accuracy = 100.0 * correct / len(self.test_loader.dataset)

        print(
            f"\nTest set: Average loss: {self.test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({self.accuracy:.2f}%)\n"
        )

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
        return pred.item()

    def save_model(self, bucket_name: str, s3_file_path: str, region_name: str = 'us-east-1'):
        # Avoid failing if you're just trying the example. Need to setup S3 access.
        try:
            if torch.distributed.get_rank() == 0:  # save only from rank 0
                import boto3

                buffer = io.BytesIO()
                torch.save(self.model.state_dict(), buffer)

                buffer.seek(0)  # Rewind the buffer to the beginning

                s3 = boto3.client("s3", region_name=region_name)
                s3.upload_fileobj(buffer, bucket_name, s3_file_path)
        except Exception as e:
            print(f"Failed to upload checkpoint: {str(e)}")

    def return_status(self):
        status = {
            "epochs_trained": self.epoch,
            "loss_test": self.test_loss,
            "accuracy_test": self.accuracy,
        }

        return status

    def load_from_checkpoint(self, bucket_name: str, checkpoint_path: str, region_name: str = 'us-east-1'):
        import boto3

        s3 = boto3.client("s3", region_name=region_name)
        obj = s3.get_object(Bucket=bucket_name, Key=checkpoint_path)
        buffer = io.BytesIO(obj["Body"].read())
        self.model.load_state_dict = torch.load(buffer, map_location=self.device)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training Example")
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    args = parser.parse_args()
    epochs, batch_size = args.epochs, args.batch_size

    s3_bucket_name = "sb-training-example"  # TODO: make it a user-provided argument as well.
    s3_path = f"checkpoints/model_checkpoint.pth"

    print(f"-------- Starting MINST Training remotely --------")
    print(f"-------- Creating a kubetorch service --------")
    gpus = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:24.01-py3").pip_install(reqs="boto3"),
        launch_timeout=600,
        inactivity_ttl="4h",
        gpu_type="NVIDIA-A10G",
    ).distribute("pytorch", workers=4)

    remote_cls_name = "train-cls-1"
    remote_cls = kt.cls(class_obj=SimpleTrainer, name=remote_cls_name).to(gpus)
    try:
        print(f"-------- Loading data remotely with batch size: {batch_size}--------")
        remote_cls.load_train("/data", batch_size, is_distributed=True)
        print(f"-------- Calling remote training with epochs: {epochs} --------")
        for epoch in range(epochs):
            loss = remote_cls.train_model()
            print(f'[Epoch {epoch}] Loss: {loss}')

    except Exception:
        remote_cls.save_model(
            bucket_name=s3_bucket_name,
            s3_file_path=s3_path,
        )

if __name__ == "__main__":
    main()
