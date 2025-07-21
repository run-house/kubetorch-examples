import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from kubetorch_example.model import SimpleNN
from kubetorch_example.transforms import get_transform

from torch.utils.data import DataLoader
from torchvision import datasets


class SimpleTrainer:
    def __init__(self, from_checkpoint=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN().to(self.device)
        if from_checkpoint:
            self.model.load_state_dict(
                torch.load(from_checkpoint, map_location=self.device)
            )

        self.train_loader = None
        self.test_loader = None
        self.epoch = 0
        self.transform = get_transform()

    def load_data(self, path, batch_size, download=True):
        def mnist(is_train):
            return datasets.MNIST(
                path, train=is_train, download=download, transform=self.transform
            )

        self.train_loader = DataLoader(mnist(True), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(mnist(False), batch_size=batch_size)

    def train_model(self, learning_rate=0.001):
        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        running_loss = 0.0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{self.epoch + 1}, loss: {running_loss / 100:.3f}")

    def test_model(self):
        self.model.eval()
        total_loss, correct = 0, 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += F.cross_entropy(output, target, reduction="sum").item()
                correct += (output.argmax(1) == target).sum().item()

        n = len(self.test_loader.dataset)
        print(
            f"Test loss: {total_loss/n:.4f}, Accuracy: {correct}/{n} ({100. * correct/n:.2f}%)"
        )

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x.to(self.device))

        return output.argmax(1).item()

    def save_model(self, bucket, s3_path):
        try:
            import boto3

            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
            boto3.client("s3").upload_fileobj(buf.seek(0) or buf, bucket, s3_path)
            print("Uploaded checkpoint")
        except Exception:
            print("Did not upload checkpoint, might not be authorized")
