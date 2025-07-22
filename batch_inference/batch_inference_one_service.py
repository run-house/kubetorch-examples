import time
from concurrent.futures import ThreadPoolExecutor

import kubetorch as kt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

# Basic toy GNN, you would import your own model below in actual use
class SampleGNN(nn.Module):
    def __init__(self, node_feat=9, edge_feat=4, hidden=256, layers=4, out=1):
        super().__init__()
        self.embed = nn.Linear(node_feat, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, out)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = F.relu(self.embed(x))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.out(global_mean_pool(x, batch))


##########
# Deploy the Inference as a Service
# We will define an image (here, using the public NVIDIA image + installs) that each replica should have, and simply decorate
# the class with the resources per replica and the scale we need. Here, we want concurrency of 1, since we want each replica of the
# service to run over a single file. Later on, we will make calls to the service in threads.
#
# You can run `kt deploy batch_inference_example.py` and it will stand up the service on Kubernetes as a properly autoscaling service.
##########

img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install(
    ["torch_geometric", "tqdm"]
)

# @kt.compute(cpus="12", image=img, name="GNN_Inference", concurrency = 1) # If you want to use CPU, not GPU
@kt.compute(
    gpus="1", cpus=8, memory="15Gi", image=img, name="GNN_Inference", concurrency=1
)
@kt.autoscale(
    initial_scale=1, min_scale=0, max_scale=5, target=1, target_utilization=100
)
class GNNInference:
    def __init__(self, compile_mode="default"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graphs = None
        self.loader = None
        self.results = None
        self.model = self.load_model(compile_mode=compile_mode)

    # Load your trained model here from disk/bucket
    def load_model(self, compile_mode):
        model = SampleGNN()
        return torch.compile(
            model.to(self.device).eval(), mode=compile_mode
        )  # compile for efficiency

    # Load each shard of your dataset from disk/bucket
    def load_dataset(self, gcs_path):
        pass

    # For our example, we just generate random graphs for us to use for inference for convenience.
    def load_dummy_dataset(
        self,
        num_obs,
        batch_size,
        num_data_workers,
        pin_memory=True,
        persistent_workers=True,
    ):
        sizes = np.random.randint(10, 31, num_obs)
        graphs = []

        for n in sizes:
            edge_count = n * 2
            graphs.append(
                Data(
                    x=torch.randn(n, 9),
                    edge_index=torch.randint(0, n, (2, edge_count)),
                    edge_attr=torch.randn(edge_count, 4),
                )
            )

        self.graphs = graphs
        self.loader = DataLoader(
            self.graphs,
            batch_size=batch_size,
            num_workers=num_data_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def predict(self):
        results = []

        for data in tqdm(self.loader):
            data = data.to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                pred = (
                    self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
            results.extend(pred)

        self.write_inference_results("fake_path/", results)

    def write_inference_results(self, gcs_path, results):
        pass

    def load_data_and_predict(
        self,
        filename,
        num_obs,
        batch_size,
        num_data_workers,
        pin_memory=True,
        persistent_workers=True,
    ):
        try:
            print("Loading Data")
            self.load_dummy_dataset(
                num_obs, batch_size, num_data_workers, pin_memory, persistent_workers
            )
            print("Starting Inference")
            self.predict()
            return {"file": filename, "status": "success"}

        except Exception as e:
            print(e)
            return {"file": filename, "status": "fail"}

    def benchmark(
        self,
        num_obs=10000,
        batch_size=1024,
        num_data_workers=4,
        pin_memory=True,
        persistent_workers=False,
    ):
        self.load_dummy_dataset(
            num_obs, batch_size, num_data_workers, pin_memory, persistent_workers
        )

        # Warmup
        print("warming up")
        for i, data in enumerate(self.loader):
            data = data.to(self.device)
            print(i)
            with torch.no_grad(), torch.cuda.amp.autocast():
                _ = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            if i == 5:
                break  # 3-5 batches is often enough

        # Time
        results = []
        start = time.time()
        for data in tqdm(self.loader):
            data = data.to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                pred = (
                    self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
            results.extend(pred)

        print(f"Ran {len(results) / (time.time() - start):.2f} inferences / second")


def inference_thread(filename):
    return GNNInference.load_data_and_predict(
        filename=filename, num_obs=500000, batch_size=2048, num_data_workers=6
    )


if __name__ == "__main__":
    # See how many tokens we can generate on the service

    GNNInference.benchmark(
        num_obs=100000, batch_size=1024, num_data_workers=6, persistent_workers=True
    )

    # How we would run in parallel by calling the remote service in parallel via threads
    files_list = [f"file{i}" for i in range(1, 100)]  # Your inference inputs
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = executor.map(inference_thread, files_list)

    print(list(results))
