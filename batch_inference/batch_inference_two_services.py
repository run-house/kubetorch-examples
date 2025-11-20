import time

import kubetorch as kt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

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


# You would implement loading your dataset here
def prepare_dataset(num_obs=50):
    sizes = np.random.randint(10, 31, num_obs)
    graphs = []

    for n in sizes:
        graph_dict = {
            "x": torch.randn(n, 9).tolist(),
            "edge_index": torch.randint(0, n, (2, n * 2)).tolist(),
            "edge_attr": torch.randn(n * 2, 4).tolist(),
            "num_nodes": int(n),
        }
        graphs.append(graph_dict)

    return graphs


# Deploy the Inference as a service
img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").pip_install(["torch_geometric"])


@kt.compute(cpus="1", image=img, name="GNN_Inference")
@kt.autoscale(initial_replicas=1, min_replicas=0, max_replicas=10, concurrency=10)
class GNNInference:
    def __init__(self, mode="default"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SampleGNN()  # load your model here
        self.model = torch.compile(model.to(self.device).eval(), mode=mode)

    def preprocess_data(self, batch):
        data_objects = []

        for graph_dict in batch:
            data = Data(
                x=torch.tensor(graph_dict["x"], dtype=torch.float32),
                edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
                edge_attr=torch.tensor(graph_dict["edge_attr"], dtype=torch.float32),
            )
            data_objects.append(data)

        return Batch.from_data_list(data_objects)

    def predict(self, data):
        print(time.time(), "preprocess")
        data = self.preprocess_data(data)
        print(time.time(), "inference")
        with torch.no_grad():
            results = self.model(data.x, data.edge_index, data.edge_attr, data.batch).detach().cpu().numpy().tolist()
            print(time.time(), "done inference")
            return results


# Deploy a service that calls Inference in a separately autoscaling service; this way, you
# can fully saturate your inference services by correctly scaling the number of I/O-bound
# services (reading data, writing data) and the number of inference services.
def run_shard(batch_size=4, num_batches=10000):
    graphs = prepare_dataset(num_batches * batch_size)  # would pass dataset id / path

    results = []
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    from tqdm import tqdm

    # Create a partial function for embedding with logging disabled
    inference_thread = partial(GNNInference.predict, stream_logs=False)

    # Run concurrent embedding requests to measure throughput
    batches_list = [graphs[i : i + batch_size] for i in range(0, len(graphs), batch_size)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        start = time.time()
        results = list(
            tqdm(
                executor.map(inference_thread, batches_list),
                total=len(batches_list),
                desc="Processing inference",
            )
        )
    print(f"Processed {(len(batches_list) * len(batches_list[0])) / (time.time() - start):.2f} inferences / second")

    # for i in range(0, len(graphs), batch_size):
    #     batch_graphs = graphs[i:i+batch_size]

    #     pred = GNNInference.predict(batch_graphs, stream_logs= False)
    #     results.append(pred)

    #     if i % 1000 == 0:  # Progress tracking
    #         print(f"Processed {i // batch_size:,} batches")

    return results  # Concatenate all results


if __name__ == "__main__":
    # inference = GNNInference(model)
    # dummy_batch = Batch.from_data_list([Data(x=torch.randn(20, 9), edge_index=torch.randint(0, 20, (2, 40)))])
    predictions = run_shard(batch_size=128, num_batches=100)
    print(f"Generated {len(predictions):,} predictions")
