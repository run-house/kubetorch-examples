# # PyTorch Multi-node Distributed Training
# A basic example showing how to Pythonically run a PyTorch distributed training script on a
# cluster of GPUs. Often distributed training is launched from multiple parallel CLI commands
# (`python -m torch.distributed.launch ...`), each spawning separate training processes (ranks).
# Here, we're creating each process as a separate worker on our compute, sending our training function
# into each worker, and calling the replicas concurrently to trigger coordinated multi-node training
# (`torch.distributed.init_process_group` causes each to wait for all to connect, and sets up the distributed
# communication). We're using two single-GPU instances (and therefore two ranks) for simplicity, but we've included
# the basic logic to handle multi-GPU nodes as well, where you'd add more worker processes per node and set `device_ids`
# accordingly.
#
# Despite it being common to use a launcher script to start distributed training, this approach is more flexible and
# allows for more complex orchestration, such as running multiple training jobs concurrently, handling exceptions,
# running distributed training alongside other tasks on the same cluster. It's also significantly easier to debug
# and monitor, as you can see the output of each rank in real-time and get stack traces if a worker fails.


import argparse
import time

import kubetorch as kt
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


# ## Define the PyTorch distributed training logic
# This is the function that will be run on each worker. It initializes the distributed training environment,
# creates a simple model and optimizer, and runs a training loop.
def train(epochs, batch_size=32):
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    print(f"Rank {rank} of {torch.distributed.get_world_size()} initialized")

    # Create a simple model and optimizer
    device_id = rank % torch.cuda.device_count()
    model = torch.nn.Linear(batch_size * 10, 1).cuda(device_id)
    model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a simple training loop
    loss = None
    for epoch in range(epochs):
        time.sleep(1)
        optimizer.zero_grad()
        output = model(torch.randn(10 * batch_size).cuda())
        loss = output.sum()
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}: Epoch {epoch}, Loss {loss.item()}")

    print(f"Rank {rank}: Final Loss {loss.item()}")
    torch.distributed.destroy_process_group()
    return loss.tolist(), rank


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

    gpus = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3"),
        launch_timeout=600,
        inactivity_ttl="4h",
    ).distribute("pytorch", workers=2)
    train_ddp = kt.fn(train).to(gpus)

    results = train_ddp(epochs=args.epochs, batch_size=args.batch_size)
    print(f"Final losses {results}")


if __name__ == "__main__":
    main()
