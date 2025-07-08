# # PyTorch Multi-Node Distributed Training
# This is a basic example showing how to Pythonically run a PyTorch distributed training script on
# multiple GPUs. Kubetorch is not solely for PyTorch training (supporting arbitrary code & distribution frameworks),
# but it is a common use case.
#
# Often distributed training is launched from multiple parallel CLI commands(`python -m torch.distributed.launch ...`),
# each spawning separate training processes (ranks). Instead, here we are calling `.to()` with Kubetorch to dispatch
# our training entrypoint to remote compute, and then calling `.distribute("pytorch", workers=4)` to create
# 4 replicas and setting up environment variables necessary for PyTorch communication. The replicas concurrently
# to trigger coordinated multi-node training (`torch.distributed.init_process_group` causes each to wait for all to connect,
# and sets up the distributed communication). We're using 4 x 1 GPU instances (and therefore four ranks).
#
# This approach is more flexible than using a launcher or any other system to launch the distributed training.
# First, each iteration loop after the first execution becomes instanteous, with hot-reloading and warm compute
# allowing for local-like iteration on distributed remote compute. Additionally, you can run this identically
# from anywhere, whether checked out from an intern laptop or from within an orchestrator node.
# It's also significantly easier to debug and monitor, as you can see the output of
# each rank in real-time, get stack traces if a worker fails (with logs streaming back), and use
# the built-in PDB debugger to debug.


import argparse
import time

import kubetorch as kt
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


# ## Define the PyTorch Distributed training logic
# This is a dummy training function, but you can think of this function as representative of your training entrypoint function,
# or the a function that will be run on each worker. It initializes the distributed training environment,
# creates a simple model and optimizer, and runs a dummy training loop for a few epochs.
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


# ## Define Compute and Execution
# In code, we will define the compute our training will run on, dispatch our function to
# the compute and replicate it over 4 workers. Then, we call the remote function
# for execution normally, as if it were local, propagating the values we receive
# from argparse through to the remote function call.
#
# The first time you call `.to()` it might take a few minutes to autoscale the nodes
# and pull down the image (PyTorch is a big image!). But then, further iteration takes
# just 1-2 seconds; change the print statement, rerun the script, and you can see that
# your distributed training will restart nearly instatneously.
#
# As a practical note, if you are adapting an existing training for Kubetorch, you can
# typically just rename your existing `main()` into something else like `train()` and
# dispatch the current training entrypoint as-is, with no changes, similarly to below.
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
    ).distribute("pytorch", workers=4)
    train_ddp = kt.fn(train).to(gpus)

    results = train_ddp(epochs=args.epochs, batch_size=args.batch_size)
    print(f"Final losses {results}")


if __name__ == "__main__":
    main()
