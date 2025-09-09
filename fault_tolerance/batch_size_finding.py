# # Demonstrating Fault Tolerance with Automatic Batch Size Finding
#
# ::youtube[Fault Tolerance]{url="https://www.youtube.com/watch?v=CH0mMcR5hZ8"}
#
# In this example, we demonstrate how to handle out-of-memory (OOM) errors during
# distributed training by automatically adjusting the batch size.
# But more generically, this shows how Kubetorch gives you powerful programmatic control
# and fault tolerance over your training and other ML jobs. You can launch nodes conservatively,
# catch OOMs, and then relaunch on larger compute (instead of always requesting maximum resources);
# or you can automatically set up retries with different configs; or you can tolerate pre-emptions;
# etc.
import time

import kubetorch as kt
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# ## Dummy Training Function
# This function simulates a distributed training job using PyTorch's DDP.
# You can think of this as the training entrypoint to any training job you have today.
def train(epochs, batch_size=32):
    torch.distributed.init_process_group(backend="nccl")
    try:
        rank = torch.distributed.get_rank()
        print(f"Rank {rank} of {torch.distributed.get_world_size()} initialized")

        # Create a simple model and optimizer
        device_id = rank % torch.cuda.device_count()
        model = torch.nn.Linear(1024, 1024).cuda(device_id)
        model = DDP(model, device_ids=[device_id])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Perform a simple training loop
        loss = None
        for epoch in range(epochs):
            time.sleep(1)
            batch = torch.randn(batch_size, 1024).cuda(device_id)
            optimizer.zero_grad()
            output = model(batch).mm(torch.transpose(batch, 0, 1))
            loss = output.sum()
            loss.backward()
            optimizer.step()

            print(f"Rank {rank}: Epoch {epoch}, Loss {loss.item()}")

        # Print memory consumption
        mem = torch.cuda.memory_allocated(device_id) / (1024**3)
        print(f"Rank {rank}: Memory allocated: {mem:.2f} GB")
        print(f"Rank {rank}: Final Loss {loss.item()}")
    finally:
        torch.distributed.destroy_process_group()
    return loss.tolist(), rank


# ## Catching an OOM and Continuing Training
# In main, we launch a distributed training job on multiple nodes (1 GPU per node, but illustrates multi-node)
# using PyTorch (which can be any training), running one epoch at a time with increasing batch sizes until it
# finds an out-of-memory (OOM) error. Then, it just goes and restarts the training from the last successful batch size.
# The compute pods, data, and model are all kept in place, even after CUDA OOM, so the restart to start real training is instant.
# You now have powerful programmatic control over your program's behavior in the face of faults happening from within
# the training, instead of the program failing.
if __name__ == "__main__":
    gpus = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3"),
        launch_timeout=600,
    ).distribute("pytorch", workers=4)
    train_ddp = kt.fn(train).to(gpus)

    batch_size = 2**12
    while batch_size <= 2**20:
        try:
            print(f"Running with batch size {2*batch_size}")
            train_ddp(epochs=1, batch_size=2 * batch_size)
            batch_size *= 2
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(
                    f"OOM with batch size {2*batch_size}, setting batch size to {batch_size}"
                )
                break
            else:
                raise e

    results = train_ddp(epochs=10, batch_size=batch_size)
    print(f"Final losses {results}")
