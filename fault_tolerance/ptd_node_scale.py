# PyTorch Distributed CPU Training at Scale
# Demonstrates Kubetorch orchestrating 400+ CPU nodes using PyTorch DDP.
# Uses Gloo backend for CPU-only distributed training with minimal resource requirements.

import argparse
import time

import kubetorch as kt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def distributed_train_cpu(epochs, iterations_per_epoch=10):
    """Lightweight distributed training function for CPU workers."""
    # Initialize the distributed process group
    dist.init_process_group(backend="gloo")  # Using gloo backend for CPU
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Log from select ranks only to avoid output spam
    if rank < 5 or rank >= world_size - 5 or rank % 100 == 0:
        print(f"Rank {rank}/{world_size} initialized")

    # Minimal model for low memory usage
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 1))

    model = DDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    final_value = 0.0
    for epoch in range(epochs):
        epoch_sum = 0.0

        for i in range(iterations_per_epoch):
            data = torch.randn(10, 100)
            target = torch.randn(10, 1)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, target)

            loss.backward()
            optimizer.step()

            epoch_sum += loss.item()

        final_value = epoch_sum / iterations_per_epoch

        # Log progress from selected ranks
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Value: {final_value:.4f}")
        elif rank % 100 == 0 and rank > 0:  # Log every 100th rank
            print(f"Rank {rank}: Epoch {epoch+1}/{epochs}")

    # All-reduce using SUM (Gloo doesn't support AVG)
    final_tensor = torch.tensor([final_value])
    dist.all_reduce(final_tensor, op=dist.ReduceOp.SUM)
    avg_final_value = final_tensor.item() / world_size

    if rank == 0:
        print(f"\nComputation completed across {world_size} CPU nodes")
        print(f"Average final value across all nodes: {avg_final_value:.4f}")

    dist.destroy_process_group()

    return {
        "rank": rank,
        "world_size": world_size,
        "final_value": final_value,
        "avg_final_value": avg_final_value,
    }


def main():
    parser = argparse.ArgumentParser(description="Large-scale CPU PyTorch Distributed Training with Kubetorch")
    parser.add_argument(
        "--workers",
        type=int,
        default=400,
        help="Number of CPU worker nodes (default: 400)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per epoch (default: 10)",
    )
    parser.add_argument(
        "--cpus-per-worker",
        type=int,
        default=1,
        help="Number of CPUs per worker node (default: 1)",
    )
    args = parser.parse_args()

    print(f"Launching distributed CPU training across {args.workers} nodes...")
    print("Configuration:")
    print(f"  - Workers: {args.workers}")
    print(f"  - CPUs per worker: {args.cpus_per_worker}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Iterations per epoch: {args.iterations}")

    cpu_compute = kt.Compute(
        cpus=args.cpus_per_worker,
        memory="1Gi",
        image=kt.Image(image_id="pytorch/pytorch:latest"),
        launch_timeout=2400,  # 40 minutes for large-scale launch
        inactivity_ttl="2h",
    ).distribute("pytorch", workers=args.workers, num_proc=1)

    distributed_train = kt.fn(distributed_train_cpu).to(cpu_compute)

    start_time = time.time()
    print("\nStarting distributed training...")

    try:
        results = distributed_train(epochs=args.epochs, iterations_per_epoch=args.iterations)

        elapsed_time = time.time() - start_time

        if isinstance(results, list) and len(results) > 0:
            rank_0_result = results[0] if isinstance(results[0], dict) else None

            if rank_0_result:
                print("\n" + "=" * 60)
                print("Training Summary:")
                print(f"  - Total workers: {rank_0_result['world_size']}")
                print(f"  - Training time: {elapsed_time:.2f} seconds")
                print(f"  - Average final value: {rank_0_result['avg_final_value']:.4f}")
                print(
                    f"  - Throughput: {args.workers * args.epochs * args.iterations / elapsed_time:.2f} iterations/second"
                )
                print("=" * 60)

    except Exception as e:
        print(f"Error during training: {e}")
        raise

    print("\nDistributed CPU training completed successfully!")


if __name__ == "__main__":
    main()
