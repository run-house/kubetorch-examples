# SPMD Large-Scale Node Connectivity Test
# Demonstrates Kubetorch orchestrating 1000+ lightweight workers using SPMD mode.
# Workers coordinate via environment variables without heavy frameworks.

import argparse
import os
import random
import time

import kubetorch as kt


def spmd_worker_task(work_iterations=100):
    """Minimal worker function to verify node connectivity."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Stagger output to avoid log flooding
    time.sleep(random.uniform(0, 0.5))

    if rank < 10 or rank >= world_size - 10 or rank % 100 == 0:
        print(f"Worker {rank}/{world_size} (local_rank={local_rank}) online")

    result = 0
    for i in range(work_iterations):
        result += (rank * i) % 1000
        if i % 50 == 0:
            time.sleep(0.001)

    final_value = result / work_iterations + rank * 0.001

    if rank == 0:
        print(f"All {world_size} workers completed their tasks")
    elif rank < 5 or rank >= world_size - 5:
        print(f"Worker {rank}: completed with value {final_value:.3f}")

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "final_value": final_value,
        "hostname": os.environ.get("HOSTNAME", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Large-scale SPMD Node Connectivity Test with Kubetorch")
    parser.add_argument(
        "--workers",
        type=int,
        default=1000,
        help="Number of worker nodes to launch (default: 1000)",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=8,
        help="Number of processes per worker (default: 8)",
    )
    parser.add_argument(
        "--work-iterations",
        type=int,
        default=500,
        help="Number of work iterations per worker (default: 500)",
    )
    parser.add_argument(
        "--cpus-per-worker",
        type=float,
        default=0.5,
        help="Number of CPUs per worker node (default: 0.5)",
    )
    parser.add_argument(
        "--memory-per-worker",
        type=str,
        default="1Gi",
        help="Memory per worker node (default: 1Gi)",
    )
    args = parser.parse_args()

    print(f"Launching SPMD connectivity test across {args.workers} nodes...")
    print("Configuration:")
    print(f"  - Workers: {args.workers}")
    print(f"  - Procs per worker: {args.procs}")
    print(f"  - CPUs per worker: {args.cpus_per_worker}")
    print(f"  - Memory per worker: {args.memory_per_worker}")
    print(f"  - Work iterations: {args.work_iterations}")

    spmd_compute = kt.Compute(
        cpus=args.cpus_per_worker,
        memory=args.memory_per_worker,
        image=kt.Image(image_id="python:3.11-slim"),
        launch_timeout=3600,
        inactivity_ttl="1h",
    ).distribute(
        "spmd",
        workers=args.workers,
        num_proc=args.procs,
    )

    distributed_worker = kt.fn(spmd_worker_task).to(spmd_compute)

    start_time = time.time()
    print(f"\nLaunching {args.workers} workers...")

    try:
        results = distributed_worker(work_iterations=args.work_iterations)

        elapsed_time = time.time() - start_time

        if isinstance(results, list) and len(results) > 0:
            print("\n" + "=" * 70)
            print("SPMD Scale Test Results:")
            print(f"  - Total workers launched: {len(results)}")
            print(f"  - Launch + execution time: {elapsed_time:.2f} seconds")

            ranks_seen = set()
            unique_hostnames = set()
            total_compute = 0

            for result in results:
                if isinstance(result, dict):
                    ranks_seen.add(f"{result['hostname']}:{result['rank']}")
                    unique_hostnames.add(result["hostname"])
                    total_compute += result["final_value"]

            print(f"  - Workers that reported back: {len(ranks_seen)}")
            print(f"  - Unique hostnames: {len(unique_hostnames)}")
            print(f"  - Total compute result: {total_compute:.2f}")
            print(f"  - Throughput: {len(results) * args.work_iterations / elapsed_time:.2f} iterations/second")

            expected_total = args.workers * args.procs
            if len(ranks_seen) == expected_total:
                print("  SUCCESS: All workers connected and completed!")
            else:
                print(f"  WARNING: {expected_total - len(ranks_seen)} workers did not report back")

            print("\nSample worker results:")
            for i, result in enumerate(results[:5]):
                if isinstance(result, dict):
                    print(f"  Worker {result['rank']}: {result['final_value']:.3f} on {result['hostname']}")

            if len(results) > 10:
                print("  ...")
                for result in results[-5:]:
                    if isinstance(result, dict):
                        print(f"  Worker {result['rank']}: {result['final_value']:.3f} on {result['hostname']}")

            print("=" * 70)
        else:
            print("No results received from workers")

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error after {elapsed_time:.2f}s: {e}")
        raise

    print(f"\nSPMD scale test completed in {elapsed_time:.2f} seconds!")


if __name__ == "__main__":
    main()
