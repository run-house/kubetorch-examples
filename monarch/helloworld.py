#!/usr/bin/env python3
"""
Monarch Hello World Example with Kubetorch

This example demonstrates how to use Kubetorch's Monarch integration to create
a distributed actor system. Monarch is a single-controller actor framework where
rank 0 coordinates distributed actors across worker nodes.
"""

import os

import kubetorch as kt


async def hello_monarch(allocator=None):
    """
    A distributed Monarch actor example.

    The allocator parameter is automatically injected by Kubetorch.
    It provides access to the distributed allocation system.

    You can use it to create allocations and meshes with custom configurations.
    """
    # Get world size from environment (Monarch controller always runs on rank 0)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"\n🎮 Controller starting with {world_size} nodes")
    print(f"📦 Allocator: {allocator}")

    from monarch._rust_bindings.monarch_hyperactor.alloc import (
        AllocConstraints,
        AllocSpec,
    )

    # Import Monarch components
    from monarch.actor import Actor, endpoint, ProcMesh

    # Define a simple distributed actor
    class HelloActor(Actor):
        """A simple actor that counts greetings."""

        def __init__(self):
            super().__init__()
            self.count = 0

        @endpoint
        async def hello(self, name: str):
            """Greet someone and increment counter."""
            self.count += 1
            return f"Hello {name}! (call #{self.count})"

        @endpoint
        async def get_count(self):
            """Get the current call count."""
            return self.count

    # Create an allocation and ProcMesh
    print("🔧 Creating allocation and process mesh...")

    # Create AllocSpec for our distributed system
    spec = AllocSpec(AllocConstraints(), hosts=world_size, gpus=1)
    alloc = allocator.allocate(spec)

    # Create ProcMesh from the allocation (not async in v0)
    proc_mesh = ProcMesh.from_alloc(alloc)
    # await proc_mesh.initialized
    print(f"🌐 Created process mesh with {world_size} nodes")

    # Spawn our distributed actor across the mesh
    hello_actor = proc_mesh.spawn("hello", HelloActor)
    await hello_actor.initialized
    print("🎭 HelloActor spawned across all nodes")

    # Make some distributed calls
    await hello_actor.hello.call("World")
    await hello_actor.hello.call("Kubetorch")
    result3 = await hello_actor.hello.call("Monarch")

    # Get the final count
    count = await hello_actor.get_count.call()

    print("\n📊 Results:")
    print(f"  - Called hello() {count.item()} times")
    print(f"  - Last response: {result3.item()}")

    return {
        "message": "Monarch distributed actors working!",
        "total_calls": count.item(),
        "world_size": world_size,
    }


def main():
    """Deploy a Monarch distributed actor system with Kubetorch."""

    # Configure compute resources
    compute = kt.Compute(
        gpus=1,
        cpus=2,
        memory="4Gi",
        image=kt.Image(image_id="pytorch/pytorch:2.9.0-cuda12.6-cudnn9-devel")
        # Install RDMA libraries for high-performance networking
        .run_bash(
            "apt-get update && apt-get install -y libibverbs1 libibverbs-dev rdma-core librdmacm1 librdmacm-dev"
        )
        # Install Monarch (pre-release includes process_allocator binary)
        .pip_install(["--pre torchmonarch==0.1.0rc7"]),
    ).distribute(
        "monarch", workers=2
    )  # Enable Monarch distribution mode

    # Deploy the function
    remote_hello_monarch = kt.fn(hello_monarch).to(compute)

    # Run the distributed actor system
    print("🚀 Launching Monarch distributed actors...")
    results = remote_hello_monarch()
    print("\n✨ Results:", results)


if __name__ == "__main__":
    main()
