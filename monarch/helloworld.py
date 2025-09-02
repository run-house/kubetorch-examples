import os

import kubetorch as kt

try:
    import monarch

    class HelloActor(monarch.actor.Actor):
        def __init__(self):
            super().__init__()
            self.count = 0

        @monarch.actor.endpoint
        async def hello(self, name: str):
            self.count += 1
            return f"Hello {name}, count: {self.count}"

except (ImportError, AttributeError):
    pass


async def hello_monarch():
    if os.environ["RANK"] != "0":
        return "Non-leader rank"

    print("Starting HelloActor...")
    hello_mesh = await monarch.actor.proc_mesh(gpus=2)
    hello_actor = await hello_mesh.spawn("hello", HelloActor)

    return await hello_actor.hello.call("World")


def main():
    gpus = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="ghcr.io/pytorch-labs/monarch:latest").run_bash(
            "process_allocator --port=26600 --program=monarch_bootstrap"
        ),
        launch_timeout=600,
        inactivity_ttl="1h",
    ).distribute("pytorch", workers=2)
    remote_hello_monarch = kt.fn(hello_monarch).to(gpus)

    results = remote_hello_monarch()
    print(results)


if __name__ == "__main__":
    main()
