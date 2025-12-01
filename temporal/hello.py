# Adapted from https://github.com/temporalio/samples-python/blob/74fdf502e0d646dc18ab60a304b472f3460be2ef/hello/hello_activity.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker


# While we could use multiple parameters in the activity, Temporal strongly
# encourages using a single dataclass instead which can have fields added to it
# in a backwards-compatible way.
@dataclass
class ComposeGreetingInput:
    greeting: str
    name: str


# Regular Python code and functions, written as part of your monorepo
def hello_world(greeting, name):
    import time

    output = f"Hello, {name}, {greeting}"
    time.sleep(1)
    return output


# Basic activity that logs and does string concatenation
@activity.defn
def compose_greeting(input: ComposeGreetingInput) -> str:
    import kubetorch as kt  # Import inside activity to avoid workflow sandbox restrictions

    activity.logger.info("Running activity with parameter %s" % input)
    compute = kt.Compute(cpus=0.1, image=kt.Image().pip_install(["temporalio"]))
    remote_hello = kt.fn(hello_world).to(compute)
    result = remote_hello(input.greeting, input.name)
    remote_hello.teardown()
    return result


# Basic workflow that logs and invokes an activity
@workflow.defn
class GreetingWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        workflow.logger.info("Running workflow with parameter %s" % name)
        return await workflow.execute_activity(
            compose_greeting,
            ComposeGreetingInput("Hello", name),
            start_to_close_timeout=timedelta(seconds=120),
        )


async def main():
    import logging

    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")

    # Start client
    client = await Client.connect(**config)

    # Run a worker for the workflow
    async with Worker(
        client,
        task_queue="hello-activity-task-queue",
        workflows=[GreetingWorkflow],
        activities=[compose_greeting],
        # Non-async activities require an executor;
        # a thread pool executor is recommended.
        # This same thread pool could be passed to multiple workers if desired.
        activity_executor=ThreadPoolExecutor(5),
    ):

        # While the worker is running, use the client to run the workflow and
        # print out its result. Note, in many production setups, the client
        # would be in a completely separate process from the worker.
        result = await client.execute_workflow(
            GreetingWorkflow.run,
            "World",
            id="hello-activity-workflow-id",
            task_queue="hello-activity-task-queue",
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
