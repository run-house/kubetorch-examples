# # Demonstrating Fault Tolerance with Automatic Try-Catch
# 1. A function to process file that fails randomly 20% of the time and throws a fake OOM
# 2. Hard coded example of instance specs that are used by `launch_service` to launch our function onto compute
# 3. Iterating through the instance config sizes and calling the remote processing (running on larger and larger compute)
# in a loop over any files that failed the previous size.

# This code would run identically from local laptop as it would from CI (e.g. GitHub Actions)
# or from an orchestrator. All actual execution is dispatched out, with the orchestrator just acting as
# the "driver" program that handles the faults that propagate back and deciding what to do next (in this case,
# trying larger compute)
import random
import time
from concurrent.futures import as_completed, ThreadPoolExecutor

import kubetorch as kt


def process_file(filename):
    """A dummy processing function; i.e. your existing data processing"""
    time.sleep(1)  # Load data, process data
    if random.random() < 0.2:  # 20% chance of OOM
        raise MemoryError("Out of memory")
    else:
        return f"success {filename}!"


# You can specify resources and also use taints to prevent scheduling
# unless you really need it.
INSTANCE_SIZES = {
    "small": {
        "cpus": 0.25,
        "memory": "0.5Gi",
    },
    "medium": {
        "cpus": 1,
        "memory": "0.5Gi",
    },
    "large": {
        "cpus": 2,
        "memory": "1Gi",
        "disk": "10Gi",
        "tolerations": [
            {
                "key": "instance-size",
                "operator": "Equal",
                "value": "large",
                "effect": "NoSchedule",
            }
        ],
    },
}

# A small helper to launch compute based on the config settings
def launch_service(
    name,
    cpus,
    memory,
    gpus=None,
    disk="10Gi",
    min_replicas=0,
    max_replicas=10,
    tolerations=None,
):
    compute = kt.Compute(
        cpus=cpus,
        memory=memory,
        gpus=gpus,
        tolerations=tolerations,
    ).autoscale(min_replicas=min_replicas, max_replicas=max_replicas, concurrency=1)

    return kt.fn(process_file, f"service-{name}").to(compute)


# Code runs and calls this dummy processing function in a loop over 100 "files",
# failing over to larger and larger compute.
if __name__ == "__main__":
    filenames = [f"file{i}" for i in range(1, 100)]  # Your files
    results = []

    for name, config in INSTANCE_SIZES.items():
        if len(filenames) > 0:
            failed_files = []

            service = launch_service(name, **config)  # Launch the next size up

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_file = {
                    executor.submit(service, filename): filename
                    for filename in filenames
                }
                for future in as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        results.append(future.result())
                    except Exception:
                        failed_files.append(filename)

            filenames = failed_files
            print(f"Failed at size {name}: ", failed_files)
            service.teardown()  # Release the resources

    print("Still failed ", failed_files)  # Do something with this in practice
