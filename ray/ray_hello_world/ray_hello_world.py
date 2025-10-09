import time

import kubetorch as kt
import ray


@ray.remote
def hello_world(x):
    return f"Hello, {x * x} World Update!"


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n


def ray_program():
    import ray

    ray.init(address="auto")

    futures = [hello_world.remote(i) for i in range(4)]
    print(ray.get(futures))

    counters = [Counter.remote() for _ in range(4)]
    [c.increment.remote() for c in counters]
    counters[0].increment.remote()
    futures = [c.read.remote() for c in counters]
    print(ray.get(futures))

    time.sleep(1)


if __name__ == "__main__":
    image = kt.Image(image_id="rayproject/ray:latest")
    compute = kt.Compute(cpus=2, image=image).distribute("ray", workers=2)

    remote_fn = kt.fn(ray_program).to(compute)
    results = remote_fn()
