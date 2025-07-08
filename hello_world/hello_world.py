import time

import kubetorch as kt


def hello_worlds(num_prints=1):
    for print_num in range(num_prints):
        print("Hello worlds", print_num)
        time.sleep(1)
    return num_prints


if __name__ == "__main__":
    img = kt.Image().pip_install("numpy")
    compute = kt.Compute(cpus=1, image=img)

    remote_hello = kt.fn(hello_worlds).to(compute)

    results = remote_hello(5)
    print(f"Results: {results}")

    # remote_hello.teardown()
