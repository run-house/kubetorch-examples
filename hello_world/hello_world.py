import time

import kubetorch as kt


def hello_world(num_prints=1):
    for print_num in range(num_prints):
        print("Hello worlds", print_num)
        time.sleep(1)
    return num_prints


if __name__ == "__main__":
    img = kt.images.Debian().pip_install("numpy")
    compute = kt.Compute(cpus=1, image=img, inactivity_ttl="15m")

    remote_hello = kt.fn(hello_world).to(compute)

    results = remote_hello(5)
    print(f"Printed hello: {results} times")
