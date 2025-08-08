import kubetorch as kt

from swerex.runtime.local import Command, LocalRuntime


def get_sandbox(name: str):
    cpus = kt.Compute(
        cpus="4",
        image=kt.Image().pip_install(["swe-rex"]),
        allowed_serialization=["pickle"],
    )
    runtime = kt.cls(LocalRuntime, name=name).to(cpus)
    runtime.serialization = "pickle"
    return runtime


if __name__ == "__main__":
    sandbox = get_sandbox("swe-rex-sandbox-1")

    # Check if the runtime is alive
    is_alive_response = sandbox.is_alive()
    print(f"Is the runtime alive? {is_alive_response.is_alive}")

    # echo a message to the runtime
    echo_response = sandbox.execute(
        Command(command="echo 'Hello, Swerex!'", shell=True)
    )

    # Close the runtime when done
    close_response = sandbox.close()
    print("Runtime closed.")