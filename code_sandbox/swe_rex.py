# # Simple Launching of Code Sandboxes
# For RL, teams frequently need to launch custom environments
# for evaluation. In practice, you would use something like the below in between
# inference and calculating rewards, as you actually execute the
# code being generated. More generically, you can imagine the sandbox
# to represent any agent (your agent) that needs to run on a separate
# image, but be trivially called into for execution from other
# services that are part of the RL training loop.
import kubetorch as kt

from swerex.runtime.local import Command, LocalRuntime

# Launch or grab the existing sandbox if it's already up.
def get_sandbox(name: str):
    cpus = kt.Compute(
        cpus="4",
        image=kt.images.Debian().pip_install(["swe-rex"]),
        allowed_serialization=["pickle"],
    )
    runtime = kt.cls(LocalRuntime, name=name).to(cpus, get_if_exists=True)
    runtime.serialization = "pickle"
    return runtime


# Call into the sandbox running remotely on Kubernetes with separate resources
# as if it were local. Here, the local is your local machine, but "local" might
# mean calling from within the evaluation step of your RL training loop.
if __name__ == "__main__":
    sandbox = get_sandbox("swe-rex-sandbox-1")

    # Check if the runtime is alive
    is_alive_response = sandbox.is_alive()
    print("Is the runtime alive? {is_alive_response.is_alive}")

    # Echo a message to the runtime
    echo_response = sandbox.execute(
        Command(command="echo 'Hello, Swerex!'", shell=True)
    )

    # Close the runtime when done
    close_response = sandbox.close()
    print("Runtime closed.")
