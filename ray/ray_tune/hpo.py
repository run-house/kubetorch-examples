# # Ray Hyperparameter Tuning
# In this example, we show you how to start a basic hyperparameter tuning using Ray Tune on remote compute.
# You simply need to write your Ray Tune program as you would normally, and then send it to the remote cluster.
# Kubetorch handles all the complexities of launching and setting up the remote Ray cluster for you.

import time

import kubetorch as kt

import ray
from ray import tune


# ## Define a Ray Tune program
# You should simply think of this as "any regular Ray Tune program" that you would write entirely agnostic of Kubetorch.
# * A dummy objective function that is used by the train_function() to score the hyperparameters.
# * A Ray Tune Tuner that runs the train_function() over a search space of hyperparameters.
def dummy_objective_function(width, height, step):
    """Dummy objective function for hyperparameter optimization."""
    # Simulate some computation time
    time.sleep(0.1)
    # Return a score based on the parameters
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


def ray_tune_hpo(num_samples=4, max_concurrent_trials=2):
    """Ray Tune hyperparameter optimization function for testing."""
    # Initialize Ray (should connect to existing cluster)
    ray.init(address="auto")
    print("hello")

    def train_function(config):
        """Training function for Ray Tune."""
        step = 0
        for step in range(3):  # Short training for testing
            score = dummy_objective_function(config["width"], config["height"], step)
            # Report the score to Tune
            tune.report(dict(score=score, step=step))

    # Define the search space
    search_space = {
        "width": tune.uniform(0, 10),
        "height": tune.uniform(-10, 10),
    }

    # Create and run the tuner
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            max_concurrent_trials=max_concurrent_trials,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result()

    # Return summary of the HPO run
    return {
        "best_score": best_result.metrics["score"],
        "best_config": best_result.config,
        "num_trials": len(results),
        "status": "completed",
    }


# ## Define Compute and Execution
# In code, we will define the compute our Ray Tune program will run on, dispatch our function to
# the compute and execute it normally, as if it were local, propagating the values we want
# to the remote function call. You must have Kuberay installed on your cluster, the installation
# instructions with Kubetorch [are here](https://www.run.house/kubetorch/installation#ray-support-optional).
def find_minimum():
    ray_compute = kt.Compute(
        cpus="2", memory="3Gi", image=kt.Image(image_id="rayproject/ray")
    ).distribute("ray", workers=2)

    remote_fn = kt.fn(ray_tune_hpo).to(ray_compute)

    # Run Ray Tune HPO with small parameters for testing
    results = remote_fn(num_samples=2, max_concurrent_trials=1)
    return results


if __name__ == "__main__":
    res = find_minimum()
    print(res)
