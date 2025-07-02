# MNIST Training on Airflow with Kubetorch

This example demonstrates how to use Airflow along with Kubetorch to dispatch
the work of training a basic Torch model to a remote GPU. We'll walk through a **simple training pipeline** that uses the MNIST dataset and PyTorch.

Beyond standard classes and methods for our training pipeline, you'll see that we only need minimal code in the form of callables to utilize Kubetorch dispatching from Airflow (or any other orchestrator).

To test this out for yourself, visit the [Kubetorch Examples](https://github.com/run-house/kubetorch-examples/tree/main/airflow) repository on Github. With minor adjustments, you should be able to deploy the DAG to your own Airflow installation on Kubernetes.

## Kubetorch + Airflow

Airflow is incredibly popular and widely used, but it comes with plenty of problems, especially when it comes to debugging workflows and translating from research to production. Kubetorch enables fast and efficient ML development right inside of your Kubernetes cluster, making it a perfect tool for your Airflow pipelines.

### Example Usage Pattern

- Write Python classes and functions using normal, ordinary coding best practices. Do not think about DAGs or DSLs at all, just write great code.
- Send the code for remote execution with Kubetorch, and figure out whether the code works, debugging it interactively. Kubetorch lets you send the code in seconds, and streams logs back. You can work on remote as if it were local.
- Once you are satisfied with your code, you can write the callables for an Airflow. The code that is actually in the Airflow DAG is the **minimal code** to call out to already working Classes and Functions, defining the order of the steps (or you can even have a one-step Airflow DAG, making Airflow purely for scheduling and observability)
- And you can easily iterate further on your code, or test the pipeline end-to-end from local with no Airflow participation

### Benefits of Using Kubetorch

- **Fast, local iteration** - Your tasks quickly launch pods that run exactly the same in local execution as they would inside of Airflow, so you can run and iterate on your code locally without waiting for helm upgrades, redeployments, and Docker builds. All you need is a `kubeconfig`.
- **Keep it simple with`PythonOperator`** - Requirements and dependencies are defined inside your Python methods with `kt.Compute` and installed on pods deployed by Kubetorch. This allows you to avoid complexities that come with `KubernetesPodOperator`, such as rebuilding and deploying new Docker images manually for each iteration and task in your DAGs.
- **Reusability** - Compute can be reused for faster iterations and our `inactivity_ttl` setting ensures nothing gets left running.
- **Flexibility and Cost Savings** - Between debugging individual tasks and defining compute on a per-task basis, you're able to optimize for only the CPU/GPU you actually need.

## Project Setup

```bash
# A diagram of the folders and files in this example directory:
airflow/
├── dags/
│   ├── kubetorch_mnist.py          # Main DAG file with Airflow tasks
│   └── utils/
│       ├── __init__.py             # Python package initialization
│       ├── callables.py            # Kubetorch callable functions for DAG tasks
│       ├── model.py                # PyTorch neural network model definition
│       ├── simple_trainer.py       # Training class for MNIST model
│       └── transforms.py           # Data transformation utilities
├── Dockerfile                      # Container image for Airflow deployment
├── GUIDE.md                        # (*) This guide file
├── rbac.yaml                       # Kubernetes RBAC permissions for Kubetorch
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Python dependencies for Airflow Dockerfile
└── values.yaml                     # Helm values for Airflow/Kubetorch RBAC resources
```

## Setup and Installation

- (Optional) Install Airflow
- Install Kubetorch on your cluster
- Update service account permissions `rbac.yaml`
- Connect to 8080
- Reach out for assistance

## NN Model and Training Class

We'll start with a definition of the neural network model and a training class that can be deployed with Kubetorch to run our training pipeline. Please note that these require PyTorch but we won't need to install the packages in our Airflow image or locally for iteration.

```python
class Trainer():
```

## Callables and DAG Setup

For our code to run in Airflow, we'll create callable methods. These use the Kubetorch API to package and dispatch our pipeline steps on the appropriate compute for each.

```python
# From kubetorch_mnist
def data_preprocessing_fn():
    logger = logging.getLogger(__name__)
    logger.info("Data Preprocessing")
    return True
```

Next we'll define a simple DAG and create tasks for the each of the callables. Note that we're able to use `PythonOperator` for the tasks because all the heavy lifting for dependencies is handled by Kubetorch dispatching.

```python
```

You can see that this is an incredible minimal amount of code in Airflow. The callable methods run as tasks in the DAG but are also runnable from a Python script, notebook, or anywhere. This allows you to iterate on primary functions before deploying to Airflow.

## Running the DAG

If you don't have an existing Airflow setup, you can use the Dockerfile in [this example](link-to-folder) to build and push an image with the DAG code included.

```bash
```

Then, install Airflow on your Kubernetes cluster using that image.

```
```
For more information on installing Airflow, please see their [installation instructions](link-to-airflow-docs) or get in touch with our team at [team@run.house](mailto:team@run.house). You may need additional settings depending on your cloud provider.

Open your Airflow dashboard by portforwarding into the UI pod.

```bash
```

[IMAGE OF THE DASHBOARD]
