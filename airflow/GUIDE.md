# MNIST Training on Airflow with Kubetorch

This example demonstrates how to use Airflow with Kubetorch to dispatch
the work of training a basic Torch model to a remote GPU. We'll walk through a **simple training pipeline** that uses the MNIST dataset and PyTorch.

Beyond standard classes and methods for our training pipeline, you'll see that we only need minimal code in the form of task callables to utilize Kubetorch dispatching from Airflow. The same structure can be used to improve development velocity, research-to-production, and fault tolerance with any pipeline orchestrator (e.g. Argo, Dagster, Prefect, Metaflow) without requiring any direct integration.

To test this out for yourself, visit the [Kubetorch Examples](https://github.com/run-house/kubetorch-examples/tree/main/airflow) repository on Github. With minor adjustments, you should be able to deploy the DAG to your own Airflow installation on Kubernetes.

## Kubetorch + Airflow

[Apache Airflow](https://airflow.apache.org/) is widely used in ML, but it comes with plenty of problems, especially when it comes to debugging workflows and translating between research and production. **Kubetorch** enables fast and efficient ML development right inside of your Kubernetes cluster and can be deployed as-is inside of your Airflow pipelines, closing the research-to-production gap. You can learn more about Kubetorch in our [documentation](https://www.run.house/kubetorch/introduction).

### Example Usage Pattern

1. **Write Python classes and functions** using normal, ordinary coding best practices. Do not think about DAGs or DSLs at all.
2. **Send the code for remote execution with Kubetorch**, and figure out whether the code works by debugging it interactively. Kubetorch lets you send the code in seconds and streams logs back. _You can work on remote as if it were local_.
3. Once you are satisfied with your code, **write the callables for Airflow tasks**. The Airflow DAG definition contains _minimal code_ to call out to already working Classes and Functions, detailing the order of the tasks. You can even create one-step DAG, leveraging Airflow purely for scheduling and observability.
4. **Easily iterate further** on your code, or test the pipeline end-to-end from local with no Airflow participation.

### Benefits of Using Kubetorch

- **Fast, local iteration** - Your tasks quickly launch Kubernetes pods that run exactly the same from local execution as they would inside of Airflow. You can run and iterate on your code locally without waiting for Docker builds, helm upgrades, and redeployments.
- **Keep it simple with`PythonOperator`** - Requirements and dependencies are defined inside your Python methods with `kt.Compute` and installed on pods deployed by Kubetorch. This allows you to avoid complexities that come with `KubernetesPodOperator`, such as the need to rebuild and deploy new Docker images for each iteration and task in your DAGs.
- **Reusability** - Compute can be reused for faster iterations and our `inactivity_ttl` setting ensures nothing gets left running.
- **Flexibility and Cost Savings** - Between debugging individual tasks and defining compute on a per-task basis, you're able to optimize for only the CPU/GPU you actually need.

## Project Setup

Below is a diagram of the folders and files in this example directory. You'll be most interested in the DAG file `/dags/kubetorch_mnist_dag.py` and the supporting Python files in `/dags/kubetorch_example`. The `tasks.py` file contains the callables that will run in your tasks, each of which can be run independently and locally for quick testing.

```bash
airflow/
├── dags/
│   ├── kubetorch_mnist_dag.py      # Main DAG file with Airflow tasks
│   └── kubetorch_examples/
│       ├── __init__.py             # Python package initialization
│       ├── model.py                # PyTorch model definition
│       ├── tasks.py                # Task callables
│       ├── trainer.py              # Training class for MNIST model
│       └── transforms.py           # Data transformation utilities
├── k8s/
│   ├── rbac.yaml                   # Kubernetes RBAC permissions for Kubetorch
│   └── values.yaml                 # Helm values for Airflow/Kubetorch RBAC resources
├── docker/
│   ├── Dockerfile                  # Container image for Airflow deployment
│   └── requirements.txt            # Python dependencies for Airflow Dockerfile
├── GUIDE.md                        # (*) This guide file
└── README.md                       # Project overview and setup instructions
```

## Setup and Installation

Ideally, you should install Kubetorch and Airflow in the same Kubernetes cluster. This will simplify connections between services and improve overall reliability.

If you do not already have Airflow running on your own Kubernetes cluster, you can navigate to the [`README`](https://github.com/run-house/kubetorch-examples/tree/main/airflow) in this example for instructions on a basic install.

Here are the necessary steps to run Airflow with Kubetorch:

- **Install Airflow (Optional)** - If you are starting fresh, you'll first need to install Airflow on your cluster. Take a look at our [`README`](https://github.com/run-house/kubetorch-examples/tree/main/airflow) for instructions on building a custom Docker image with your DAGs, or visit Airflow's [installation docs](https://airflow.apache.org/docs/apache-airflow/3.0.3/installation/index.html)
- **Install Kubetorch** - Please visit [our documentation](https://www.run.house/kubetorch/installation) for instructions on installing Kubetorch on your cluster.
- **Update Service Account Permissions** - Your Airflow workers will need a set of permissions to run Kubetorch methods. By installing Kubetorch, you'll have already created the appropriate roles and you can apply them to your worker's service account via the YAML file in this example.
  ```bash
  # Be sure to update the YAML file with your service account name and namespace
  kubectl apply -f k8s/rbac.yaml
  ```

Kubetorch is built to run on any Kubernetes cluster, but it's possible that you may run into issues depending on your particular setup. Please reach out to our team at [team@run.house](mailto:team@run.house) for hands-on assistance.

## NN Model and Training Class

We'll start with a definition of a simple neural network model and a training class that can be deployed with Kubetorch to run our training pipeline. Please note that these require PyTorch but we won't need to install the packages in our Airflow image or locally for iteration.

### Model

A very basic feedforward neural network with three fully connected layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Trainer

Next, we define a class that will hold the various methods needed to fine-tune the model using PyTorch. We'll later wrap this with `kt.cls` to create a local instance that dispatches operations to a GPU on our Kubernetes cluster.

Notice that this is a regular Python class without any Kubetorch opinionation.

```python
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from kubetorch_example.model import SimpleNN
from kubetorch_example.transforms import get_transform

from torch.utils.data import DataLoader
from torchvision import datasets


class SimpleTrainer:
    def __init__(self, from_checkpoint=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN().to(self.device)
        if from_checkpoint:
            self.model.load_state_dict(
                torch.load(from_checkpoint, map_location=self.device)
            )

        self.train_loader = None
        self.test_loader = None
        self.epoch = 0
        self.transform = get_transform()

    def load_data(self, path, batch_size, download=True):
        ...

    def train_model(self, learning_rate=0.001):
        ...

    def test_model(self):
        ...

    def predict(self, data):
        ...

    def save_model(self, bucket, s3_path):
        ...

```

You can see the full definition of `SimpleTrainer` in the [`trainer.py`](https://github.com/run-house/kubetorch-examples/tree/main/airflow/dags/kubetorch_example/trainer.py) file.

## Python Callables for Tasks

For our code to run in Airflow, we'll create callable methods. These use the Kubetorch API to package and dispatch our pipeline steps on the appropriate compute for each. This code is runnable and meant as a starting point to illustrate a training pipeline using Kubetorch.

The beauty of Kubetorch is that each of these tasks can be run identitically on your local machine and within Airflow (or any orchestrator).

### Task 1: Data Preprocessing

In many cases, you'll need a preprocessing step in your training pipeline. This task might copy your dataset to a more convenient location or transform the data to feed into your training function.

The method below is a placeholder to illustrate how you can run each task on an appropriate node (e.g. CPU-only) with Kubetorch to save on resources.

```python
def data_preprocessing(**kwargs):
    image = kt.Image()
    compute = kt.Compute(
        cpus="0.1",
        image=image,
        inactivity_ttl="10m",
    )
    logger.info("Step 1: Data Preprocessing")
    preprocessor = kt.fn(data_preprocessing_fn, name="data-preproc").to(compute)
    time.sleep(5)
    try:
        success = preprocessor()
        logger.info(f"Data Preprocessed: {success}")
    except Exception as e:
        logger.error(f"Data Preprocessing failed: {e}")
        raise e
```

### Task 2: Run Training

Next, we'll get into the primary task of the pipeline. This training function uses the `SimpleTrainer` class defined above to load in the MNIST dataset, train our NN model, test against a dataset sample, and save the model to an AWS bucket.

We can simply put the dispatch and execution of the model in the callable identical to how we have run it locally, ensuring identical research-to-production execution.

```python
def run_training(**kwargs):
    from kubetorch_example.trainer import SimpleTrainer

    logger.info("Step 2: Run Training")
    compute = kt.Compute(
        gpus="1",
        image=kt.Image(image_id=PYTORCH_IMAGE_ID),
        launch_timeout=600,
        inactivity_ttl="10m",
    )

    model = kt.cls(SimpleTrainer).to(compute)

    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    model.load_data("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)
        model.test_model()
        model.save_model(
            bucket_name=S3_BUCKET_NAME,
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )
```

### Task 3: Deploy Inference

Finally, we'll deploy a new service for inference with the trained model checkpoint. Note that we are defining a new compute object rather than reusing the training compute above. We load down the model weights in the image to achieve faster cold start times for our inference service.

```python
def deploy_inference(**kwargs):
    from kubetorch_example.trainer import SimpleTrainer

    logger.info("Step 3: Deploy Inference")
    checkpoint_path = f"s3://{S3_BUCKET_NAME}/checkpoints/model_final.pth"
    local_checkpoint_path = "/model.pth"
    img = kt.Image(image_id=PYTORCH_IMAGE_ID).run_bash(
        f"aws s3 cp {checkpoint_path} {local_checkpoint_path}"
    )
    inference_compute = kt.Compute(
        gpus="1",
        image=img,
        launch_timeout=600,
        inactivity_ttl="10m",
    )

    inference = kt.cls(SimpleTrainer).to(
        inference_compute, init_args={"from_checkpoint": local_checkpoint_path}
    )
    # We distribute the inference service as an autoscaling pool of between 0 and 6 replicas, with a maximum concurrency of 16.
    inference.distribute(num_nodes=(0, 6), max_concurrency=16)
```

### Testing and Deploying

The [`tasks.py`](https://github.com/run-house/kubetorch-examples/tree/main/airflow/dags/kubetorch_example/tasks.py) file includes the full implementation for each of the Airflow tasks above. For local testing, you can run these methods by adding code to call each in main.

```python
# Swap out the method with any of the three tasks defined above.
if __name__ == "__main__":
    run_training()
```

Then in your terminal run the file directly. For dispatching to Kubernetes, you'll need access to your cluster via a `kubeconfig` file or similar.

```bash
python tasks.py
```

## Defining our DAG

Next we'll define a simple DAG and create tasks for the each of the callables. Note that we're able to use `PythonOperator` for the tasks because all the heavy lifting for dependencies is handled by Kubetorch dispatching.

```python
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from kubetorch_example.tasks import data_preprocessing, deploy_inference, run_training


default_args = {
    "owner": "matt",
    "depends_on_past": False,
    "start_date": datetime(2024, 8, 6),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "schedule_interval": None,  # or your desired schedule
    "catchup": False,
    "max_active_runs": 1,
}

with DAG(
    "mnist_training_pipeline",
    default_args=default_args,
    description="A simple PyTorch training DAG with multiple steps",
    schedule=None,
) as dag:

    preprocess_data_task = PythonOperator(
        task_id="preprocess_data_task",
        python_callable=data_preprocessing,
        dag=dag,
    )

    train_model_task = PythonOperator(
        task_id="train_model_task",
        python_callable=run_training,
        dag=dag,
    )

    deploy_inference_task = PythonOperator(
        task_id="deploy_inference_task",
        python_callable=deploy_inference,
        dag=dag,
    )

    preprocess_data_task >> train_model_task >> deploy_inference_task
```

You can see that this is an incredibly minimal amount of code in Airflow. The callable methods run as tasks in the DAG but are also runnable from a Python script, notebook, or anywhere. This allows you to iterate on primary functions before deploying to Airflow.

## Running the DAG

Assuming you've successfully installed Airflow and Kubetorch, you can connect to your Airflow dashboard via a `port-forward` command.
```bash
kubectl port-forward svc/airflow-api-server 8080:8080 --namespace airflow
```
Then navigate to `localhost:8080` in your browser. By default, you'll be able to log in with username `admin` and password `admin`.

![SCreenshot of Airflow dashboard with DAGs](https://runhouse-tutorials.s3.us-east-1.amazonaws.com/airflow-dashboard-dags.jpg)

## Summary

Kubetorch makes it possible to dispatch and execute your code on your Kubernetes cluster from anywhere. This allows you to rapidly test and iterate on your ML code locally with scaled compute (e.g. GPUs in your cloud) and easily translate to production when ready. The concepts covered in this example are specific to Airflow but can be applied to any orchestrator of your choice.

If you're interested in trying Kubetorch or there's another use case you'd love to see, please reach out to [team@run.house](mailto:team@run.house) and we'll work together to make it happen.
