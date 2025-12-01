# # GPU Training with Temporal and Kubetorch
# This example shows how to orchestrate GPU-based XGBoost training using Temporal workflows
# and execute the training on remote GPU compute with Kubetorch. Kubetorch+Temporal is
# the easiest way to submit arbitrary (including GPU, distributed, horizontally scaled, Ray)
# programs to Kubernetes for execution.
#
# In this file, we simply import the `launch_training()` function from `trainer.py`,
# and call that from within a Temporal worker. This function was developed locally,
# with very fast iteration loops (redeploying local code changes takes <2 seconds).
#
# Then, once ready, you use it as is in production. We rely on Temporal to schedule
# the execution, make execution durable, and unify the orchestration of
# training (including GPU training) into our Temporal-based orchestration system
# without taking on any overhead. This simply requires the Temporal Worker to have
# authorization to access a Kubernetes cluster (either via OIDC or Kubeconfig), and
# you can configure the Worker to have production-y Kubetorch settings (such as setting
# namespace or username to `prod` with `kt config set`)

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

from trainer import launch_training

# Input/Output dataclasses for Temporal activities
@dataclass
class TrainingConfig:
    """Configuration for XGBoost training."""

    num_rounds: int
    train_params: dict


@dataclass
class TrainingResult:
    """Results from XGBoost training."""

    accuracy: float
    model_path: str


# Temporal Activities
@activity.defn
def train_xgboost_on_gpu(config: TrainingConfig) -> TrainingResult:
    """
    Activity that runs XGBoost training on remote GPU compute using KubeTorch.
    Delegates to trainer.launch_training() which handles all the GPU setup and training.
    """
    # Call the training function from trainer.py
    # Pass activity.logger so training logs appear in Temporal
    accuracy, model_path = launch_training(
        config, logger=activity.logger, teardown=True
    )

    return TrainingResult(accuracy=accuracy, model_path=model_path)


# Temporal Workflow
@workflow.defn
class XGBoostTrainingWorkflow:
    """Workflow that orchestrates GPU-based XGBoost training."""

    @workflow.run
    async def run(self, config: TrainingConfig) -> TrainingResult:
        workflow.logger.info(
            f"Starting XGBoost training workflow with config: {config}"
        )

        # Execute training activity with extended timeout for GPU training
        result = await workflow.execute_activity(
            train_xgboost_on_gpu,
            config,
            start_to_close_timeout=timedelta(minutes=30),  # GPU training can take time
        )

        workflow.logger.info(f"Training completed with accuracy: {result.accuracy:.4f}")
        return result


async def main():
    import logging

    logging.basicConfig(level=logging.INFO)

    # Temporal configuration
    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(**config)

    # Run worker for the workflow
    async with Worker(
        client,
        task_queue="xgboost-gpu-training-queue",
        workflows=[XGBoostTrainingWorkflow],
        activities=[train_xgboost_on_gpu],
        activity_executor=ThreadPoolExecutor(max_workers=5),
    ):
        # XGBoost parameters optimized for GPU
        train_params = {
            "objective": "multi:softmax",
            "num_class": 10,
            "eval_metric": ["mlogloss", "merror"],
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "predictor": "gpu_predictor",
            "device": "cuda",
            "seed": 42,
            "n_jobs": -1,
        }

        # Execute the workflow
        training_config = TrainingConfig(
            num_rounds=100,
            train_params=train_params,
        )

        result = await client.execute_workflow(
            XGBoostTrainingWorkflow.run,
            training_config,
            id="xgboost-gpu-training-workflow",
            task_queue="xgboost-gpu-training-queue",
        )

        print(f"Training completed; model accuracy: {result.accuracy:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
