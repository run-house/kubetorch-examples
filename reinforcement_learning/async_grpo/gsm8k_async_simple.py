# # Async RL Training with Kubetorch
# This example demonstrates asynchronous GRPO training, overlapping inference and training.
#
# In the context of this example, you can understand Kubetorch as an actor framework.
# The key components are modularized into
# [separate files](https://github.com/run-house/kubetorch-examples/tree/main/reinforcement_learning/async_grpo)
# not shown here:
# - `inference.py`: vLLM-based inference with dynamic LoRA weight loading
# - `trainer.py`: DDP-based training with LoRA and weight publishing
# - `agent.py`: MathAgent that coordinates rollout generation and reward calculation
#
# Checkpoints are published as LoRA weights to the Kubetorch data store, which allow
# the inference workers to poll for updates and load the weights as tensors via
# GPU-to-GPU transfer directly. This logic is contained within the inference
# and training classes.
#
# We import and deploy these as actors to different compute in `main()`.
# Specifically, the vLLM inference is deployed to horizontally scaling replicas with GPUs
# while the training is a set of workers with PyTorch distributed wired up between them.
#
# These deployed actors are now callable from any Python process. Specifically,
# the logic of orchestrating the execution is contained within the `run_grpo()` function
# which makes calls from wherever you run `python gsm8k_async_simple.py` into the
# running training and inference services.
#
# The training loop uses a simple "inference ahead by 1" pattern where we always
# have the next batch's inference running while training on the current batch.

import asyncio
from pathlib import Path

import kubetorch as kt
import numpy as np
import yaml
from agent import SimpleMathAgent

from inference import vLLM
from trainer import GRPOTrainer


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


async def run_eval_in_background(
    agent, test_questions, test_answers, eval_samples, step
):
    """Run evaluation in background without blocking training."""
    try:
        eval_results = await agent.evaluate_accuracy(
            test_questions, test_answers, num_samples=eval_samples, step=step
        )
        if eval_results:
            print(f"[EVAL] Step {step}: Accuracy {eval_results['accuracy']:.3f}")
    except Exception as e:
        print(f"[EVAL] Failed at step {step}: {e}")


# ## Async GRPO Training Loop
# The core training loop that overlaps inference and training. For each batch:
# 1. Start generating rollouts for the next batch (non-blocking)
# 2. Train on the previous batch's rollouts
# 3. Periodically publish LoRA checkpoints and run evaluations
#
# This "inference ahead by 1" pattern keeps both inference and training GPUs busy.
async def run_grpo(
    dataset, test_dataset, train_service, inference_service, config: dict
):
    """Async GRPO training loop: inference ahead by 1, train sequentially."""
    train_config = config.get("training", {})
    inference_config = config.get("inference", {})
    num_epochs = train_config.get("num_epochs", 3)
    batch_size = train_config.get("batch_size", 8)
    num_generations = train_config.get("num_generations", 4)
    checkpoint_interval = train_config.get("checkpoint_interval", 10)
    eval_interval = train_config.get("eval_interval", 5)
    eval_samples = train_config.get("eval_samples", 100)
    num_inference_workers = inference_config.get("num_workers", 1)

    agent = SimpleMathAgent(
        inference_service,
        config,
        checkpoint_version=0,
        num_inference_workers=num_inference_workers,
    )
    print(
        f"Starting GRPO: {num_epochs} epochs, {len(dataset) // batch_size} batches/epoch"
    )

    test_questions = [test_dataset[i]["question"] for i in range(len(test_dataset))]
    test_answers = [test_dataset[i]["answer"] for i in range(len(test_dataset))]

    step = 0
    pending_rollout = None
    pending_evals = []

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        indices = np.random.permutation(len(dataset))

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            step += 1
            batch_indices = indices[i : i + batch_size]
            questions = [dataset[int(idx)]["question"] for idx in batch_indices]
            answers = [dataset[int(idx)]["answer"] for idx in batch_indices]

            next_inference = asyncio.create_task(
                agent.generate_rollouts(questions, answers, num_generations, step=step)
            )

            pending_evals = [t for t in pending_evals if not t.done()]

            if pending_rollout is not None:
                result = (
                    await train_service.train_batch(
                        pending_rollout.prompts,
                        pending_rollout.completions,
                        pending_rollout.token_ids,
                        pending_rollout.rewards,
                        num_generations,
                    )
                )[0]
                m = result.get("metrics", {})
                if m:
                    print(
                        f"[TRAIN] Step {pending_rollout.step}: loss={m['loss']:.4f}, reward={m['reward_mean']:.3f}"
                    )

                if pending_rollout.step % checkpoint_interval == 0:
                    key = f"lora/v{pending_rollout.step}"
                    _, new_version, _ = (
                        await train_service.publish_lora_weights(key, workers=[0])
                    )[0]
                    agent.checkpoint_version = new_version
                    print(f"[CHECKPOINT] Published v{new_version}")

                if pending_rollout.step % eval_interval == 0:
                    eval_task = asyncio.create_task(
                        run_eval_in_background(
                            agent,
                            test_questions,
                            test_answers,
                            eval_samples,
                            pending_rollout.step,
                        )
                    )
                    pending_evals.append(eval_task)

            pending_rollout = await next_inference

    # Train on final rollout
    if pending_rollout is not None:
        result = (
            await train_service.train_batch(
                pending_rollout.prompts,
                pending_rollout.completions,
                pending_rollout.token_ids,
                pending_rollout.rewards,
                num_generations,
            )
        )[0]
        m = result.get("metrics", {})
        if m:
            print(
                f"[TRAIN] Step {pending_rollout.step}: loss={m['loss']:.4f}, reward={m['reward_mean']:.3f}"
            )

    if pending_evals:
        await asyncio.gather(*pending_evals, return_exceptions=True)

    print("\nTraining complete!")


# ## Setup and Deploy with Kubetorch
# Sets up inference and training services on Kubernetes using Kubetorch. The inference
# service uses autoscaling with vLLM, while training uses PyTorch DDP across multiple
# workers. LoRA weights are published to a shared location that inference workers poll
# for updates, enabling dynamic weight loading without service restarts.
async def main():
    print("Starting")
    from datasets import load_dataset

    config = load_config()
    MODEL_ID = config["model"]["id"]
    trainer_config = config.get("trainer", {})
    inference_config = config.get("inference", {})

    print("Loading GSM8K datasets...")
    dataset = load_dataset("gsm8k", "main", split="train")
    test_dataset = load_dataset("gsm8k", "main", split="test")

    inference_compute = kt.Compute(
        gpus="1",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").run_bash(
            "uv pip install --system --break-system-packages --no-deps "
            "-r async_grpo/requirements-inference.txt"
        ),
        launch_timeout=1200,
    ).autoscale(min_scale=inference_config.get("num_workers", 1))

    train_compute = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").pip_install(
            [
                "'torch>=2.2.0'",
                "transformers==4.56.1",
                "datasets==4.1.0",
                "accelerate==1.10.1",
                "peft==0.17.1",
            ]
        ),
        launch_timeout=1200,
        allowed_serialization=["json", "pickle"],
    ).distribute("pytorch", workers=4)

    engine_config = config.get("inference_engine", {})

    # PEFT config for tensor-based LoRA loading (passed to inference at init)
    peft_config = {
        "r": trainer_config.get("lora_r", 16),
        "lora_alpha": trainer_config.get("lora_alpha", 32),
        "lora_dropout": trainer_config.get("lora_dropout", 0.1),
        "target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
    }

    inference_service, train_service = await asyncio.gather(
        kt.cls(vLLM).to_async(
            inference_compute,
            init_args={
                "model_id": MODEL_ID,
                "engine_config": engine_config,
                "peft_config": peft_config,
            },
            get_if_exists=False,
        ),
        kt.cls(GRPOTrainer).to_async(
            train_compute,
            init_args={"model_id": MODEL_ID, "trainer_config": trainer_config},
            get_if_exists=False,
        ),
    )

    inference_service.async_ = True
    train_service.async_ = True

    await train_service.setup()

    # Publish LoRA metadata for inference workers to discover (poller starts automatically on init)
    await train_service.publish_lora_metadata(workers=[0])

    await run_grpo(
        dataset,
        test_dataset,
        train_service,
        inference_service,
        config,
    )


if __name__ == "__main__":
    asyncio.run(main())
