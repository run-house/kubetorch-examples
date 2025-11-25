import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kubetorch as kt
import numpy as np

from inference.sglang_engine import SGLang
from rl_gsm8k.math_agent import SimpleMathAgent
from rl_gsm8k.trainer import GRPOTrainer


async def simple_async_grpo(
    dataset,
    train_service,
    inference_service,
    num_epochs=3,
    batch_size=8,
    num_generations=4,
    checkpoint_interval=10,
):
    agent = SimpleMathAgent(inference_service, checkpoint_version=0)
    indices = np.random.permutation(len(dataset))

    training_tasks = []
    inference_tasks = []
    steps_completed = 0
    total_steps = num_epochs * len(indices) // batch_size

    # Lock to ensure only ONE train_batch() call at a time
    # Multiple inferences can run in parallel, but training must be serial for FSDP
    training_lock = asyncio.Lock()

    print(f"Starting simple async GRPO for {total_steps} steps")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        for i in range(0, len(indices), batch_size):
            # Control parallelism - limit concurrent inferences to avoid overwhelming GPU memory
            max_inference_parallel = 2  # Max parallel inference requests

            # Clean up completed inference tasks that finished
            done_inference = [t for t in inference_tasks if t.done()]
            for task in done_inference:
                inference_tasks.remove(task)

            # Wait if too many inferences are running
            while len(inference_tasks) >= max_inference_parallel:
                await asyncio.sleep(
                    0.1
                )  # Yield to event loop so inference tasks can run
                done_inference = [t for t in inference_tasks if t.done()]
                for task in done_inference:
                    inference_tasks.remove(task)

            # Clean up completed training tasks (training is serialized by lock)
            done_tasks = [t for t in training_tasks if t.done()]
            for task in done_tasks:
                training_tasks.remove(task)
                await task  # Ensure exceptions are raised

            # Get batch data
            batch_indices = indices[i : i + batch_size]
            questions = [dataset[int(idx)]["question"] for idx in batch_indices]
            answers = [dataset[int(idx)]["answer"] for idx in batch_indices]

            # Start inference task
            current_step = steps_completed + 1
            inference_task = asyncio.create_task(
                agent.generate_batch(
                    questions, answers, num_generations, step_num=current_step
                )
            )
            inference_tasks.append(inference_task)
            print(f"[SCHEDULER] Launched inference for step {current_step}")

            # When inference completes, start training
            async def train_when_ready(inf_task, step_num):
                print(f"[TRAINING] Waiting for inference results for step {step_num}")
                result = await inf_task
                inference_tasks.remove(inf_task)

                # Check if result was stale and skipped
                if result[0] is None:
                    print(
                        f"[TRAINING] Skipping training for stale request at step {step_num}"
                    )
                    return step_num

                prompts, completions, token_ids, rewards = result

                # Acquire lock to ensure only ONE train_batch() call at a time
                # This is critical for FSDP - concurrent calls will corrupt memory
                async with training_lock:
                    print(
                        f"[TRAINING] Starting training for step {step_num} (lock acquired)"
                    )

                    # Train on this batch (data will be split by the service automatically)
                    # This await completes only after ALL distributed workers finish (including barrier)
                    metrics = await train_service.train_batch(
                        prompts, completions, token_ids, rewards, num_generations
                    )

                    # Verify we got valid metrics (training actually ran)
                    if (
                        not metrics
                        or not isinstance(metrics, list)
                        or len(metrics) == 0
                    ):
                        raise RuntimeError(
                            f"Training failed to return valid metrics for step {step_num}: {metrics}"
                        )

                    print(
                        f"[TRAINING] Completed step {step_num}: loss={metrics[0]['metrics']['loss']:.4f}, "
                        f"reward={metrics[0]['metrics']['reward_mean']:.3f} (ALL workers synced)"
                    )

                    # Save checkpoint periodically (must be inside lock - accesses model)
                    # Training is guaranteed complete at this point (barrier passed)
                    if step_num % checkpoint_interval == 0:
                        print(
                            f"[CHECKPOINT] Training complete for step {step_num}, saving checkpoint..."
                        )

                        # Save checkpoint (kubetorch SPMD returns list of results from each worker)
                        # This waits for all workers to save their checkpoint shards
                        checkpoint_results = await train_service.save_checkpoint()
                        # Take result from rank 0 worker
                        checkpoint_result = (
                            checkpoint_results[0]
                            if isinstance(checkpoint_results, list)
                            else checkpoint_results
                        )
                        checkpoint_path, new_version = checkpoint_result
                        print(f"[CHECKPOINT] Checkpoint saved: {checkpoint_path}")

                        print(
                            f"[CHECKPOINT] Hot-swapping inference to v{new_version}..."
                        )
                        # Use training service to redeploy inference with new checkpoint
                        # This rsyncs checkpoint and calls update_weights_from_disk
                        # If update fails, this will raise an exception and halt training
                        try:
                            redeploy_results = await train_service.redeploy_inference(
                                inference_service,
                                checkpoint_path,
                                serialization="pickle",
                            )
                            # Take result from rank 0 worker (SPMD modules always return lists)
                            new_service = (
                                redeploy_results[0]
                                if isinstance(redeploy_results, list)
                                else redeploy_results
                            )

                            # Update agent's reference and version
                            agent.inference_service = new_service
                            agent.inference_service.async_ = True
                            agent.checkpoint_version = new_version
                            print(
                                f"[CHECKPOINT] Hot-swap complete: v{new_version} @ {checkpoint_path}"
                            )
                        except Exception as e:
                            print(f"[CHECKPOINT] FATAL: Hot-swap failed: {e}")
                            print(
                                "[CHECKPOINT] Inference service may be unhealthy, halting training"
                            )
                            raise RuntimeError(
                                f"Checkpoint hot-swap failed at step {step_num}"
                            ) from e

                return step_num

            # Create training task that waits for inference
            training_task = asyncio.create_task(
                train_when_ready(inference_task, steps_completed + 1)
            )
            training_tasks.append(training_task)
            steps_completed += 1

    # Wait for remaining tasks
    await asyncio.gather(*training_tasks)
    print("\nTraining complete!")


async def main():
    import os

    import yaml
    from datasets import load_dataset

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset_split = config.get("train_split", "train[:100]")
    dataset = load_dataset("gsm8k", "main", split=dataset_split)

    # Setup inference service - single GPU with async engine
    inference_img = kt.Image(image_id="lmsysorg/sglang:latest").run_bash(
        "uv pip install --break-system-packages --system 'git+https://github.com/ClawSeven/sglang.git@dev-dllm#subdirectory=python'"
    )
    inference_compute = kt.Compute(gpus=1, image=inference_img, launch_timeout=1200)

    # Setup training service - distributed across multiple GPUs
    train_img = (
        kt.Image(image_id="nvcr.io/nvidia/ai-workbench/python-cuda126:1.0.2")
        .run_bash(
            "uv pip install --system --break-system-packages torch torchvision torchaudio triton datasets transformers wandb diffusers tiktoken torchdata psutil timm einops safetensors pyyaml"
        )
        .pip_install(["bitsandbytes", "liger-kernel"])
        .set_env_vars(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            }
        )
    )

    # Use number of workers from config or default to 2
    num_workers = config.get("num_training_workers", 3)
    gpus_per_worker = config.get("num_gpus_per_worker", 1)
    train_compute = kt.Compute(
        gpus=gpus_per_worker,
        image=train_img,
        launch_timeout=1200,
        allowed_serialization=["json", "pickle"],
        inactivity_ttl="1h",
    ).distribute("pytorch", workers=num_workers)

    # Deploy services in parallel - Kubetorch handles the orchestration
    print("Deploying services...")
    inference_service_task = kt.cls(SGLang).to_async(
        inference_compute,
        init_args={
            "model_id": config["model_path"],
            "checkpoint_version": 0,
            "config": config,
        },
        get_if_exists=True,
    )
    train_service_task = kt.cls(GRPOTrainer).to_async(
        train_compute, init_args={"config": config}, get_if_exists=True
    )
    inference_service, train_service = await asyncio.gather(
        inference_service_task, train_service_task
    )

    # Enable async mode for non-blocking calls
    inference_service.async_ = True
    train_service.async_ = True

    # Initialize distributed training
    await train_service.setup()

    # Run the async GRPO training loop with config values
    await simple_async_grpo(
        dataset,
        train_service,
        inference_service,
        num_epochs=config.get("num_epochs", 2),
        batch_size=config.get("global_batch_size", 32)
        // config.get("num_generations", 4),
        num_generations=config.get("num_generations", 4),
        checkpoint_interval=config.get("checkpoint_interval", 10),
    )


if __name__ == "__main__":
    asyncio.run(main())
