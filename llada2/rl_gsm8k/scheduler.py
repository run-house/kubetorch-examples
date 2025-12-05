"""Async GRPO Scheduler for coordinating inference and training.

Manages async task scheduling with:
- Backpressure control for inference parallelism
- Training lock for FSDP serialization
- Checkpoint hot-swapping coordination
"""
import asyncio
from typing import Any, List, Optional

import numpy as np


class AsyncGRPOScheduler:
    """Manages async inference/training coordination for GRPO.

    Coordinates the producer-consumer pattern where:
    - Inference generates completions and rewards (producer)
    - Training consumes the data with FSDP serialization (consumer)
    """

    def __init__(
        self,
        train_service,
        agent,
        max_inference_parallel: int = 2,
        max_training_pending: int = 3,
        checkpoint_interval: int = 10,
    ):
        """Initialize scheduler.

        Args:
            train_service: Async training service (kubetorch)
            agent: Agent for generating completions and rewards
            max_inference_parallel: Max concurrent inference requests
            max_training_pending: Max pending training tasks
            checkpoint_interval: Steps between checkpoints
        """
        self.train_service = train_service
        self.agent = agent
        self.max_inference_parallel = max_inference_parallel
        self.max_training_pending = max_training_pending
        self.checkpoint_interval = checkpoint_interval

        self.training_lock = asyncio.Lock()
        self.inference_tasks: List[asyncio.Task] = []
        self.training_tasks: List[asyncio.Task] = []
        self.steps_completed = 0

    async def run(
        self,
        dataset,
        num_epochs: int,
        batch_size: int,
        num_generations: int,
    ):
        """Run the async GRPO training loop.

        Args:
            dataset: Training dataset with 'question' and 'answer' fields
            num_epochs: Number of training epochs
            batch_size: Batch size (number of prompts per step)
            num_generations: Completions per prompt (K for GRPO)
        """
        indices = np.random.permutation(len(dataset))
        total_steps = num_epochs * len(indices) // batch_size

        print(f"Starting async GRPO: {total_steps} steps, batch_size={batch_size}")

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            for i in range(0, len(indices), batch_size):
                await self._manage_backpressure()

                # Get batch data
                batch_indices = indices[i : i + batch_size]
                questions = [dataset[int(idx)]["question"] for idx in batch_indices]
                answers = [dataset[int(idx)]["answer"] for idx in batch_indices]

                # Launch inference and training tasks
                await self._launch_step(questions, answers, num_generations)

        # Wait for all remaining tasks
        await asyncio.gather(*self.training_tasks)
        print("\nTraining complete!")

    async def _manage_backpressure(self):
        """Control parallelism to prevent memory overflow."""
        # Cleanup completed inference tasks
        self._cleanup_done_tasks(self.inference_tasks)

        # Wait if too many inferences running
        while len(self.inference_tasks) >= self.max_inference_parallel:
            await asyncio.sleep(0.1)
            self._cleanup_done_tasks(self.inference_tasks)

        # Cleanup and await completed training tasks
        await self._cleanup_training_tasks()

        # Wait if too many training tasks pending
        while len(self.training_tasks) >= self.max_training_pending:
            await asyncio.sleep(0.1)
            await self._cleanup_training_tasks()

    def _cleanup_done_tasks(self, task_list: List[asyncio.Task]):
        """Remove completed tasks from list."""
        done = [t for t in task_list if t.done()]
        for task in done:
            task_list.remove(task)

    async def _cleanup_training_tasks(self):
        """Cleanup training tasks and propagate exceptions."""
        done = [t for t in self.training_tasks if t.done()]
        for task in done:
            self.training_tasks.remove(task)
            await task  # Propagate any exceptions

    async def _launch_step(
        self,
        questions: List[str],
        answers: List[str],
        num_generations: int,
    ):
        """Launch inference task and attach training task."""
        step_num = self.steps_completed + 1

        # Start inference
        inference_task = asyncio.create_task(
            self.agent.generate_batch(
                questions, answers, num_generations, step_num=step_num
            )
        )
        self.inference_tasks.append(inference_task)
        print(f"[SCHEDULER] Launched inference for step {step_num}")

        # Create training task that waits for inference
        training_task = asyncio.create_task(
            self._train_when_ready(inference_task, step_num, num_generations)
        )
        self.training_tasks.append(training_task)
        self.steps_completed += 1

    async def _train_when_ready(
        self,
        inference_task: asyncio.Task,
        step_num: int,
        num_generations: int,
    ) -> int:
        """Wait for inference, then train with lock.

        This is the key coordination function - ensures:
        1. Training waits for inference to complete
        2. Only one training call at a time (FSDP requirement)
        3. Checkpoint saving is synchronized

        Args:
            inference_task: Async task generating completions
            step_num: Current step number
            num_generations: K for GRPO

        Returns:
            Step number
        """
        print(f"[TRAINING] Waiting for inference results for step {step_num}")
        result = await inference_task

        # Remove from inference list
        if inference_task in self.inference_tasks:
            self.inference_tasks.remove(inference_task)

        # Check for stale result
        if result[0] is None:
            print(f"[TRAINING] Skipping stale request at step {step_num}")
            return step_num

        prompts, completions, token_ids, rewards = result

        # Acquire lock for training (critical for FSDP)
        async with self.training_lock:
            print(f"[TRAINING] Starting step {step_num} (lock acquired)")

            # Train
            metrics = await self.train_service.train_batch(
                prompts, completions, token_ids, rewards, num_generations
            )

            # Validate metrics
            if not metrics or not isinstance(metrics, list) or len(metrics) == 0:
                raise RuntimeError(f"Training failed for step {step_num}: {metrics}")

            rank0_result = next((m for m in metrics if m.get("metrics")), None)
            if not rank0_result:
                raise RuntimeError(f"No valid metrics for step {step_num}: {metrics}")

            print(
                f"[TRAINING] Completed step {step_num}: "
                f"loss={rank0_result['metrics']['loss']:.4f}, "
                f"reward={rank0_result['metrics']['reward_mean']:.3f}"
            )

            # Checkpoint if needed
            if step_num % self.checkpoint_interval == 0:
                await self._save_and_hotswap_checkpoint(step_num)

        return step_num

    async def _save_and_hotswap_checkpoint(self, step_num: int):
        """Save checkpoint and hot-swap inference weights."""
        print(f"[CHECKPOINT] Saving checkpoint at step {step_num}...")

        # Save checkpoint
        checkpoint_results = await self.train_service.save_checkpoint()
        checkpoint_result = (
            checkpoint_results[0]
            if isinstance(checkpoint_results, list)
            else checkpoint_results
        )
        key, new_version, checkpoint_folder = checkpoint_result
        print(f"[CHECKPOINT] Saved: {key}")

        # Hot-swap inference
        print(f"[CHECKPOINT] Hot-swapping to v{new_version}...")
        try:
            await self.agent.inference_service.update_weights_from_disk(
                key, new_version, checkpoint_folder
            )
            print(f"[CHECKPOINT] Hot-swap complete: v{new_version}")
        except Exception as e:
            print(f"[CHECKPOINT] FATAL: Hot-swap failed: {e}")
            raise RuntimeError(f"Checkpoint hot-swap failed at step {step_num}") from e
