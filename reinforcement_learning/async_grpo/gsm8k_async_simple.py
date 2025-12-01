# # Async GRPO Training with Kubetorch
# This example demonstrates a simplified async GRPO implementation that showcases
# Kubetorch's ability to orchestrate parallel training and inference services.
# Unlike the synchronous version, this runs inference and training simultaneously,
# allowing for better resource utilization and faster iteration.
#
# The key components are:
# - A `vLLM` class with AsyncLLMEngine for automatic request batching
# - A `SimpleMathAgent` for reward calculation with version tracking
# - A `GRPOTrainer` with LoRA for memory-efficient training
# - An async pipeline that naturally handles parallel execution
#
# The async design allows inference to run ahead while training catches up,
# with automatic hot-swapping of checkpoints and stale request filtering.

import asyncio
import os
import re
from pathlib import Path

import kubetorch as kt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ## vLLM Inference Class with AsyncLLMEngine
# This class wraps vLLM's AsyncLLMEngine for efficient async inference.
# Key features:
# - Uses AsyncLLMEngine for automatic request batching and memory management
# - Supports hot-swapping of LoRA adapters without restarting the engine
# - Implements version tracking to filter stale requests after checkpoint updates
# - Caches the engine across redeployments for fast checkpoint switching
class vLLM:
    """Simple vLLM wrapper with hot-swapping support using AsyncLLMEngine."""

    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        lora_checkpoint=None,
        checkpoint_version=0,
        kt_cached_state=None,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.current_lora_request = None
        self.checkpoint_version = checkpoint_version
        self.model_id = model_id

        # Reuse cached engine if available (for hot-swapping)
        if kt_cached_state and kt_cached_state.get("model") is not None:
            print(f"Reusing AsyncLLMEngine from cache (version {self.checkpoint_version})")
            self.model = kt_cached_state["model"]
            if lora_checkpoint and os.path.exists(lora_checkpoint):
                self.load_lora_adapter(lora_checkpoint)
            return

        # Create new engine if not cached
        print(f"Creating new AsyncLLMEngine (version {self.checkpoint_version})")

        # Configure engine args
        engine_args = AsyncEngineArgs(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,
            enforce_eager=True,
            enable_lora=True,
            max_lora_rank=32,
        )

        # Create async engine
        self.model = AsyncLLMEngine.from_engine_args(engine_args)

        if lora_checkpoint and os.path.exists(lora_checkpoint):
            self.load_lora_adapter(lora_checkpoint)

    def __kt_cached_state__(self):
        """Return state to be cached by Kubetorch across reloads.

        This method is called by Kubetorch before reloading the class.
        The returned dictionary will be passed to the new instance's __init__
        via the kt_cached_state parameter.
        """
        # Preserve the AsyncLLMEngine for hot-swapping
        return {"model": self.model}

    def load_lora_adapter(self, lora_path):
        """Hot-swap LoRA adapter without restarting."""
        from vllm.lora.request import LoRARequest

        lora_id = f"adapter_{hash(lora_path)}"
        self.current_lora_request = LoRARequest(
            lora_name=lora_id,
            lora_int_id=hash(lora_id) % 100000,
            lora_local_path=lora_path,
        )
        print(f"LoRA adapter loaded from {lora_path}")

    async def generate(self, prompts, request_version=None, **kwargs):
        import asyncio
        import uuid

        from vllm import SamplingParams

        # Check if this request is from an old checkpoint version
        if request_version is not None and request_version != self.checkpoint_version:
            print(f"Ignoring stale request from version {request_version} (current: {self.checkpoint_version})")
            # Return empty results for stale requests
            return [""] * len(prompts), [[]] * len(prompts)

        sampling_params = SamplingParams(**kwargs)

        # Create tasks for all prompts to run in parallel
        async def process_single_prompt(prompt):
            request_id = str(uuid.uuid4())

            # Generate for this single request
            result_generator = self.model.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=self.current_lora_request if self.current_lora_request else None,
            )

            # Collect the final result
            async for output in result_generator:
                if output.finished:
                    return output
            return None

        # Process all prompts in parallel
        tasks = [process_single_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        # Extract completions and token IDs
        completions = []
        token_ids = []
        for result in results:
            if result:
                completions.append(result.outputs[0].text)
                token_ids.append(result.outputs[0].token_ids)
            else:
                completions.append("")
                token_ids.append([])

        return completions, token_ids


# ## Math Agent Evaluation Class
# This agent handles the evaluation of math problems and reward calculation.
# It calls the inference service to generate solutions and compares them
# against ground truth answers to compute rewards. The agent tracks
# checkpoint versions to ensure it only processes results from the current model.
class SimpleMathAgent:
    """Math problem solver using vLLM."""

    def __init__(self, inference_service, checkpoint_version=0):
        self.inference_service = inference_service
        self.checkpoint_version = checkpoint_version
        self.system_prompt = (
            "You are a helpful math assistant. "
            "Solve the following problem step by step. "
            "End with '#### <answer>' where <answer> is just the number."
        )

    async def generate_batch(self, questions, answers, num_generations=4, step_num=None):
        """Generate multiple completions per question and calculate rewards."""
        if step_num:
            print(f"[INFERENCE] Starting generation for step {step_num} (checkpoint v{self.checkpoint_version})")

        # Expand for multiple generations
        expanded_questions = []
        expanded_answers = []
        for q, a in zip(questions, answers):
            expanded_questions.extend([q] * num_generations)
            expanded_answers.extend([a] * num_generations)

        # Format prompts
        prompts = [f"{self.system_prompt}\n\nQuestion: {q}\n\nSolution:" for q in expanded_questions]

        # Generate completions with version tracking
        completions, token_ids = await self.inference_service.generate(
            prompts,
            request_version=self.checkpoint_version,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )

        if step_num:
            print(f"[INFERENCE] Completed generation for step {step_num}")

        # Check if request was ignored due to being stale
        if all(c == "" for c in completions):
            print(f"Request was stale (version {self.checkpoint_version}), skipping batch")
            return None, None, None, None

        # Calculate rewards
        rewards = []
        for completion, true_answer in zip(completions, expanded_answers):
            # Extract predicted answer
            match = re.search(r"####\s*([-+]?\d*\.?\d+)", completion)
            pred_answer = match.group(1).strip() if match else None

            # Extract true answer
            true_match = re.search(r"####\s*([-+]?\d*\.?\d+)", true_answer)
            true_value = true_match.group(1).strip() if true_match else true_answer.strip()

            # Simple reward: 1.0 for correct, -0.2 for wrong
            reward = 1.0 if pred_answer == true_value else -0.2
            rewards.append(reward)

        print(f"[INFERENCE] Generated {len(completions)} samples.")
        return prompts, completions, token_ids, rewards


# ## GRPO Trainer with LoRA
# This trainer implements Group Relative Policy Optimization (GRPO) using
# LoRA (Low-Rank Adaptation) for memory-efficient training. Key optimizations:
# - LoRA adapters train only ~0.5% of parameters
# - Gradient checkpointing reduces memory during backpropagation
# - Distributed training with PyTorch DDP across multiple GPUs
# - Implements DrGRPO token-level loss for improved training signal
class GRPOTrainer:
    """Simplified GRPO trainer with LoRA."""

    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", learning_rate=1e-5):
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = None
        self.steps = 0
        self.checkpoint_version = 0

    def setup(self):
        """Initialize model with LoRA and memory optimizations."""
        import gc

        from peft import get_peft_model, LoraConfig, TaskType

        # Clear any existing CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Load base model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # Critical for memory efficiency
        )

        # Apply LoRA for memory-efficient training
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Enable gradient checkpointing to save memory
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Distributed setup
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        self.model = self.model.to(self.device)

        from torch.nn.parallel import DistributedDataParallel as DDP

        self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=False)

        # Optimizer for LoRA params only (much fewer parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

        print(f"Trainer setup complete on {self.device} with LoRA training")

    def train_batch(self, prompts, completions, token_ids, rewards, num_generations):
        """Train on a batch using DrGRPO."""
        if not self.model:
            self.setup()
        self.model.train()
        self.optimizer.zero_grad()

        # Tokenize prompts
        prompt_encoding = self.tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        prompt_ids = prompt_encoding.input_ids.to(self.device)

        # Pad completions
        max_len = min(max(len(ids) for ids in token_ids), 512)
        padded_completion_ids = []
        completion_masks = []

        pad_id = self.tokenizer.pad_token_id
        for ids in token_ids:
            padded = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))
            mask = [1.0] * min(len(ids), max_len) + [0.0] * max(0, max_len - len(ids))
            padded_completion_ids.append(padded)
            completion_masks.append(mask)

        completion_ids = torch.tensor(padded_completion_ids, dtype=torch.long).to(self.device)
        completion_mask = torch.tensor(completion_masks, dtype=torch.float).to(self.device)

        # Calculate advantages
        rewards_tensor = torch.tensor(rewards).view(-1, num_generations)
        advantages = (rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)) / (
            rewards_tensor.std(dim=1, keepdim=True) + 1e-8
        )
        advantages = advantages.view(-1).to(self.device)

        # Forward pass
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, prompt_ids.size(1) - 1 : -1, :]

        # Compute DrGRPO loss
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = completion_ids.reshape(-1)

        token_losses = F.cross_entropy(flat_logits, flat_targets, reduction="none").reshape(completion_ids.shape)

        # Weight by advantages (DrGRPO)
        weighted_loss = (token_losses * advantages.unsqueeze(-1) * completion_mask).sum() / completion_mask.sum()

        # Backward and update
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1

        return {
            "loss": weighted_loss.item(),
            "reward_mean": torch.tensor(rewards).mean().item(),
            "reward_std": torch.tensor(rewards).std().item(),
        }

    def save_checkpoint(self):
        """Save LoRA checkpoint."""
        self.checkpoint_version = getattr(self, "checkpoint_version", 0) + 1
        checkpoint_path = Path(f"checkpoint-v{self.checkpoint_version}-{self.steps}")
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(checkpoint_path)
        print(f"Checkpoint v{self.checkpoint_version} saved to {checkpoint_path}")
        return str(checkpoint_path), self.checkpoint_version

    def redeploy_inference(self, inference_service, checkpoint_path):
        """Redeploy inference service with new checkpoint via rsync."""
        # Sync checkpoint to inference service's image
        inference_service.compute.image.rsync(source=checkpoint_path, dest="./")

        # Redeploy with the new checkpoint and version
        new_service = kt.cls(vLLM).to(
            inference_service.compute,
            init_args={
                "lora_checkpoint": checkpoint_path,
                "checkpoint_version": self.checkpoint_version,
            },
        )
        return new_service


# ## Async GRPO Training Pipeline
# This is the core async training loop that orchestrates parallel execution
# of inference and training. Unlike traditional synchronous RL training:
# - Inference runs ahead, generating trajectories asynchronously
# - Training processes results as they become available
# - Checkpoints are hot-swapped without stopping inference
# - Stale requests from old checkpoints are automatically filtered
# The pipeline naturally handles parallelism without explicit coordination.
async def simple_async_grpo(
    dataset,
    train_service,
    inference_service,
    num_epochs=3,
    batch_size=8,
    num_generations=4,
    checkpoint_interval=10,
):
    """
    Simple async GRPO: spawn training tasks as data becomes available.
    No separate loops, no buffer, just natural async flow.
    """
    agent = SimpleMathAgent(inference_service, checkpoint_version=0)
    indices = np.random.permutation(len(dataset))

    training_tasks = []
    inference_tasks = []
    steps_completed = 0
    total_steps = num_epochs * len(indices) // batch_size

    print(f"Starting simple async GRPO for {total_steps} steps")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        for i in range(0, len(indices), batch_size):
            # Get batch data
            batch_indices = indices[i : i + batch_size]
            questions = [dataset[int(idx)]["question"] for idx in batch_indices]
            answers = [dataset[int(idx)]["answer"] for idx in batch_indices]

            # Start inference task
            current_step = steps_completed + 1
            inference_task = asyncio.create_task(
                agent.generate_batch(questions, answers, num_generations, step_num=current_step)
            )
            inference_tasks.append(inference_task)
            print(f"[SCHEDULER] Launched inference for step {current_step}")

            # When inference completes, start training
            async def train_when_ready(inf_task, step_num):
                nonlocal inference_service  # Access the outer scope variable

                print(f"[TRAINING] Waiting for inference results for step {step_num}")
                result = await inf_task
                inference_tasks.remove(inf_task)

                # Check if result was stale and skipped
                if result[0] is None:
                    print(f"[TRAINING] Skipping training for stale request at step {step_num}")
                    return step_num

                prompts, completions, token_ids, rewards = result

                print(f"[TRAINING] Starting training for step {step_num}")
                # Train on this batch
                metrics = await train_service.train_batch(prompts, completions, token_ids, rewards, num_generations)

                print(
                    f"[TRAINING] Completed step {step_num}: loss={metrics[0]['loss']:.4f}, "
                    f"reward={metrics[0]['reward_mean']:.3f}"
                )

                # Save checkpoint periodically
                if step_num % checkpoint_interval == 0:
                    print(f"[CHECKPOINT] Saving checkpoint at step {step_num}")
                    checkpoint_result = (await train_service.save_checkpoint(workers=[0]))[0]
                    checkpoint_path, new_version = checkpoint_result

                    print(f"[CHECKPOINT] Hot-swapping inference service to v{new_version}")
                    # Use training service to redeploy inference with new checkpoint
                    new_service = (
                        await train_service.redeploy_inference(
                            inference_service,
                            checkpoint_path,
                            serialization="pickle",
                            workers=[0],
                        )
                    )[
                        0
                    ]  # SPMD modules always return lists

                    # Update agent's reference and version
                    agent.inference_service = new_service
                    agent.inference_service.async_ = True
                    agent.checkpoint_version = new_version  # Update agent's version
                    inference_service = new_service  # Update outer scope reference
                    print(f"[CHECKPOINT] Successfully hot-swapped to v{new_version}: {checkpoint_path}")

                return step_num

            # Control parallelism by waiting before scheduling new training tasks, but
            # inference can continue running in the background
            # Increase to allow greater parallelism
            while len(training_tasks) >= 2:
                print(
                    f"[SCHEDULER] {len(training_tasks)} training tasks in queue, {len(inference_tasks)} inference tasks running"
                )
                # Wait for any task to complete and clean up finished ones
                done, pending = await asyncio.wait(training_tasks, return_when=asyncio.FIRST_COMPLETED)
                # Remove completed tasks from our list
                for task in done:
                    training_tasks.remove(task)
                    await task  # Ensure any exceptions are raised

            # Create training task that waits for inference
            training_task = asyncio.create_task(train_when_ready(inference_task, steps_completed + 1))
            training_tasks.append(training_task)
            steps_completed += 1

    # Wait for remaining tasks
    await asyncio.gather(*training_tasks)
    print("\nTraining complete!")


# ## Setup and Deploy with Kubetorch
# This section demonstrates how Kubetorch orchestrates distributed services
# on Kubernetes. Key aspects:
# - Inference runs on a single GPU with AsyncLLMEngine managing batching
# - Training runs distributed across 2 GPUs with PyTorch DDP
# - Services communicate asynchronously without blocking each other
# - Kubetorch handles deployment, networking, and lifecycle management
async def main():
    from datasets import load_dataset

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train[:100]")  # Use subset for demo

    # Setup inference service - single GPU with async engine
    inference_compute = kt.Compute(
        gpus="1",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").run_bash(
            "uv pip install --system --break-system-packages --no-deps -r async_grpo/requirements-inference.txt"
        ),
        launch_timeout=1200,
    ).autoscale(
        min_scale=2,
    )
    inference_service_task = kt.cls(vLLM).to_async(inference_compute)

    # Setup training service - distributed across 2 GPUs
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
        launch_timeout=600,
        allowed_serialization=["json", "pickle"],
    ).distribute("pytorch", workers=4)
    train_service_task = kt.cls(GRPOTrainer).to_async(train_compute)

    # Deploy services in parallel - Kubetorch handles the orchestration
    print("Deploying services...")
    inference_service, train_service = await asyncio.gather(inference_service_task, train_service_task)

    # Enable async mode for non-blocking calls
    inference_service.async_ = True
    train_service.async_ = True

    # Initialize distributed training
    await train_service.setup()

    # Run the async GRPO training loop
    await simple_async_grpo(
        dataset,
        train_service,
        inference_service,
        num_epochs=5,
        batch_size=8,
        num_generations=4,
        checkpoint_interval=2,
    )


if __name__ == "__main__":
    asyncio.run(main())
