# # Illustrative RL Training with Kubetorch
# In this example, we do the simplest possible demonstration of Kubetorch's ability
# to do reinforcement learning by defining 3 classes: a `vLLM` class for inference, a
# `MathAgent` class for calculating rewards, and a `GRPOTrainer` class to do PyTorch
# DDP training. Each of these components are vanilla and called in a loop in sequence
# by `SyncGRPOPipeline`.
#
# Finally, in `main`, we launch each of the services. Inference and training launch very
# differently: while training is launched with PyTorch distribution wired up (and any calls made
# to the launched service is vectorized over it's replicas), inference is launched as a autoscaling
# service with different underlying image.
#
# Note: this is not an optimized RL training example, which takes place in other examples.
# Instead, this is simplified, synchronous loops to show how Kubetorch
# lets you define resources and services in code, launch them, and call into each.
import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kubetorch as kt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ## vLLM Inference Class
# A regular inference class with method `generate()` used for inference, with
# an added method to kill any existing / previous vLLM deployments as a convenience.
class vLLM:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        from vllm import LLM

        # Kill any existing vLLM process using GPU
        self.stop_existing_server()

        print("Loading model in vLLM:", model_id)
        self.model_id = model_id
        self.model = LLM(
            self.model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,  # Reduces size of KV store
            enforce_eager=True,  # Disable compilation to avoid hash issues with reloaded checkpoints
        )

    @staticmethod
    def stop_existing_server():
        import os
        import subprocess

        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            for line in result.stdout.split("\n"):
                if "python" in line.lower() and "C" in line:
                    elems = line.split()
                    pid = elems[elems.index("C") - 1]
                    if pid.isdigit():
                        print(f"Killing existing vLLM process: {pid}")
                        os.system(f"kill -9 {pid}")
        except:
            pass  # nvidia-smi not available or no GPU processes

    def generate(
        self,
        queries,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
    ):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            min_p=min_p,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        print(f"Generating response with model_id: {self.model_id}")
        all_outputs = self.model.generate(queries, sampling_params)
        completions = [
            output.text for outputs in all_outputs for output in outputs.outputs
        ]
        token_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        return completions, token_ids


# ## MathAgent Evaluation Class
# A basic evaluation for GSM8K that calls into the inference service and contains
# the method to assign a custom reward to the returned answer.
class MathAgent:
    def __init__(self, inference_service: vLLM = None):
        self.inference_service = inference_service
        self.system_prompt = (
            "You are a helpful math assistant. "
            "Solve the following problem step by step. "
            "Show your work and reasoning. "
            "End your response with '#### <answer>' where <answer> is just the numerical answer. "
            "Do not include any units or the word answer in your response."
        )

    async def answer(
        self, questions: List[str], temperature=0.7, max_tokens=512, top_p=0.95
    ):
        """Generate answers for a batch of questions."""
        prompts = [
            f"{self.system_prompt}\n\nQuestion: {q}\n\nSolution:" for q in questions
        ]
        # Since inference_service.async_ is True, generate returns a coroutine
        completions, token_ids = await self.inference_service.generate(
            prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )
        return completions, token_ids

    def extract_answer(self, completion: str) -> Optional[str]:
        """Extract numerical answer from completion."""
        # Look for pattern #### followed by number
        match = re.search(r"####\s*([-+]?\d*\.?\d+)", completion)
        if match:
            return match.group(1).strip()
        return None

    def calculate_rewards(
        self, questions: List[str], completions: List[str], true_answers: List[str]
    ) -> List[float]:
        """Calculate rewards for completions based on correctness."""
        rewards = []

        for q, c, true_ans in zip(questions, completions, true_answers):
            # Extract predicted answer
            pred_answer = self.extract_answer(c)

            # Extract true answer (GSM8K answers are in format with #### at the end)
            true_match = re.search(r"####\s*([-+]?\d*\.?\d+)", true_ans)
            if true_match:
                true_value = true_match.group(1).strip()
            else:
                true_value = true_ans.strip()

            # Calculate reward
            if pred_answer is None:
                # No answer found - small penalty
                reward = -0.5
            elif pred_answer == true_value:
                # Correct answer - full reward
                reward = 1.0
            else:
                # Wrong answer - penalty
                reward = -0.2

            # Bonus for showing work (having intermediate steps)
            if len(c) > 50 and "=" in c:  # Simple heuristic for showing work
                reward += 0.1

            rewards.append(reward)

        return rewards


# ## Trainer Class
# Encapsulating DDP-based training of a base model, including calculation of
# loss and checkpointing.
class GRPOTrainer:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        learning_rate=1e-5,
        steps_per_epoch=10,
    ):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.steps = 0
        self.latest_checkpoint = None
        self.device = None
        self.pad_token_id = None
        # Set microbatch size for gradient accumulation
        self.microbatch_size = 1  # Process 1 sample at a time to save memory

    def setup(self):
        """Initialize model, tokenizer, and optimizer with memory optimizations."""
        import gc

        # Clear any existing CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # Reduce CPU memory during loading
        )

        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id

        torch.distributed.init_process_group(backend="nccl")
        self.device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        self.model = self.model.to(self.device)

        # Wrap model with DDP
        from torch.nn.parallel import DistributedDataParallel as DDP

        self.model = DDP(
            self.model, device_ids=[self.device], find_unused_parameters=False
        )
        print(f"Distributed training initialized on {self.device}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,  # Add weight decay for regularization
        )

        print(f"Model setup complete on {self.device} with memory optimizations")

    def compute_token_level_loss(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute DrGRPO token-level loss."""
        # Concatenate prompts and completions
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Forward pass
        # Note that we do not take the KL divergence with a reference model here, per DrGRPO
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        # Shift for next-token prediction
        logits = logits[
            :, prompt_ids.size(1) - 1 : -1, :
        ]  # Get logits for completion tokens

        # Compute cross-entropy loss per token
        # Reshape for cross entropy
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = completion_ids.reshape(-1)

        # Compute per-token loss (not reduced)
        token_losses = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).reshape(completion_ids.shape)

        # Apply mask
        masked_losses = token_losses * completion_mask

        # DrGRPO: Weight each token by the advantage (https://arxiv.org/pdf/2503.20783)
        # Expand advantages to match token dimension
        token_advantages = advantages.unsqueeze(-1).expand_as(masked_losses)

        # Token-weighted loss (DrGRPO)
        weighted_token_loss = (
            masked_losses * token_advantages * completion_mask
        ).sum() / completion_mask.sum()

        # Also compute standard sequence-level loss for comparison
        sequence_loss = masked_losses.sum(dim=1).mean()

        return weighted_token_loss, sequence_loss

    def train_batch(
        self,
        prompts: List[str],
        completions: List[str],
        completion_ids: List[List[int]],
        rewards: List[float],
        num_generations: int,
    ) -> Dict:
        """Train on a single batch using DrGRPO with memory-efficient gradient accumulation."""
        import gc

        from torch.amp import autocast

        self.model.train()

        # Clear gradients once at the start
        self.optimizer.zero_grad()

        # Process data
        prompt_encoding = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,  # Limit sequence length to save memory
            return_tensors="pt",
        )
        prompt_ids_tensor = prompt_encoding.input_ids.to(self.device)

        # Calculate advantages
        rewards_array = torch.tensor(rewards).view(-1, num_generations)
        mean_rewards = rewards_array.mean(dim=1, keepdim=True)
        std_rewards = rewards_array.std(dim=1, keepdim=True)
        advantages = (rewards_array - mean_rewards) / (std_rewards + 1e-8)
        advantages_tensor = advantages.view(-1).to(self.device)

        # Pad completion_ids
        max_len = min(
            max(len(ids) for ids in completion_ids), 512
        )  # Limit completion length
        padded_completion_ids = []
        completion_masks = []

        for ids in completion_ids:
            padded = ids[:max_len] + [self.pad_token_id] * max(0, max_len - len(ids))
            mask = [1.0] * min(len(ids), max_len) + [0.0] * max(0, max_len - len(ids))
            padded_completion_ids.append(padded)
            completion_masks.append(mask)

        completion_ids_tensor = torch.tensor(
            padded_completion_ids, dtype=torch.long
        ).to(self.device)
        completion_mask_tensor = torch.tensor(completion_masks, dtype=torch.float).to(
            self.device
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)

        # Process in microbatches to save memory
        batch_size = prompt_ids_tensor.size(0)
        total_loss = 0
        num_microbatches = max(1, batch_size // self.microbatch_size)

        for i in range(0, batch_size, self.microbatch_size):
            end_idx = min(i + self.microbatch_size, batch_size)

            # Get microbatch
            micro_prompt_ids = prompt_ids_tensor[i:end_idx]
            micro_completion_ids = completion_ids_tensor[i:end_idx]
            micro_completion_mask = completion_mask_tensor[i:end_idx]
            micro_advantages = advantages_tensor[i:end_idx]

            # Use bfloat16 autocast for memory efficiency (no GradScaler needed)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                token_loss, seq_loss = self.compute_token_level_loss(
                    micro_prompt_ids,
                    micro_completion_ids,
                    micro_completion_mask,
                    micro_advantages,
                )
                # Scale loss by number of microbatches for proper averaging
                loss = token_loss / num_microbatches

            # Backward pass without gradient scaling (not needed for bfloat16)
            loss.backward()
            total_loss += token_loss.item()

        # Update with gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Clear cache periodically
        if self.steps % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        self.steps += 1

        return {
            "loss": total_loss / num_microbatches,
            "token_loss": total_loss / num_microbatches,
            "seq_loss": seq_loss.item() if "seq_loss" in locals() else 0,
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "advantages_mean": advantages_tensor.mean().item(),
        }

    def save_checkpoint(self):
        """Save the current model checkpoint."""
        checkpoint_path = Path(f"qwen-grpo-checkpoint-{self.steps}-steps")

        # Get the underlying model (unwrap DDP if needed)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        model_to_save.save_pretrained(checkpoint_path.resolve())
        self.tokenizer.save_pretrained(checkpoint_path.resolve())
        self.latest_checkpoint = str(checkpoint_path)
        print(f"Checkpoint saved at {self.latest_checkpoint}")
        return {"checkpoint_path": self.latest_checkpoint}

    def deploy_inference_service(self, inference_service):
        """Redeploy the inference service with the latest model checkpoint."""
        if not self.latest_checkpoint:
            print("No checkpoint to deploy")
            return inference_service

        # We do this from the training service so we can sync the latest checkpoint into the inference service
        # image. We could also just save it to a shared storage location like blob storage and redeploy from within
        # the AsyncGRPOPipeline.
        inference_service.compute.image.rsync(source=self.latest_checkpoint, dest="./")

        print(
            f"Redeploying inference service with checkpoint: {self.latest_checkpoint}"
        )
        init_args = {"model_id": self.latest_checkpoint}
        service = inference_service.to(inference_service.compute, init_args=init_args)
        service.generate(["Test"], max_tokens=10)  # Warm up the service
        return service


# ## Simple synchronous on-policy GRPO training pipeline
# Both the inference and the training services are happening on Kubernetes, "remote" to
# this pipeline, but being called into by this pipeline regularly.
class SyncGRPOPipeline:
    def __init__(
        self,
        train_service,
        agent,
        batch_size=32,
        num_generations=4,  # Number of completions per prompt
    ):
        self.train_service = train_service
        self.agent = agent
        self.batch_size = batch_size
        self.num_generations = num_generations

    async def train_epoch(self, dataset, num_batches=None):
        """Run one epoch of synchronous on-policy training."""
        # Prepare data loader
        indices = np.random.permutation(len(dataset))
        if num_batches:
            indices = indices[: num_batches * self.batch_size]

        total_loss = 0
        total_reward = 0
        num_batches_processed = 0

        # Process batches sequentially for true on-policy training
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_prompts = [dataset[int(idx)]["question"] for idx in batch_indices]
            batch_answers = [dataset[int(idx)]["answer"] for idx in batch_indices]

            # Expand prompts for multiple generations
            expanded_prompts = []
            expanded_answers = []
            for p, a in zip(batch_prompts, batch_answers):
                expanded_prompts.extend([p] * self.num_generations)
                expanded_answers.extend([a] * self.num_generations)

            try:
                # Generate completions (on-policy: using current model)
                completions, token_ids = await self.agent.answer(
                    expanded_prompts,
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                )

                # Calculate rewards
                rewards = self.agent.calculate_rewards(
                    expanded_prompts, completions, expanded_answers
                )

                # Log first batch examples
                if num_batches_processed == 0:
                    for j in range(min(2, len(completions))):
                        print(f"\n--- Example {j+1} ---")
                        print(f"Question: {expanded_prompts[j]}")
                        print(f"Completion: {completions[j]}")
                        print(f"Reward: {rewards[j]}")
                        print(f"True Answer: {expanded_answers[j]}")

                # Train on this batch immediately (on-policy)
                metrics_list = await self.train_service.train_batch(
                    prompts=expanded_prompts,
                    completions=completions,
                    completion_ids=token_ids,
                    rewards=rewards,
                    num_generations=self.num_generations,
                )
                metrics = metrics_list[0]  # Only need metrics from one worker

                total_loss += metrics["loss"]
                total_reward += metrics["reward_mean"]
                num_batches_processed += 1

                print(
                    f"Batch {num_batches_processed}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Reward={metrics['reward_mean']:.4f}"
                )

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        avg_loss = avg_reward = 0
        if num_batches_processed > 0:
            avg_loss = total_loss / num_batches_processed
            avg_reward = total_reward / num_batches_processed
            print(
                f"Epoch complete: Avg Loss={avg_loss:.4f}, Avg Reward={avg_reward:.4f}"
            )

        return {
            "avg_loss": avg_loss,
            "avg_reward": avg_reward,
            "num_batches": num_batches_processed,
        }


# ## Setup and Deploy with Kubetorch
# Here is where we use Kubetorch to take our classes from above and launch
# them on Kubernets. Training is launched with PyTorch Distributed setup, and any calls
# to the service will hit every replica ("multi-controller"); inference is launched with
# autoscaling, as we set the number of replicas as high as 5 with concurrency of 64 per
# replica. Calls to the vLLM remote service/class will route between all the replicas
# without the user needing to be aware of that routing.
async def main():
    from datasets import load_dataset

    # Configuration
    batch_size = 4
    num_generations = 4
    num_epochs = 3
    batches_per_epoch = 1

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    # Setup inference service
    print("Setting up inference service...")
    inference_gpus = kt.Compute(
        gpus="1",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").run_bash(
            "uv pip install --system --break-system-packages --no-deps -r kubetorch-examples/rl_grpo/basic_grpo/requirements-inference.txt"
        ),
        shared_memory_limit="2Gi",
        launch_timeout=1200,
        secrets=["huggingface"],
    ).autoscale(initial_scale=1, min_scale=1, max_scale=5, concurrency=64)

    # Setup training service compute
    print("Setting up training service compute...")
    train_gpus = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").run_bash(
            "uv pip install --system --break-system-packages 'torch>=2.2.0' transformers datasets accelerate"
        ),
        launch_timeout=600,
        allowed_serialization=["pickle", "json"],
    ).distribute("pytorch", workers=2)

    # Deploy services in parallel
    print("Deploying inference and training services...")
    inference_task = kt.cls(vLLM).to_async(inference_gpus)
    train_task = kt.cls(GRPOTrainer).to_async(train_gpus)

    inference_service, train_service = await asyncio.gather(inference_task, train_task)
    inference_service.async_ = True
    train_service.async_ = True

    # Initialize training service
    await train_service.setup()

    agent = MathAgent(inference_service=inference_service)

    # Test inference service
    print("Testing math agent...")
    test_response = await agent.answer(["What is 2+2?"], max_tokens=50)
    print(f"Test response: {test_response[0]}")

    # Create pipeline
    pipeline = SyncGRPOPipeline(
        train_service=train_service,
        agent=agent,
        batch_size=batch_size,
        num_generations=num_generations,
    )

    # Training loop
    print("\nStarting synchronous on-policy GRPO training...")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # Train for one epoch
        epoch_metrics = await pipeline.train_epoch(
            dataset, num_batches=batches_per_epoch
        )

        # Save and redeploy checkpoint after each epoch for on-policy training
        if epoch_metrics["num_batches"] > 0:
            print(f"Epoch {epoch + 1} complete. Saving checkpoint...")
            await train_service.save_checkpoint(workers=[0])

            # Redeploy inference service with new checkpoint for on-policy training
            if epoch < num_epochs - 1:  # Don't redeploy on last epoch
                print("Redeploying inference service with new checkpoint...")
                new_service = (
                    await train_service.deploy_inference_service(
                        inference_service,
                        serialization="pickle",
                        workers=[0],
                    )
                )[0]
                inference_service = new_service
                inference_service.async_ = True
                agent.inference_service = inference_service

    print("\nTraining complete!")

    # Final checkpoint
    await train_service.save_checkpoint()


if __name__ == "__main__":
    asyncio.run(main())
