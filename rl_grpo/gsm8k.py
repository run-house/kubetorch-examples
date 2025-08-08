import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kubetorch as kt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class vLLM:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        import os
        import subprocess

        from vllm import LLM

        # Kill any existing vLLM process using GPU - we need this for redeployment
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            for line in result.stdout.split("\n"):
                if (
                    "python" in line.lower() and "C" in line
                ):  # C indicates compute process
                    elems = line.split()
                    pid = elems[elems.index("C") - 1]  # Pick the element before "C"
                    if pid.isdigit():
                        print(f"Killing existing vLLM process: {pid}")
                        os.system(f"kill -9 {pid}")
        except:
            pass  # nvidia-smi not available or no GPU processes

        print("Loading model in vLLM:", model_id)
        self.model_id = model_id
        self.model = LLM(
            self.model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,  # Reduces size of KV store
            # enforce_eager=True,
        )

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

        all_outputs = self.model.generate(queries, sampling_params)
        completions = [
            output.text for outputs in all_outputs for output in outputs.outputs
        ]
        token_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        return completions, token_ids


class MathAgent:
    def __init__(self, inference_service: vLLM = None):
        self.inference_service = inference_service
        self.system_prompt = (
            "You are a helpful math assistant. "
            "Solve the following problem step by step. "
            "Show your work and reasoning. "
            "End your response with '#### [Final Answer]' where [Final Answer] is just the numerical answer."
        )

    def answer(self, questions: List[str], temperature=0.7, max_tokens=512):
        """Generate answers for a batch of questions."""
        prompts = [
            f"{self.system_prompt}\n\nQuestion: {q}\n\nSolution:" for q in questions
        ]
        completions, token_ids = self.inference_service.generate(
            prompts, max_tokens=max_tokens, temperature=temperature, top_p=0.95
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


def pad_right(tensors: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """Pad a list of tensors to the same shape on the right."""
    if not tensors:
        return torch.tensor([])

    # Find max shape
    max_shape = [
        max(t.shape[i] if i < len(t.shape) else 0 for t in tensors)
        for i in range(max(len(t.shape) for t in tensors))
    ]

    # Create output tensor
    output = torch.full(
        (len(tensors), *max_shape),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    # Fill with actual values
    for i, t in enumerate(tensors):
        slices = tuple(slice(0, s) for s in t.shape)
        output[i][slices] = t

    return output


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
        if os.environ.get("RANK", None) != "0":
            return

        self.latest_checkpoint = f"qwen-grpo-checkpoint-{self.steps}-steps"
        Path(self.latest_checkpoint).mkdir(parents=True, exist_ok=True)

        # Get the underlying model (unwrap DDP if needed)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        model_to_save.save_pretrained(self.latest_checkpoint)
        self.tokenizer.save_pretrained(self.latest_checkpoint)
        print(f"Checkpoint saved at {self.latest_checkpoint}")
        return {"checkpoint_path": self.latest_checkpoint}

    def deploy_inference_service(self, inference_service):
        """Redeploy the inference service with the latest model checkpoint."""
        if os.environ.get("RANK", None) != "0":
            return

        if not self.latest_checkpoint:
            print("No checkpoint to deploy")
            return

        # We do this from the training service so we can sync the latest checkpoint into the inference service
        # image. We could also just save it to a shared storage location like blob storage and redeploy from within
        # the AsyncGRPOPipeline.
        inference_service.compute.image.rsync(
            source=self.latest_checkpoint, dest=self.latest_checkpoint
        )
        init_args = {
            "model_id": self.latest_checkpoint.replace("/", "-"),
        }

        service = inference_service.to(inference_service.compute, init_args=init_args)
        return service


class AsyncGRPOPipeline:
    """Async pipeline for GRPO training with GSM8K."""

    def __init__(
        self,
        inference_service,
        train_service,
        agent,
        batch_size=32,
        num_generations=4,  # Number of completions per prompt
        max_workers=1,
    ):  # Reduced to 1 to avoid vLLM concurrency issues
        self.inference_service = inference_service
        self.train_service = train_service
        self.agent = agent
        self.batch_size = batch_size
        self.num_generations = num_generations
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize train service
        self.train_service.setup()

    def generate_completions(
        self, prompts: List[str], answers: List[str]
    ) -> Tuple[List[str], List[List[int]], List[float]]:
        """Generate completions and calculate rewards for a batch."""
        # Expand prompts for multiple generations
        expanded_prompts = []
        expanded_answers = []
        for p, a in zip(prompts, answers):
            expanded_prompts.extend([p] * self.num_generations)
            expanded_answers.extend([a] * self.num_generations)

        # Generate completions using inference service
        completions, token_ids = self.inference_service.generate(
            expanded_prompts, max_tokens=512, temperature=0.7, top_p=0.95
        )

        # Calculate rewards locally
        rewards = self.agent.calculate_rewards(
            expanded_prompts, completions, expanded_answers
        )

        return completions, token_ids, rewards

    def train_epoch(self, dataset, num_batches=None):
        """Run one epoch of training."""
        # Prepare data loader
        indices = np.random.permutation(len(dataset))
        if num_batches:
            indices = indices[: num_batches * self.batch_size]

        # Process in batches
        futures = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_prompts = [dataset[int(idx)]["question"] for idx in batch_indices]
            batch_answers = [dataset[int(idx)]["answer"] for idx in batch_indices]

            # Submit async generation task
            future = self.executor.submit(
                self.generate_completions, batch_prompts, batch_answers
            )
            futures.append((future, batch_prompts))

        # Process completed batches
        total_loss = 0
        total_reward = 0
        num_batches_processed = 0

        for future, original_prompts in futures:
            try:
                # Get generation results
                completions, token_ids, rewards = future.result(timeout=300)

                # Expand original prompts to match completions
                expanded_prompts = []
                for p in original_prompts:
                    expanded_prompts.extend([p] * self.num_generations)

                # Train on batch using train service (single call)
                metrics = self.train_service.train_batch(
                    prompts=expanded_prompts,
                    completions=completions,
                    completion_ids=token_ids,
                    rewards=rewards,
                    num_generations=self.num_generations,
                )[
                    0
                ]  # Only need metrics from one worker

                total_loss += metrics["loss"]
                total_reward += metrics["reward_mean"]
                num_batches_processed += 1

                print(
                    f"Batch {num_batches_processed}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"TokenLoss={metrics['token_loss']:.4f}, "
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
            "avg_loss": avg_loss if num_batches_processed > 0 else None,
            "avg_reward": avg_reward if num_batches_processed > 0 else None,
            "num_batches": num_batches_processed,
        }


def main():
    from datasets import load_dataset

    # Configuration - Further reduced for memory
    batch_size = 2  # Very small batch size
    num_generations = 2  # Reduced generations
    num_epochs = 3
    batches_per_epoch = 3  # Fewer batches for testing

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    # Setup inference service
    print("Setting up inference service...")
    inference_gpus = kt.Compute(
        gpus="1",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:24.10-py3").run_bash(
            "uv pip install --system --break-system-packages --no-build-isolation flash-attn==2.7.3 vllm==0.9.0 'transformers<4.54.0'"
        ),
        shared_memory_limit="2Gi",
        launch_timeout=1200,
        secrets=["huggingface"],
    ).autoscale(initial_scale=1, min_scale=1, max_scale=5, concurrency=64)

    inference_service = kt.cls(vLLM).to(inference_gpus, get_if_exists=True)

    # Test inference service
    print("Testing inference service...")
    test_response = inference_service.generate(["What is 2+2?"], max_tokens=50)
    print(f"Test response: {test_response[0]}")

    # Setup training service
    print("Setting up training service...")
    train_gpus = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:24.10-py3").run_bash(
            "uv pip install --system --break-system-packages 'torch>=2.2.0' transformers datasets accelerate"
        ),
        launch_timeout=600,
        allowed_serialization=["pickle", "json"],
    ).distribute("pytorch", workers=2)

    train_service = kt.cls(GRPOTrainer).to(train_gpus)

    # Create pipeline
    agent = MathAgent()
    pipeline = AsyncGRPOPipeline(
        inference_service=inference_service,
        train_service=train_service,
        agent=agent,
        batch_size=batch_size,
        num_generations=num_generations,
        max_workers=1,
    )

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # Train for one epoch
        epoch_metrics = pipeline.train_epoch(dataset, num_batches=batches_per_epoch)

        # Save checkpoint periodically
        if (epoch + 1) % 1 == 0 and epoch_metrics["num_batches"] > 0:
            print(f"Epoch {epoch + 1} complete. Saving checkpoint.")
            train_service.save_checkpoint()

            # Optionally redeploy inference service with new checkpoint
            if epoch < num_epochs - 1:  # Don't redeploy on last epoch
                print("Redeploying inference service with new checkpoint.")
                train_service.deploy_inference_service(
                    inference_service, serialization="pickle"
                )
                # time.sleep(30)  # Wait for service to restart

    print("\nTraining complete!")

    # Final checkpoint
    train_service.save_checkpoint()

    # Cleanup
    pipeline.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
