import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kubetorch as kt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Trajectory:
    """Container for a single trajectory/rollout."""

    prompt: str
    completion: str
    token_ids: List[int]
    reward: float
    checkpoint_version: int
    timestamp: float


class TrajectoryBuffer:
    """Simple async buffer for storing trajectories."""

    def __init__(self, max_size: int = 10000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.buffer = []  # For accumulating batch

    async def add(self, trajectories: List[Trajectory]):
        """Add trajectories to the buffer."""
        for traj in trajectories:
            await self.queue.put(traj)

    async def sample(
        self, batch_size: int, timeout: float = 120.0
    ) -> Optional[List[Trajectory]]:
        """Get a batch from the buffer."""
        batch = []
        deadline = time.time() + timeout

        # Try to get batch_size items
        for i in range(batch_size):
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # Return what we got (could be partial batch or empty)
        if batch:
            print(f"Sampled batch of {len(batch)} trajectories")
            return batch
        return None

    def size(self) -> int:
        """Get current buffer size."""
        return self.queue.qsize()


class vLLM:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        lora_checkpoint=None,
        kt_cached_state=None,
    ):
        """Initialize vLLM with optional cached state from Kubetorch.

        Args:
            model_id: The base model ID to use
            lora_checkpoint: Path to LoRA checkpoint to load
            kt_cached_state: Cached state from previous instance (provided by Kubetorch upon redeploy)
        """
        import os

        from vllm import LLM

        print(
            f"Initializing vLLM with model_id={model_id}, lora_checkpoint={lora_checkpoint}"
        )
        self.model_id = model_id
        self.base_model_id = model_id  # Keep track of base model
        self.current_lora_request = None

        # Check if we have cached state from Kubetorch
        if kt_cached_state and "model" in kt_cached_state:
            print("Reusing existing vLLM engine from Kubetorch cached state")
            self.model = kt_cached_state["model"]

            # If a new LoRA checkpoint is provided, hot-swap it
            if lora_checkpoint and os.path.exists(lora_checkpoint):
                print(f"Hot-swapping LoRA adapter to: {lora_checkpoint}")
                self.load_lora_adapter(lora_checkpoint)
            return

        # First time initialization - create the vLLM engine
        print("Creating new vLLM engine with LoRA support")
        self.model = LLM(
            self.model_id,  # Always use base model for LoRA
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,
            enforce_eager=True,
            enable_lora=True,  # Always enable LoRA support
            max_lora_rank=32,  # Maximum LoRA rank
            max_loras=4,  # Can load multiple LoRA adapters
        )

        # If a LoRA checkpoint was provided and exists, load it
        if lora_checkpoint and os.path.exists(lora_checkpoint):
            print(f"Loading LoRA checkpoint: {lora_checkpoint}")
            self.load_lora_adapter(lora_checkpoint)

    def __kt_cached_state__(self):
        """Return state to be cached by Kubetorch across reloads.

        This method is called by Kubetorch before reloading the class.
        The returned dictionary will be passed to the new instance's __init__
        via the kt_cached_state parameter.
        """
        # Preserve the vLLM engine
        return {"model": self.model}

    def load_lora_adapter(self, lora_path, lora_id=None):
        """Load a LoRA adapter without restarting the server - fast hot-swap."""
        if lora_id is None:
            # Use checkpoint version from path if available
            import re

            version_match = re.search(r"v(\d+)", lora_path)
            if version_match:
                lora_id = f"adapter_v{version_match.group(1)}"
            else:
                lora_id = f"adapter_{hash(lora_path)}"

        print(f"Hot-swapping LoRA adapter from {lora_path} with ID {lora_id}")

        # For newer vLLM versions (>0.4.0)
        from vllm.lora.request import LoRARequest

        self.current_lora_request = LoRARequest(
            lora_name=lora_id,
            lora_int_id=hash(lora_id) % 100000,  # Unique int ID
            lora_local_path=lora_path,
        )
        self.model_id = lora_path  # Update model_id to track current adapter
        print(f"LoRA adapter hot-swapped with ID {lora_id}")

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

        # If we have a LoRA adapter loaded, use it
        if self.current_lora_request:
            all_outputs = self.model.generate(
                queries, sampling_params, lora_request=self.current_lora_request
            )
        else:
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
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream_logs=False,
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


class GRPOTrainer:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        learning_rate=1e-5,
        steps_per_epoch=10,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    ):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.steps = 0
        self.latest_checkpoint = None
        self.checkpoint_version = 0
        self.device = None
        self.pad_token_id = None
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Set microbatch size for gradient accumulation
        # With LoRA we can process larger batches
        self.microbatch_size = 4

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

        # Always use LoRA in async version
        from peft import get_peft_model, LoraConfig, TaskType

        print(f"Setting up LoRA with r={self.lora_r}, alpha={self.lora_alpha}")

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Qwen2.5 modules
            bias="none",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Enable gradient checkpointing with LoRA
        self.model.enable_input_require_grads()
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

        # For LoRA, wrap with DDP without find_unused_parameters
        self.model = DDP(
            self.model, device_ids=[self.device], find_unused_parameters=False
        )
        print(f"Distributed training initialized on {self.device}")

        # Only optimize trainable parameters (much faster with LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01,  # Add weight decay for regularization
        )

        print(f"Model setup complete on {self.device} with LoRA training")

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

            loss = None
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
        """Save the current model checkpoint with versioning."""
        self.checkpoint_version += 1

        # Save LoRA adapters only (much smaller and faster)
        checkpoint_path = Path(
            f"qwen-lora-checkpoint-v{self.checkpoint_version}-{self.steps}-steps"
        )

        # Get the underlying model (unwrap DDP if needed)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # Save only the LoRA adapters - this will create the proper adapter_config.json
        model_to_save.save_pretrained(checkpoint_path.resolve())

        # The save_pretrained method above already creates a proper adapter_config.json
        # with all required fields (r, target_modules, etc.) from the PEFT model.
        # We can add our custom metadata to a separate file if needed.
        metadata = {
            "checkpoint_version": self.checkpoint_version,
            "base_model": self.model_id,
        }
        with open(checkpoint_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.latest_checkpoint = str(checkpoint_path)
        print(
            f"LoRA checkpoint v{self.checkpoint_version} saved at {self.latest_checkpoint}"
        )
        return {
            "checkpoint_path": self.latest_checkpoint,
            "version": self.checkpoint_version,
        }

    def deploy_inference_service(self, inference_service):
        """Redeploy the inference service with the latest model checkpoint."""
        if not self.latest_checkpoint:
            print("No checkpoint to deploy")
            return

        # We do this from the training service so we can sync the latest checkpoint into the inference service
        # image. We could also just save it to a shared storage location like blob storage and redeploy from within
        # the AsyncOffPolicyGRPO.
        inference_service.compute.image.rsync(source=self.latest_checkpoint, dest="./")

        print(
            f"Redeploying inference service with checkpoint: {self.latest_checkpoint}"
        )

        # For LoRA, pass the adapter path for hot-swapping
        init_args = {
            "model_id": self.model_id,  # Keep base model
            "lora_checkpoint": self.latest_checkpoint,  # Pass LoRA checkpoint for hot-swap
        }

        service = inference_service.to(inference_service.compute, init_args=init_args)
        return service


class AsyncOffPolicyGRPO:
    """Async off-policy GRPO with parallel inference and training."""

    def __init__(
        self,
        train_service,
        agent,
        buffer: TrajectoryBuffer,
        batch_size=32,
        num_generations=4,
        checkpoint_update_interval=100,  # Update inference checkpoint every N training steps
        min_buffer_size=100,  # Try to maintain at least this many trajectories in buffer
        max_concurrent_batches=4,  # Maximum number of concurrent inference batches
    ):
        self.train_service = train_service
        self.agent = agent
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_generations = num_generations
        self.checkpoint_update_interval = checkpoint_update_interval
        self.current_checkpoint_version = 0
        self.training_steps = 0
        self.should_stop = False
        self.min_buffer_size = min_buffer_size
        self.max_concurrent_batches = max_concurrent_batches

    async def process_batch(self, dataset, indices, start_idx, end_idx):
        """Process a single batch of inference."""
        batch_indices = indices[start_idx:end_idx]
        batch_prompts = [dataset[int(idx)]["question"] for idx in batch_indices]
        batch_answers = [dataset[int(idx)]["answer"] for idx in batch_indices]

        # Expand for multiple generations
        expanded_prompts = []
        expanded_answers = []
        for p, a in zip(batch_prompts, batch_answers):
            expanded_prompts.extend([p] * self.num_generations)
            expanded_answers.extend([a] * self.num_generations)

        # Generate completions
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

        # Create trajectory objects
        trajectories = []
        for prompt, completion, ids, reward in zip(
            expanded_prompts, completions, token_ids, rewards
        ):
            traj = Trajectory(
                prompt=prompt,
                completion=completion,
                token_ids=ids,
                reward=reward,
                checkpoint_version=self.current_checkpoint_version,
                timestamp=time.time(),
            )
            trajectories.append(traj)

        return trajectories

    async def inference_loop(self, dataset, num_batches=None):
        """Continuously generate trajectories with adaptive request rate."""
        print("Starting adaptive inference loop...")
        indices = np.random.permutation(len(dataset))
        if num_batches:
            indices = indices[: num_batches * self.batch_size]

        batch_idx = 0
        pending_tasks = set()

        while not self.should_stop:
            buffer_size = self.buffer.size()

            # Adaptively decide how many concurrent batches to run
            # If buffer is low, increase concurrent requests to trigger autoscaling
            if buffer_size < self.min_buffer_size:
                # Buffer is low - ramp up inference
                target_concurrent = min(
                    self.max_concurrent_batches,
                    max(
                        2,
                        self.min_buffer_size
                        // (self.batch_size * self.num_generations),
                    ),
                )
            else:
                # Buffer is healthy - maintain steady pace
                target_concurrent = 1

            # Launch new batches if needed
            while len(pending_tasks) < target_concurrent and not self.should_stop:
                start_idx = (batch_idx * self.batch_size) % len(indices)
                end_idx = min(start_idx + self.batch_size, len(indices))

                # If we've wrapped around, reshuffle
                if end_idx >= len(indices):
                    indices = np.random.permutation(len(dataset))
                    if num_batches:
                        indices = indices[: num_batches * self.batch_size]
                    start_idx = 0
                    end_idx = min(self.batch_size, len(indices))

                # Create task for this batch
                task = asyncio.create_task(
                    self.process_batch(dataset, indices, start_idx, end_idx)
                )
                pending_tasks.add(task)
                batch_idx += 1

                print(
                    f"Launched inference batch {batch_idx} "
                    f"(concurrent: {len(pending_tasks)}/{target_concurrent}, buffer: {buffer_size})"
                )

            # Wait for at least one task to complete
            if pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for task in done:
                    try:
                        trajectories = await task
                        await self.buffer.add(trajectories)

                        buffer_size = self.buffer.size()
                        print(
                            f"Added {len(trajectories)} trajectories. Buffer size: {buffer_size}"
                        )
                    except Exception as e:
                        print(f"Error processing batch: {e}")
            else:
                # No tasks running, brief pause
                await asyncio.sleep(0.1)

    async def training_loop(self):
        """Continuously train on trajectories from the buffer."""
        print("Starting training loop...")

        while not self.should_stop:
            # Sample batch from buffer
            trajectories = await self.buffer.sample(
                self.batch_size * self.num_generations,
            )

            if trajectories is None:
                print("Training: Timeout waiting for trajectories")
                continue

            # If an incomplete batch came back, make sure they're the right shape that
            # the trainer can still use them (and drop any stragglers)
            if len(trajectories) % self.num_generations != 0:
                # Trim to the largest multiple of num_generations
                aligned_size = (
                    len(trajectories) // self.num_generations
                ) * self.num_generations
                if aligned_size == 0:
                    print(
                        f"Training: Batch too small ({len(trajectories)} < {self.num_generations}), skipping"
                    )
                    continue
                print(
                    f"Training: Trimming incomplete batch from {len(trajectories)} to {aligned_size} trajectories"
                )
                trajectories = trajectories[:aligned_size]

            # Unpack trajectories into the format expected by train_batch
            prompts = [t.prompt for t in trajectories]
            completions = [t.completion for t in trajectories]
            completion_ids = [t.token_ids for t in trajectories]
            rewards = [t.reward for t in trajectories]

            # Train on batch
            metrics_list = await self.train_service.train_batch(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_ids,
                rewards=rewards,
                num_generations=self.num_generations,
            )
            metrics = (
                metrics_list[0] if metrics_list else {}
            )  # Only need metrics from one worker

            self.training_steps += 1

            # Log metrics
            checkpoint_versions = set(t.checkpoint_version for t in trajectories)
            if "loss" in metrics:
                print(
                    f"Training step {self.training_steps}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Reward={metrics.get('reward_mean', 0):.4f}, "
                    f"Checkpoint versions in batch: {checkpoint_versions}"
                )
            else:
                print(
                    f"Training step {self.training_steps}: "
                    f"Metrics: {metrics}, "
                    f"Checkpoint versions in batch: {checkpoint_versions}"
                )

            # Periodically save and update inference checkpoint
            if self.training_steps % self.checkpoint_update_interval == 0:
                print(f"Saving checkpoint after {self.training_steps} steps...")
                checkpoint_info_list = await self.train_service.save_checkpoint(
                    workers=[0]
                )
                checkpoint_info = checkpoint_info_list[0]

                # We only want the first value back
                if checkpoint_info:
                    # Update the inference service with new checkpoint
                    await self.update_inference_checkpoint(checkpoint_info)

    async def update_inference_checkpoint(self, checkpoint_info):
        """Update the inference service with a new checkpoint."""
        new_version = checkpoint_info["version"]

        print(f"Updating inference service to checkpoint v{new_version}...")

        # Call the training service's deploy method (which runs on the training node and can rsync)
        new_service = (
            await self.train_service.deploy_inference_service(
                self.agent.inference_service,
                serialization="pickle",
                workers=[0],
            )
        )[
            0
        ]  # spmd calls always return lists, so grab the first element for rank 0's response

        await new_service.generate(["Test"], max_tokens=10)  # Warm up the service

        # Update our reference to the new service
        self.agent.inference_service = new_service
        self.agent.inference_service.async_ = True

        # Update version tracker
        self.current_checkpoint_version = new_version

        print(f"Inference service updated to checkpoint v{new_version}")

    async def run(self, dataset, num_epochs=3, batches_per_epoch=10):
        """Run the async off-policy training pipeline."""
        print("Starting async off-policy GRPO training...")

        # Start both loops concurrently
        inference_task = asyncio.create_task(
            self.inference_loop(dataset, num_batches=batches_per_epoch)
        )
        training_task = asyncio.create_task(self.training_loop())

        # Run for specified duration
        target_steps = num_epochs * batches_per_epoch

        try:
            while self.training_steps < target_steps:
                await asyncio.sleep(5)  # Check progress every 5 seconds

                # Check if tasks have failed
                if inference_task.done():
                    # This will raise the exception if the task failed
                    inference_task.result()
                if training_task.done():
                    # This will raise the exception if the task failed
                    training_task.result()

                buffer_size = self.buffer.size()
                print(
                    f"Progress: {self.training_steps}/{target_steps} steps, "
                    f"Buffer size: {buffer_size}"
                )

            # Final checkpoint
            await self.train_service.save_checkpoint()
            print("Training complete!")

        finally:
            # Stop loops
            print("Stopping training...")
            self.should_stop = True

            # Wait for tasks to complete (don't suppress exceptions)
            await asyncio.gather(inference_task, training_task)


async def main():
    from datasets import load_dataset

    # Configuration - increased for LoRA efficiency
    batch_size = 8  # Larger batch with LoRA
    num_generations = 4
    num_epochs = 3
    batches_per_epoch = 10
    buffer_size = 2000

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")

    # Setup inference service
    print("Setting up inference service...")
    inference_gpus = kt.Compute(
        gpus="1",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").run_bash(
            "uv pip install --system --break-system-packages --no-deps -r async_grpo/requirements-inference.txt"
        ),
        shared_memory_limit="2Gi",
        launch_timeout=1200,
        secrets=["huggingface"],
        concurrency=1,
    ).autoscale(
        initial_scale=1,
        min_scale=1,
        max_scale=5,
        concurrency=1,  # vLLM LLM class doesn't support concurrent requests well so we queue
        target=1,  # Scale up as soon as there's a queue (aggressive scaling)
    )

    # Setup training service compute
    print("Setting up training service compute...")
    train_gpus = kt.Compute(
        gpus=1,
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3").run_bash(
            "uv pip install --system --break-system-packages 'torch>=2.2.0' transformers datasets accelerate peft"
        ),
        launch_timeout=600,
        allowed_serialization=["pickle", "json"],
    ).distribute("pytorch", workers=2)

    # Deploy services in parallel
    print("Deploying inference and training services in parallel...")
    inference_task = kt.cls(vLLM).to_async(inference_gpus)
    train_task = kt.cls(GRPOTrainer).to_async(train_gpus)

    inference_service, train_service = await asyncio.gather(inference_task, train_task)
    inference_service.async_ = True
    train_service.async_ = True

    # Initialize training service
    await train_service.setup()

    # Create components
    agent = MathAgent(inference_service=inference_service)
    buffer = TrajectoryBuffer(max_size=buffer_size)

    # Test inference
    print("Testing math agent...")
    test_response = await agent.answer(["What is 2+2?"], max_tokens=50)
    print(f"Test response: {test_response[0]}")

    # Create and run pipeline
    pipeline = AsyncOffPolicyGRPO(
        train_service=train_service,
        agent=agent,
        buffer=buffer,
        batch_size=batch_size,
        num_generations=num_generations,
        checkpoint_update_interval=5,  # Update checkpoint every 5 steps for testing
        min_buffer_size=100,  # Try to maintain at least 100 trajectories
        max_concurrent_batches=2,  # Start conservatively with 2 concurrent batches
    )

    await pipeline.run(
        dataset, num_epochs=num_epochs, batches_per_epoch=batches_per_epoch
    )


if __name__ == "__main__":
    asyncio.run(main())
