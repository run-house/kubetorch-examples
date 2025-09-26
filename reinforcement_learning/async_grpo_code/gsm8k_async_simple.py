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
import time

# Removed ThreadPoolExecutor import - using async/await instead
from pathlib import Path
from typing import Any, Dict

import kubetorch as kt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml

from code_sandbox import CodeSandbox
from data import (
    expand_for_multiple_generations,
    extract_dataset_fields,
    extract_test_dataset_fields,
    load_leetcode_dataset,
)
from reward import batch_evaluate, batch_execute_with_rewards
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    AsyncInitMixin,
    calculate_request_size_mb,
    extract_code,
    log_training_metrics,
    print_step_summary,
)


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
        model_id="Qwen/Qwen2.5-3B-Instruct",
        lora_checkpoint=None,
        checkpoint_version=0,
        kt_cached_state=None,
        config=None,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.current_lora_request = None
        self.checkpoint_version = checkpoint_version
        self.model_id = model_id

        # Reuse cached engine if available (for hot-swapping)
        if kt_cached_state and kt_cached_state.get("model") is not None:
            print(
                f"Reusing AsyncLLMEngine from cache (version {self.checkpoint_version})"
            )
            self.model = kt_cached_state["model"]
            if lora_checkpoint and os.path.exists(lora_checkpoint):
                self.load_lora_adapter(lora_checkpoint)
            return

        # Create new engine if not cached
        print(f"Creating new AsyncLLMEngine (version {self.checkpoint_version})")

        # Configure engine args using config values
        compute_config = config.get("compute", {}) if config else {}
        training_config = config.get("training", {}) if config else {}

        engine_args = AsyncEngineArgs(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=compute_config.get(
                "inference_gpu_memory_utilization", 0.95
            ),
            max_model_len=compute_config.get("inference_max_model_len", 1400),
            max_num_seqs=compute_config.get("max_num_seqs", 128),
            enforce_eager=False,
            enable_lora=True,
            max_lora_rank=training_config.get("lora_rank", 64),
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
            print(
                f"Ignoring stale request from version {request_version} (current: {self.checkpoint_version})"
            )
            # Return empty results for stale requests
            return [""] * len(prompts), [[]] * len(prompts)

        sampling_params = SamplingParams(**kwargs)

        # Create tasks for all prompts to run in parallel
        async def process_single_prompt(prompt):
            request_id = str(uuid.uuid4())

            # Generate for this single request
            try:
                result_generator = self.model.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    lora_request=self.current_lora_request
                    if self.current_lora_request
                    else None,
                )

                # Collect the final result
                async for output in result_generator:
                    if output.finished:
                        return output
                return None

            except asyncio.CancelledError:
                raise  # Re-raise cancellation
            except:
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


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ## Code Agent Evaluation Class
# This agent handles the evaluation of LeetCode problems and reward calculation.
# It calls the inference service to generate code solutions and executes them
# against test cases to compute rewards. The agent tracks
# checkpoint versions to ensure it only processes results from the current model.
class SimpleCodeAgent(AsyncInitMixin):
    """LeetCode problem solver using vLLM."""

    def __init__(
        self,
        inference_service,
        checkpoint_version=0,
        max_concurrent_evals=10,
        reward_config=None,
        config=None,
    ):
        super().__init__()
        self.inference_service = inference_service
        self.checkpoint_version = checkpoint_version
        self.system_prompt = ""
        self.code_agent = None  # Will be set by async_init
        self.eval_semaphore = asyncio.Semaphore(max_concurrent_evals)
        self.reward_config = reward_config or {}

        # Store config values for use in methods
        if config:
            training_config = config.get("training", {})
            self.code_timeout = training_config.get("code_timeout", 5)
            self.eval_samples = training_config.get("eval_samples", 100)
            self.max_tokens = training_config.get("max_tokens", 1024)
            self.temperature = training_config.get("temperature", 0.6)
            self.top_p = training_config.get("top_p", 0.95)
            self.eval_temperature = training_config.get("eval_temperature", 0.9)
        else:
            # Default values if no config provided
            self.code_timeout = 5
            self.eval_samples = 100
            self.max_tokens = 1024
            self.temperature = 0.6
            self.top_p = 0.95
            self.eval_temperature = 0.9

    async def _async_init(self):
        """Async initialization implementation"""
        if self.code_agent is None:
            self.code_agent = await self.launch_sandbox()

    async def launch_sandbox(self):
        """Start a CodeAgent to run Python in SWE REX sandbox"""
        cpus = kt.Compute(
            cpus="0.25",
            image=kt.Image().run_bash("uv pip install  --system pandas numpy swe-rex"),
            tolerations=[
                {
                    "key": "gpu",
                    "operator": "Equal",
                    "value": "true",
                    "effect": "PreferNoSchedule",
                }
            ],
        ).autoscale(min_scale=30, max_scale=30, concurrency=1, metric="concurrency")
        agent = await kt.cls(CodeSandbox).to_async(cpus)
        agent.async_ = True
        return agent

    async def generate_batch(
        self, contexts, tasks, tests, entrypoints, num_generations=4, step_num=None
    ):
        """Generate multiple completions per question and calculate rewards."""
        if step_num:
            print(
                f"[INFERENCE] Starting generation for step {step_num} (checkpoint v{self.checkpoint_version})"
            )

        # Expand for multiple generations using utility function
        (
            expanded_contexts,
            expanded_tasks,
            expanded_tests,
            expanded_entrypoints,
        ) = expand_for_multiple_generations(
            contexts, tasks, tests, entrypoints, num_generations
        )

        # Generate completions with version tracking
        print("Awaiting generation")
        for attempt in range(1, 3):
            try:
                completions, token_ids = await asyncio.wait_for(
                    self.inference_service.generate(
                        expanded_tasks,
                        request_version=self.checkpoint_version,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    ),
                    timeout=90,
                )
                break
            except asyncio.CancelledError:
                raise  # Re-raise cancellation
            except Exception as e:
                print(f"Failed generation for step {step_num}: {e}")
                completions, token_ids = None, None

        print("Generation Complete)")
        completions = list(map(extract_code, completions))

        if step_num:
            print(f"[INFERENCE] Completed generation for step {step_num}")

        # Check if request was ignored due to being stale
        if all(c == "" for c in completions):
            print(
                f"Request was stale (version {self.checkpoint_version}), skipping batch"
            )
            return None, None, None, None

        # Calculate rewards using batch execution
        print("*** entering code exec")
        print(f"Processing {len(completions)} completions")

        # Use the batch reward calculation function
        try:
            rewards = await asyncio.wait_for(
                batch_execute_with_rewards(
                    completions,
                    expanded_contexts,
                    expanded_tests,
                    expanded_entrypoints,
                    self.code_agent,
                    self.eval_semaphore,
                    timeout=self.code_timeout,
                    reward_config=self.reward_config,
                ),
                timeout=60,
            )
        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            print(f"batch failed: {e}")
            rewards = [0] * len(completions)

        print("*** exiting code exec")
        print(f"rewards length: {len(rewards)}")
        print(f"rewards: {rewards}")

        print(f"[INFERENCE] Generated {len(completions)} samples.")
        return expanded_tasks, completions, token_ids, rewards

    async def evaluate_accuracy(
        self, test_contexts, test_tasks, test_tests, test_entrypoints, num_samples=100
    ):
        """Evaluate model accuracy on test dataset."""
        print(f"[EVAL] Starting evaluation on {num_samples} test samples")

        # Take a subset for evaluation using configured sample size
        actual_samples = min(num_samples or self.eval_samples, len(test_contexts))
        eval_contexts = test_contexts[:actual_samples]
        eval_tasks = test_tasks[:actual_samples]
        eval_tests = test_tests[:actual_samples]
        eval_entrypoints = test_entrypoints[:actual_samples]

        # Generate single completion per problem for evaluation
        try:
            completions, _ = await asyncio.wait_for(
                self.inference_service.generate(
                    eval_tasks,
                    request_version=self.checkpoint_version,
                    max_tokens=self.max_tokens,
                    temperature=self.eval_temperature,  # Lower temperature for evaluation
                    top_p=self.top_p,
                ),
                timeout=90,
            )
        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            print(f"Failed with {e}")
            completions, _ = None, None
        completions = list(map(extract_code, completions))
        # Check if request was ignored due to being stale
        if all(c == "" for c in completions):
            print(
                f"[EVAL] Request was stale (version {self.checkpoint_version}), skipping evaluation"
            )
            return None

        # Calculate accuracy by running the code
        total = len(eval_contexts)
        print(f"Generated {total} code examples for evaluation")
        # Use batch evaluation function
        try:
            results = await asyncio.wait_for(
                batch_evaluate(
                    completions,
                    eval_contexts,
                    eval_tests,
                    eval_entrypoints,
                    self.code_agent,
                    self.eval_semaphore,
                    timeout=self.code_timeout,
                ),
                timeout=60,
            )
        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception as e:
            print(f"Failed with {e}")
            results = [0] * total
        correct = sum(results)
        accuracy = correct / total
        print(f"[EVAL] Accuracy: {correct}/{total} = {accuracy:.3f}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


# ## GRPO Trainer with LoRA
# This trainer implements Group Relative Policy Optimization (GRPO) using
# LoRA (Low-Rank Adaptation) for memory-efficient training. Key optimizations:
# - LoRA adapters train only ~0.5% of parameters
# - Gradient checkpointing reduces memory during backpropagation
# - Distributed training with PyTorch DDP across multiple GPUs
# - Implements DrGRPO token-level loss for improved training signal
class GRPOTrainer:
    """Simplified GRPO trainer with LoRA."""

    def __init__(self, config=None):
        self.full_config = config or {}  # Store full config
        self.config = (
            config.get("training", {}) if config else {}
        )  # Training-specific config
        self.model_id = self.config.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
        self.learning_rate = self.config.get("learning_rate", 3e-6)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = None
        self.steps = 0
        self.checkpoint_version = 0

        self.wandb_run = None

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
            r=self.config.get("lora_rank", 64),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            target_modules="all-linear",
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Enable gradient checkpointing to save memory
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Distributed setup
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        self.model = self.model.to(self.device)

        from torch.nn.parallel import DistributedDataParallel as DDP

        self.model = DDP(
            self.model, device_ids=[self.device], find_unused_parameters=False
        )

        # Optimizer for LoRA params only (much fewer parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

        if torch.distributed.get_rank() == 0:
            wandb.login(key="798c70cca6cce882ace430aeb3e99b74ed146d2e")

            # Get dataset and compute configs for wandb logging
            dataset_config = self.full_config.get("dataset", {})
            compute_config = self.full_config.get("compute", {})

            self.wandb_run = wandb.init(
                project="async_grpo_leetcode",
                name="qwen2_5_async_grpo_leetcode",
                config={
                    "model": self.model_id,
                    "dataset": dataset_config.get(
                        "dataset_name", "newfacade/LeetCodeDataset"
                    ),
                    "num_epochs": self.config.get("num_epochs", 15),
                    "batch_size": self.config.get("batch_size", 24),
                    "num_generations": self.config.get("num_generations", 4),
                    "learning_rate": self.learning_rate,
                    "lora_rank": self.config.get("lora_rank", 64),
                    "lora_alpha": self.config.get("lora_alpha", 32),
                    "checkpoint_interval": self.config.get("checkpoint_interval", 20),
                    "test_interval": self.config.get("test_interval", 5),
                    "workers": compute_config.get("training_workers", 6),
                },
            )
        print(f"Trainer setup complete on {self.device} with LoRA training")

    def train_batch(self, prompts, completions, token_ids, rewards, num_generations):
        """Train on a batch using DrGRPO with proper data parallelism."""
        start_time = time.time()
        timings = {}

        if not self.model:
            self.setup()
        self.model.train()
        self.optimizer.zero_grad()

        # Split batch across distributed workers at PROMPT level for GRPO correctness
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # Calculate number of prompts (not individual samples)
            # Each prompt generates num_generations samples
            total_samples = len(prompts)
            assert (
                total_samples % num_generations == 0
            ), f"Total samples {total_samples} must be divisible by num_generations {num_generations}"

            total_prompts = total_samples // num_generations
            prompts_per_worker = total_prompts // world_size
            remainder = total_prompts % world_size

            # Handle uneven splits by giving extra prompts to first few workers
            if rank < remainder:
                start_prompt = rank * (prompts_per_worker + 1)
                end_prompt = start_prompt + prompts_per_worker + 1
            else:
                start_prompt = (
                    remainder * (prompts_per_worker + 1)
                    + (rank - remainder) * prompts_per_worker
                )
                end_prompt = start_prompt + prompts_per_worker

            # Convert prompt indices to sample indices (each prompt has num_generations samples)
            start_idx = start_prompt * num_generations
            end_idx = end_prompt * num_generations

            # Split the data for this worker (maintaining complete prompt groups)
            worker_prompts = prompts[start_idx:end_idx]
            worker_completions = completions[start_idx:end_idx]
            worker_token_ids = token_ids[start_idx:end_idx]
            worker_rewards = rewards[start_idx:end_idx]

            print(
                f"[WORKER {rank}] Processing prompts {start_prompt}-{end_prompt-1} ({end_prompt-start_prompt} prompts, {len(worker_prompts)} samples)"
            )

            # Use worker-specific data
            prompts = worker_prompts
            completions = worker_completions
            token_ids = worker_token_ids
            rewards = worker_rewards

        # Tokenize prompts
        tokenize_start = time.time()
        prompt_encoding = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        )
        prompt_ids = prompt_encoding.input_ids.to(self.device)
        timings["tokenize"] = time.time() - tokenize_start

        # Pad completions
        pad_start = time.time()
        max_len = min(max(len(ids) for ids in token_ids), 512)
        padded_completion_ids = []
        completion_masks = []

        pad_id = self.tokenizer.pad_token_id
        for ids in token_ids:
            padded = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))
            mask = [1.0] * min(len(ids), max_len) + [0.0] * max(0, max_len - len(ids))
            padded_completion_ids.append(padded)
            completion_masks.append(mask)

        completion_ids = torch.tensor(padded_completion_ids, dtype=torch.long).to(
            self.device
        )
        completion_mask = torch.tensor(completion_masks, dtype=torch.float).to(
            self.device
        )
        timings["data_prep"] = time.time() - pad_start

        # Calculate advantages
        adv_start = time.time()
        rewards_tensor = torch.tensor(rewards).view(-1, num_generations)
        advantages = (rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)) / (
            rewards_tensor.std(dim=1, keepdim=True) + 1e-8
        )
        advantages = advantages.view(-1).to(self.device)
        timings["advantages"] = time.time() - adv_start

        # Forward pass
        forward_start = time.time()
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, prompt_ids.size(1) - 1 : -1, :]
        timings["forward"] = time.time() - forward_start

        # Compute DrGRPO loss
        loss_start = time.time()
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = completion_ids.reshape(-1)

        token_losses = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).reshape(completion_ids.shape)

        # Weight by advantages (DrGRPO)
        weighted_loss = (
            token_losses * advantages.unsqueeze(-1) * completion_mask
        ).sum() / completion_mask.sum()
        timings["loss_compute"] = time.time() - loss_start

        # Backward and update
        backward_start = time.time()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        timings["backward"] = time.time() - backward_start

        self.steps += 1
        timings["total"] = time.time() - start_time

        # Collect comprehensive metrics
        rewards_tensor = torch.tensor(rewards)
        response_lengths = [len(ids) for ids in token_ids]
        prompt_lengths = [len(p.split()) for p in prompts]

        # Calculate token counts
        prompt_tokens = prompt_ids.numel()  # Total tokens in prompt tensor
        completion_tokens = sum(response_lengths)  # Total tokens in completions
        total_tokens = prompt_tokens + completion_tokens

        # Local metrics for this worker
        local_metrics = {
            "loss": weighted_loss.item(),
            "reward_mean": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "reward_max": rewards_tensor.max().item(),
            "reward_min": rewards_tensor.min().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "response_length_mean": np.mean(response_lengths),
            "prompt_length_mean": np.mean(prompt_lengths),
            "num_samples": len(rewards),
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        # Aggregate metrics across workers for logging
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # Gather metrics from all workers for proper aggregation
            metrics_to_aggregate = [
                "reward_mean",
                "reward_std",
                "reward_max",
                "reward_min",
                "advantages_mean",
                "advantages_std",
                "response_length_mean",
                "prompt_length_mean",
                "num_samples",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
            ]

            aggregated_metrics = local_metrics.copy()

            for metric_name in metrics_to_aggregate:
                metric_tensor = torch.tensor(local_metrics[metric_name]).to(self.device)

                if metric_name in ["reward_max", "advantages_mean"]:
                    # Use max reduction for max values
                    torch.distributed.all_reduce(
                        metric_tensor, op=torch.distributed.ReduceOp.MAX
                    )
                elif metric_name in ["reward_min"]:
                    # Use min reduction for min values
                    torch.distributed.all_reduce(
                        metric_tensor, op=torch.distributed.ReduceOp.MIN
                    )
                elif metric_name in [
                    "num_samples",
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                ]:
                    # Sum totals across workers
                    torch.distributed.all_reduce(
                        metric_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                else:
                    # Average across workers for means and stds
                    torch.distributed.all_reduce(
                        metric_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                    metric_tensor /= world_size

                aggregated_metrics[metric_name] = metric_tensor.item()

            # Only log on rank 0 to avoid duplicate logs
            if rank == 0:
                log_training_metrics(
                    self.steps, aggregated_metrics, timings, self.wandb_run
                )
        else:
            # Single worker case
            log_training_metrics(self.steps, local_metrics, timings, self.wandb_run)

        return local_metrics

    def log_evaluation_metrics(self, eval_results, step_num):
        """Log evaluation metrics to wandb from trainer where it's initialized"""
        if (
            torch.distributed.get_rank() == 0
            and self.wandb_run is not None
            and eval_results
        ):
            eval_metrics = {
                "eval/accuracy": eval_results["accuracy"],
                "eval/correct": eval_results["correct"],
                "eval/total": eval_results["total"],
                "step": step_num,
            }
            self.wandb_run.log(eval_metrics)
            print(f"[WANDB] Logged evaluation metrics for step {step_num}")

    def save_checkpoint(self):
        """Save LoRA checkpoint."""
        self.checkpoint_version = getattr(self, "checkpoint_version", 0) + 1
        checkpoint_path = Path(f"checkpoint-v{self.checkpoint_version}-{self.steps}")
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
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
    test_dataset,
    train_service,
    inference_service,
    config=None,
    num_epochs=3,
    batch_size=8,
    num_generations=4,
    checkpoint_interval=10,
    test_interval=5,
    max_concurrent_training=1,
    max_concurrent_evals=10,
):
    """
    Simple async GRPO: spawn training tasks as data becomes available.
    No separate loops, no buffer, just natural async flow.
    """
    # Create semaphore for limiting concurrent training tasks
    training_semaphore = asyncio.Semaphore(max_concurrent_training)

    # Get reward config from YAML config
    reward_config = config.get("reward", {}) if config else {}

    # Create agent with proper configuration
    agent = SimpleCodeAgent(
        inference_service,
        checkpoint_version=0,
        max_concurrent_evals=max_concurrent_evals,
        reward_config=reward_config,
        config=config,
    )
    agent = await agent.ensure_initialized()  # Properly initialize async components
    indices = np.random.permutation(len(dataset))

    # Extract test data for evaluation using utility function
    (
        test_contexts,
        test_tasks,
        test_tests,
        test_entrypoints,
    ) = extract_test_dataset_fields(test_dataset)

    training_tasks = []
    inference_tasks = []
    steps_completed = 0
    total_steps = num_epochs * len(indices) // batch_size

    print(f"Starting simple async GRPO for {total_steps} steps")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        for i in range(0, len(indices), batch_size):
            # Get batch data using utility function
            batch_indices = indices[i : i + batch_size]
            contexts, tasks, tests, entrypoints = extract_dataset_fields(
                dataset, batch_indices
            )

            # Start inference task
            current_step = steps_completed + 1
            inference_task = asyncio.create_task(
                agent.generate_batch(
                    contexts,
                    tasks,
                    tests,
                    entrypoints,
                    num_generations,
                    step_num=current_step,
                )
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
                    print(
                        f"[TRAINING] Skipping training for stale request at step {step_num}"
                    )
                    return step_num

                prompts, completions, token_ids, rewards = result

                # Use semaphore to limit concurrent training operations
                async with training_semaphore:
                    print(f"[TRAINING] Starting training for step {step_num}")
                    # Train on this batch

                    # Calculate request size using utility function
                    request_data = {
                        "prompts": prompts,
                        "completions": completions,
                        "token_ids": token_ids,
                        "rewards": rewards,
                        "num_generations": num_generations,
                    }
                    size_mb = calculate_request_size_mb(request_data)
                    print(f"Request size: {size_mb:.2f} MB")

                    metrics = await train_service.train_batch(
                        prompts, completions, token_ids, rewards, num_generations
                    )

                    # Print step summary using utility function
                    print_step_summary(step_num, metrics[0])

                    # Save checkpoint periodically
                    if step_num % checkpoint_interval == 0:
                        print(f"[CHECKPOINT] Saving checkpoint at step {step_num}")
                        checkpoint_result = (
                            await train_service.save_checkpoint(workers=[0])
                        )[0]
                        checkpoint_path, new_version = checkpoint_result

                        print(
                            f"[CHECKPOINT] Hot-swapping inference service to v{new_version}"
                        )
                        # Use training service to redeploy inference with new checkpoint
                        redeploy_service_time = time.time()
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
                        print(
                            f"[CHECKPOINT] Successfully hot-swapped to v{new_version}: {checkpoint_path} in {time.time() - redeploy_service_time}"
                        )

                    # Run evaluation periodically
                    if step_num % test_interval == 0:
                        print(f"[EVAL] Running evaluation at step {step_num}")
                        try:
                            eval_results = await agent.evaluate_accuracy(
                                test_contexts,
                                test_tasks,
                                test_tests,
                                test_entrypoints,
                                num_samples=50,
                            )
                            if eval_results:
                                print(
                                    f"[EVAL] Step {step_num} Test Accuracy: {eval_results['accuracy']:.3f}"
                                )
                                # Log evaluation metrics to wandb via trainer
                                await train_service.log_evaluation_metrics(
                                    eval_results, step_num
                                )
                        except Exception as e:
                            print(f"[EVAL] Evaluation failed at step {step_num}: {e}")

                return step_num

            # Control parallelism by waiting before scheduling new training tasks, but
            # inference can continue running in the background
            # Increase to allow greater parallelism
            while len(training_tasks) >= 1:
                print(
                    f"[SCHEDULER] {len(training_tasks)} training tasks in queue, {len(inference_tasks)} inference tasks running"
                )
                # Wait for any task to complete and clean up finished ones
                done, pending = await asyncio.wait(
                    training_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                # Remove completed tasks from our list
                for task in done:
                    training_tasks.remove(task)
                    await task  # Ensure any exceptions are raised

            # Create training task that waits for inference
            training_task = asyncio.create_task(
                train_when_ready(inference_task, steps_completed + 1)
            )
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
    # Load configuration from YAML
    config = load_config("config.yaml")
    dataset_config = config.get("dataset", {})

    # Load datasets using utility function
    dataset, test_dataset = load_leetcode_dataset(
        dataset_config.get("dataset_name", "newfacade/LeetCodeDataset")
    )

    # Setup inference service - single GPU with async engine
    compute_config = config.get("compute", {})
    inference_compute = kt.Compute(
        gpus=compute_config.get("inference_gpus", "1"),
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3")
        .run_bash(
            "uv pip install --system --break-system-packages --no-deps -r async_grpo_code/requirements-inference.txt"
        )
        .pip_install("wandb"),
        launch_timeout=compute_config.get("launch_timeout", 1200),
    ).autoscale(
        min_scale=compute_config.get("inference_min_scale", 2),
    )

    # Setup training service - distributed across multiple GPUs
    train_compute = kt.Compute(
        gpus=compute_config.get("training_gpus", 1),
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:25.04-py3")
        .run_bash(
            "uv pip install --system --break-system-packages 'torch>=2.2.0' "
            "transformers==4.56.1 datasets==4.1.0 accelerate==1.10.1 peft==0.17.1 wandb"
        )
        .set_env_vars({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}),
        launch_timeout=compute_config.get("training_launch_timeout", 600),
        allowed_serialization=["json", "pickle"],
    ).distribute("pytorch", workers=compute_config.get("training_workers", 6))

    # Deploy services in parallel - Kubetorch handles the orchestration
    print("Deploying services...")
    inference_service_task = kt.cls(vLLM).to_async(
        inference_compute, init_args={"config": config}, get_if_exists=True
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

    # Run the async GRPO training loop using configuration
    training_config = config.get("training", {})
    await simple_async_grpo(
        dataset,
        test_dataset,
        train_service,
        inference_service,
        config=config,
        num_epochs=training_config.get("num_epochs", 15),
        batch_size=training_config.get("batch_size", 24),
        num_generations=training_config.get("num_generations", 4),
        checkpoint_interval=training_config.get("checkpoint_interval", 20),
        test_interval=training_config.get("test_interval", 5),
        max_concurrent_training=training_config.get("max_concurrent_training", 1),
        max_concurrent_evals=training_config.get("max_concurrent_evals", 15),
    )

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
