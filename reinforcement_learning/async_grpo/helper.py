def log_training_metrics(step, metrics, timings=None, wandb_run=None):
    import torch

    log_msg = f"step:{step}"

    # Prepare wandb logging dict
    wandb_metrics = {"training/global_step": step}

    # Core training metrics
    if "loss" in metrics:
        log_msg += f" - actor/pg_loss:{metrics['loss']:.6f}"
        wandb_metrics["actor/pg_loss"] = metrics["loss"]
    if "reward_mean" in metrics:
        log_msg += f" - critic/rewards/mean:{metrics['reward_mean']:.6f}"
        wandb_metrics["critic/rewards/mean"] = metrics["reward_mean"]
    if "reward_std" in metrics:
        log_msg += f" - critic/rewards/std:{metrics['reward_std']:.6f}"
        wandb_metrics["critic/rewards/std"] = metrics["reward_std"]
    if "reward_max" in metrics:
        log_msg += f" - critic/rewards/max:{metrics['reward_max']:.6f}"
        wandb_metrics["critic/rewards/max"] = metrics["reward_max"]
    if "reward_min" in metrics:
        log_msg += f" - critic/rewards/min:{metrics['reward_min']:.6f}"
        wandb_metrics["critic/rewards/min"] = metrics["reward_min"]

    # Add advantage statistics if available
    if "advantages_mean" in metrics:
        log_msg += f" - critic/advantages/mean:{metrics['advantages_mean']:.6f}"
        wandb_metrics["critic/advantages/mean"] = metrics["advantages_mean"]
    if "advantages_std" in metrics:
        log_msg += f" - critic/advantages/std:{metrics['advantages_std']:.6f}"
        wandb_metrics["critic/advantages/std"] = metrics["advantages_std"]

    # Response and prompt lengths
    if "response_length_mean" in metrics:
        log_msg += f" - response_length/mean:{metrics['response_length_mean']:.2f}"
        wandb_metrics["response_length/mean"] = metrics["response_length_mean"]
    if "prompt_length_mean" in metrics:
        log_msg += f" - prompt_length/mean:{metrics['prompt_length_mean']:.2f}"
        wandb_metrics["prompt_length/mean"] = metrics["prompt_length_mean"]

    # Token counts
    if "total_tokens" in metrics:
        log_msg += f" - perf/total_num_tokens:{metrics['total_tokens']}"
        wandb_metrics["perf/total_num_tokens"] = metrics["total_tokens"]
    if "prompt_tokens" in metrics:
        log_msg += f" - perf/prompt_tokens:{metrics['prompt_tokens']}"
        wandb_metrics["perf/prompt_tokens"] = metrics["prompt_tokens"]
    if "completion_tokens" in metrics:
        log_msg += f" - perf/completion_tokens:{metrics['completion_tokens']}"
        wandb_metrics["perf/completion_tokens"] = metrics["completion_tokens"]

    # Timing information
    if timings:
        for timing_name, timing_value in timings.items():
            log_msg += f" - timing_s/{timing_name}:{timing_value:.3f}"
            wandb_metrics[f"timing_s/{timing_name}"] = timing_value

    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        log_msg += f" - perf/max_memory_allocated_gb:{memory_allocated:.2f}"
        log_msg += f" - perf/max_memory_reserved_gb:{memory_reserved:.2f}"
        wandb_metrics["perf/max_memory_allocated_gb"] = memory_allocated
        wandb_metrics["perf/max_memory_reserved_gb"] = memory_reserved

    print(log_msg)

    # Log to wandb if initialized
    if wandb_run is not None:
        wandb_run.log(wandb_metrics, step=step)


def log_evaluation_metrics(eval_results, wandb_run=None):
    """Log evaluation metrics to wandb with proper step tracking."""
    if not eval_results:
        return

    step = eval_results.get("step")
    accuracy = eval_results.get("accuracy")
    correct = eval_results.get("correct")
    total = eval_results.get("total")

    # Print evaluation results
    if accuracy is not None:
        print(f"[EVAL] Step {step}: Accuracy {accuracy:.3f} ({correct}/{total})")

    # Log to wandb if initialized
    if wandb_run is not None and step is not None:
        eval_metrics = {
            "eval/accuracy": accuracy,
            "eval/correct": correct,
            "eval/total": total,
            "eval/samples": total,
        }
        wandb_run.log(eval_metrics, step=step)
