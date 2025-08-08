# # RL with TRL and SWE-REX Code Sandbox
# In this example, we will show you how simple it is to launch an RL training
# while dispatching execution to a code sandbox (running on separate compute)
# from within the trainer program.
#
# There are three main components here:
# * A code agent class, which will run code and return stdout/err + success/fail
# * A training encapsulation class that launches the code agent on 0.5 CPUs on separate
# compute on the same Kubernetes cluster, instantiates a TRL trainer, loads data (dummy),
# and then runs the training.
# * Our main, which sends our trainer to GPUs on Kubernetes, instantiates it
# there with specific configs (here, hardcoded; but you should use a config system).
#
# An important thing to note is that this CodeAgent runs on its own compute and
# image, can be called in parallel across many threads, and made to be autoscaling.
# This is not a subprocess within the pod that runs our TRL training, but a service
# being stood up on the fly and called out to to generate results. You can stand up
# arbitrarily complex agents with entirely different image/compute requirements there.


import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import kubetorch as kt
from swerex.runtime.local import Command, LocalRuntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeSandboxTask:
    """Represents a coding task to be executed in a sandbox"""

    prompt: str
    expected_output: Optional[str] = None
    test_code: Optional[str] = None
    max_execution_time: int = 30


class CodeAgent:
    """Code execution environment using swe-rex for sandboxed execution"""

    def __init__(self):
        self.runtime = None
        self._current_tasks = None
        try:
            self.runtime = LocalRuntime()

            logger.info("Code sandbox initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup sandbox: {e}")
            raise

    def execute_code(self, code: str, timeout: int = 30) -> Dict:
        """Execute Python code in the sandbox environment."""
        encoded_code = base64.b64encode(code.encode()).decode()
        try:
            wrapper_script = f"""
import sys, io, json, traceback, base64
from contextlib import redirect_stdout, redirect_stderr

code = base64.b64decode("{encoded_code}").decode()

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()

try:
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exec(code, {{'__name__': '__main__'}})
    result = {{
        "success": True,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue()
    }}
except Exception as e:
    result = {{
        "success": False,
        "stdout": stdout_buf.getvalue(),
        "stderr": traceback.format_exc()
    }}

print(json.dumps(result))
"""

            response = asyncio.run(
                self.runtime.execute(
                    Command(
                        command=["python3", "-c", wrapper_script],
                        shell=False,
                        timeout=timeout,
                    )
                )
            )
            return json.loads(response.stdout.strip())

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Code execution failed " + str(e),
            }


class TRLCodeSandboxTrainer:
    """Main trainer class for GRPO RL fine-tuning with code sandbox"""

    def __init__(self, **config):
        from trl import GRPOConfig, GRPOTrainer

        self.model_name = config.pop("model_name")
        self.grpo_config = GRPOConfig(**config)
        self.agent = self.launch_sandbox()
        self.dataset = None

        # Create reward function for GRPO
        def grpo_reward_function(prompts, completions, **kwargs):
            rewards = []
            for prompt, completion in zip(prompts, completions):
                task_text = prompt.replace("# Task: ", "").replace(
                    "\n# Solution:\n", ""
                )
                task = next(
                    task for task in self._current_tasks if task.prompt == task_text
                )
                result = self.agent.execute_code(completion)
                eval = self.agent.execute_code(f"{completion}\n\n{task.test_code}")
                reward = self.calculate_reward(task, result, eval)
                rewards.append(reward)

            return rewards

        # Initialize GRPO trainer with proper parameters
        self.grpo_trainer = GRPOTrainer(
            model=self.model_name,
            reward_funcs=grpo_reward_function,
            args=self.grpo_config,
        )

        logger.info("TRL GRPO components initialized successfully")

    def launch_sandbox(self):
        """Start a CodeAgent to run Python in SWE REX sandbox"""
        cpus = kt.Compute(
            cpus="0.5",
            image=kt.Image().pip_install(["swe-rex", "numpy", "pandas"]),
        ).autoscale(min_scale=1, max_scale=5)

        return kt.cls(CodeAgent).to(cpus, get_if_exists=True)

    def calculate_reward(
        self, task: CodeSandboxTask, result: dict, test_result: dict = None
    ) -> float:
        """Calculate reward based on code execution results"""
        # Execute the generated code
        reward = 0.0

        # Base reward for successful execution
        if result["success"]:
            reward += 0.5

            # Additional reward for correct output and clean execution
            if (
                task.expected_output
                and task.expected_output.strip() in result["stdout"]
            ):
                reward += 0.3
            if not result["stderr"].strip():
                reward += 0.1

            # Evaluate based on test results, if provided
            if test_result:
                if (
                    task.expected_output
                    and task.expected_output.strip() in test_result["stdout"]
                ):
                    reward += 0.4
        else:
            # Penalty for runtime errors
            reward -= 0.3
            if "SyntaxError" in result.get("error", ""):
                reward += 0.1
            if "Code execution failed" in result.get("error", ""):
                reward = 0

        # Normalize reward to [-1, 1]
        return max(-1.0, min(1.0, reward))

    def train_epoch(self, num_steps: int = 100):
        """Train one epoch with GRPO"""
        logger.info(f"Starting training epoch with {num_steps} steps")

        # Create a subset of the dataset for this epoch
        epoch_dataset = self.dataset.shuffle().select(
            range(min(num_steps, len(self.dataset)))
        )
        self.grpo_trainer.train_dataset = epoch_dataset

        try:
            self.grpo_trainer.train()
            logger.info("Epoch training completed successfully")
        except Exception as e:
            logger.warning(f"Training step failed: {e}, continuing...")

    def load_dataset(self):
        """Dummy method for creating sample coding tasks for training"""
        from datasets import Dataset

        self._current_tasks = [
            CodeSandboxTask(
                prompt="Write a function to calculate the factorial of a number",
                expected_output="120",
                test_code="print(factorial(5))",
            ),
            CodeSandboxTask(
                prompt="Create a function that reverses a string",
                expected_output="olleh",
                test_code="print(reverse_string('hello'))",
            ),
            CodeSandboxTask(
                prompt="Implement a function to check if a number is prime",
                expected_output="True",
                test_code="print(is_prime(17))",
            ),
            CodeSandboxTask(
                prompt="Write a function to find the maximum element in a list",
                expected_output="9",
                test_code="print(find_max([1, 5, 3, 9, 2]))",
            ),
            CodeSandboxTask(
                prompt="Create a function that calculates the sum of even numbers in a range",
                expected_output="30",
                test_code="print(sum_even_numbers(1, 10))",
            ),
        ]

        prompts = []
        for task in self._current_tasks:
            prompt = f"# Task: {task.prompt}\n# Solution:\n"
            prompts.append(prompt)

        self.dataset = Dataset.from_dict({"prompt": prompts})
        logger.info(f"Loaded dataset with {len(self.dataset)} prompts")


def main(grpo_cfg, epochs):
    """Main training function to run locally"""
    logger.info("Starting TRL GRPO Code Sandbox training...")

    # Kubetorch setup for remote execution
    img = kt.Image(
        image_id="nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.6"
    ).pip_install(
        [
            "trl>=0.7.0",
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "peft>=0.6.0",
            "swe-rex",
            "rich",
        ]
    )

    compute = kt.Compute(
        gpus=1,
        image=img,
        launch_timeout=1200,
    )

    try:
        # Initialize trainer
        trainer = kt.cls(TRLCodeSandboxTrainer).to(compute, init_args=grpo_cfg)
        trainer.load_dataset()

        # Train for a few epochs
        for epoch in range(epochs):
            logger.info(f"Training epoch {epoch + 1}/{epochs}")
            trainer.train_epoch()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":

    # Configs directly coded here for visibility
    grpo_cfg = {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "per_device_train_batch_size": 2,
        "generation_batch_size": 4,
        "max_prompt_length": 128,
        "max_completion_length": 256,
        "num_generations": 4,
        "beta": 0.1,
        "log_completions": True,
        "bf16": True,
    }

    main(grpo_cfg=grpo_cfg, epochs=3)
