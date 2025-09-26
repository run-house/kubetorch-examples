"""
Reward calculation logic for LeetCode problem evaluation.
Contains reward configurations and calculation functions.
"""

import asyncio
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def get_default_reward_config() -> Dict[str, float]:
    """Get default reward configuration"""
    return {
        "success_reward": 1.0,
        "failed_run_reward": -1.0,
        "name_error_reward": -0.9,
        "indent_error_reward": -0.8,
        "assertion_error_reward": -0.5,
        "other_error_reward": -0.7,
    }


def calculate_reward(
    success: bool, stderr: str, reward_config: Dict[str, float] = None
) -> float:
    """Calculate reward based on code execution results"""
    if reward_config is None:
        reward_config = get_default_reward_config()

    if success:
        return reward_config.get("success_reward", 1.0)
    elif "failed to run" in stderr:
        return reward_config.get("failed_run_reward", -1.0)
    elif "NameError" in stderr:
        return reward_config.get("name_error_reward", -0.9)
    elif "IndentationError" in stderr:
        return reward_config.get("indent_error_reward", -0.8)
    elif "AssertionError" in stderr:
        return reward_config.get("assertion_error_reward", -0.5)
    else:
        return reward_config.get("other_error_reward", -0.7)


def build_executable_code(
    context: str, completion: str, test: str, entrypoint: str
) -> str:
    """Build the complete code string for execution"""
    return (
        context
        + "\n\n"
        + completion
        + "\n\n"
        + test
        + "\n\n"
        + "solution = Solution()"
        + "\n\n"
        + f"check({entrypoint})"
    )


async def execute_and_reward(
    completion: str,
    context: str,
    test: str,
    entrypoint: str,
    code_agent,
    timeout: int = 5,
    reward_config: Dict[str, float] = None,
) -> float:
    """Execute code and return calculated reward"""
    code_to_run = build_executable_code(context, completion, test, entrypoint)

    try:
        # Use the timeout built into execute_code instead of nested timeout
        result = await code_agent.execute_code(
            code_to_run, timeout=timeout, stream_logs=False
        )
        success = result["success"]
        stderr = result["stderr"]
    except asyncio.TimeoutError:
        logger.warning(f"Code execution timed out after {timeout}s")
        success = False
        stderr = "execution timeout"
    except Exception as e:
        logger.error(f"Exception in execute_code: {type(e).__name__}: {str(e)}")
        success = False
        stderr = "failed to run"

    return calculate_reward(success, stderr, reward_config)


async def execute_for_evaluation(
    completion: str,
    context: str,
    test: str,
    entrypoint: str,
    code_agent,
    timeout: int = 5,
) -> bool:
    """Execute code for evaluation and return success status"""
    code_to_run = build_executable_code(context, completion, test, entrypoint)

    try:
        # Use the timeout built into execute_code instead of nested timeout
        result = await code_agent.execute_code(
            code_to_run, timeout=timeout, stream_logs=False
        )
        return result["success"]
    except asyncio.TimeoutError:
        logger.warning(f"Evaluation timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"Exception in evaluation: {type(e).__name__}: {str(e)}")
        return False


async def batch_execute_with_rewards(
    completions: List[str],
    contexts: List[str],
    tests: List[str],
    entrypoints: List[str],
    code_agent,
    semaphore: asyncio.Semaphore,
    timeout: int = 5,
    reward_config: Dict[str, float] = None,
) -> List[float]:
    """Execute multiple code samples concurrently and return rewards"""
    print("Batch start")
    default_config = reward_config or get_default_reward_config()

    async def execute_single(completion, context, test, entrypoint):
        """Execute a single code sample with proper error handling"""
        try:
            async with semaphore:
                # Execute with timeout
                result = await execute_and_reward(
                    completion,
                    context,
                    test,
                    entrypoint,
                    code_agent,
                    timeout,
                    default_config,
                )
                return result
        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception:
            return default_config.get("other_error_reward", -0.7)

    tasks = [
        execute_single(completion, context, test, entrypoint)
        for completion, context, test, entrypoint in zip(
            completions, contexts, tests, entrypoints
        )
    ]

    # Use return_exceptions=True to ensure all tasks complete
    rewards = await asyncio.gather(*tasks, return_exceptions=True)
    print("Rewards present")

    # Handle any remaining exceptions (should be rare with new error handling)
    for i, reward in enumerate(rewards):
        if isinstance(reward, Exception):
            logger.error(f"Gather returned exception for task {i}: {reward}")
            rewards[i] = default_config.get("other_error_reward", -0.7)

    return rewards


async def batch_evaluate(
    completions: List[str],
    contexts: List[str],
    tests: List[str],
    entrypoints: List[str],
    code_agent,
    semaphore: asyncio.Semaphore,
    timeout: int = 5,
) -> List[bool]:
    """Execute multiple code samples for evaluation and return success statuses"""

    async def evaluate_single(completion, context, test, entrypoint):
        """Evaluate a single code sample with proper error handling"""
        try:
            async with semaphore:
                result = await execute_for_evaluation(
                    completion, context, test, entrypoint, code_agent, timeout
                )
                return result
        except asyncio.CancelledError:
            raise  # Re-raise cancellation
        except Exception:
            return False

    tasks = [
        evaluate_single(completion, context, test, entrypoint)
        for completion, context, test, entrypoint in zip(
            completions, contexts, tests, entrypoints
        )
    ]

    # Use return_exceptions=True to ensure all tasks complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any remaining exceptions (should be rare with new error handling)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Gather returned exception for evaluation task {i}: {result}")
            results[i] = False

    return results
