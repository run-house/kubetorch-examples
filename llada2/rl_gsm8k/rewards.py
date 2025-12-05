"""Reward computation for math problem solving.

Extracts answers from completions and computes rewards based on correctness.
"""
import re
from typing import List, Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from '#### <answer>' format.

    Args:
        text: Completion or answer text

    Returns:
        Extracted answer string or None if not found
    """
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
    return match.group(1).strip() if match else None


def compute_math_rewards(
    completions: List[str],
    answers: List[str],
    correct_reward: float = 1.0,
    incorrect_reward: float = -0.2,
) -> List[float]:
    """Compute rewards for math completions.

    Args:
        completions: Model completion strings
        answers: True answer strings
        correct_reward: Reward for correct answer
        incorrect_reward: Reward for incorrect answer

    Returns:
        List of rewards for each completion
    """
    rewards = []
    for completion, true_answer in zip(completions, answers):
        pred = extract_answer(completion)
        true_val = extract_answer(true_answer)

        # Fallback: use stripped answer if extraction fails
        if true_val is None:
            true_val = true_answer.strip()

        reward = correct_reward if pred == true_val else incorrect_reward
        rewards.append(reward)

    return rewards
