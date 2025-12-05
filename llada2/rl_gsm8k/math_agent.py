"""Math problem solving agent for GRPO training.

Generates completions and computes rewards for math problems.
"""
from typing import List, Optional, Tuple

from llada2.rl_gsm8k.rewards import compute_math_rewards


class SimpleMathAgent:
    """Math problem solver using inference service."""

    SYSTEM_PROMPT = (
        "You are a helpful math assistant. "
        "Solve the following problem step by step. "
        "End with '#### <answer>' where <answer> is just the number."
    )

    def __init__(self, inference_service, checkpoint_version: int = 0):
        """Initialize agent.

        Args:
            inference_service: Async inference service
            checkpoint_version: Current checkpoint version for staleness detection
        """
        self.inference_service = inference_service
        self.checkpoint_version = checkpoint_version

    async def generate_batch(
        self,
        questions: List[str],
        answers: List[str],
        num_generations: int = 4,
        step_num: Optional[int] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[List[int]]], Optional[List[float]]]:
        """Generate completions and compute rewards.

        Args:
            questions: Problem questions
            answers: True answers
            num_generations: Completions per question (K for GRPO)
            step_num: Current training step (for logging)

        Returns:
            Tuple of (prompts, completions, token_ids, rewards) or all None if stale
        """
        if step_num:
            print(f"[INFERENCE] Starting step {step_num} (v{self.checkpoint_version})")

        # Expand for K generations per question
        expanded_questions, expanded_answers = self._expand_batch(
            questions, answers, num_generations
        )

        # Format prompts
        prompts = [
            f"{self.SYSTEM_PROMPT}\n\nQuestion: {q}\n\nSolution:"
            for q in expanded_questions
        ]

        # Generate completions
        completions, token_ids = await self.inference_service.generate(
            prompts,
            request_version=self.checkpoint_version,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )

        if step_num:
            print(f"[INFERENCE] Completed step {step_num}")

        # Check for stale request
        if all(c == "" for c in completions):
            print(f"Request stale (v{self.checkpoint_version}), skipping")
            return None, None, None, None

        # Compute rewards
        rewards = compute_math_rewards(completions, expanded_answers)

        print(f"[INFERENCE] Generated {len(completions)} samples")
        return prompts, completions, token_ids, rewards

    def _expand_batch(
        self,
        questions: List[str],
        answers: List[str],
        num_generations: int,
    ) -> Tuple[List[str], List[str]]:
        """Expand batch for K generations per question."""
        expanded_questions = []
        expanded_answers = []
        for q, a in zip(questions, answers):
            expanded_questions.extend([q] * num_generations)
            expanded_answers.extend([a] * num_generations)
        return expanded_questions, expanded_answers
