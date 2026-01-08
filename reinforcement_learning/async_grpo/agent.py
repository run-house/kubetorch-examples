import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RolloutBatch:
    """Result of a single rollout batch."""

    step: int
    prompts: List[str]
    completions: List[str]
    token_ids: List[List[int]]
    rewards: List[float]


class SimpleMathAgent:
    """Math problem solver using vLLM with optional sharded inference."""

    def __init__(
        self,
        inference_service,
        config: dict,
        checkpoint_version=0,
        num_inference_workers=1,
    ):
        self.inference_service = inference_service
        self.checkpoint_version = checkpoint_version
        self.config = config
        self.num_inference_workers = num_inference_workers
        self.system_prompt = (
            "You are a helpful math assistant. "
            "Solve the following problem step by step. "
            "You should end with the answer after four pound signs "
            "like '#### THE_ANSWER', where THE_ANSWER is just a number."
            "End with that pattern have no further commentary or text after that"
        )

    async def generate(self, prompts: List[str], **kwargs) -> tuple:
        """Generate across inference workers and join results.

        If num_inference_workers > 1, splits prompts into N chunks,
        fires N parallel calls (kubetorch round-robins to workers),
        and joins results.
        """
        if self.num_inference_workers <= 1:
            return await self.inference_service.generate(prompts, **kwargs)

        # Split prompts into N chunks
        n = self.num_inference_workers
        chunk_size = (len(prompts) + n - 1) // n
        chunks = [prompts[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]
        chunks = [c for c in chunks if c]

        # Fire N parallel calls - kubetorch round-robins to workers
        results = await asyncio.gather(
            *[self.inference_service.generate(chunk, **kwargs) for chunk in chunks]
        )

        # Join results maintaining order
        all_completions = []
        all_token_ids = []
        for completions, token_ids in results:
            all_completions.extend(completions)
            all_token_ids.extend(token_ids)

        return all_completions, all_token_ids

    def _extract_answer(self, text):
        """Extract numeric answer from text, trying multiple formats."""
        # Try #### format first (GSM8K ground truth format)
        match = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
        if match:
            return match.group(1).strip()
        # Try <answer> format (model output format)
        match = re.search(r"<answer>\$?([-+]?[\d,]+\.?\d*)</answer>", text)
        if match:
            return match.group(1).strip().replace(",", "")
        return None

    def _compute_rewards(self, completions, answers):
        rewards_config = self.config.get("rewards", {})
        correct_reward = rewards_config.get("correct", 1.0)
        incorrect_reward = rewards_config.get("incorrect", -0.2)

        rewards = []
        for completion, true_answer in zip(completions, answers):
            pred_answer = self._extract_answer(completion)
            true_value = self._extract_answer(true_answer)

            reward = correct_reward if pred_answer == true_value else incorrect_reward
            rewards.append(reward)
        return rewards

    async def generate_rollouts(
        self, questions, answers, num_generations=4, step=None
    ) -> Optional[RolloutBatch]:
        """Generate rollouts and compute rewards."""
        expanded_questions = [q for q in questions for _ in range(num_generations)]
        expanded_answers = [a for a in answers for _ in range(num_generations)]

        prompts = [
            f"{self.system_prompt}\n\nQuestion: {q}\n\nSolution:"
            for q in expanded_questions
        ]

        gen_config = self.config.get("generation", {})
        completions, token_ids = await self.generate(
            prompts,
            request_version=self.checkpoint_version,
            max_tokens=gen_config.get("max_tokens", 512),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.95),
        )

        if all(c == "" for c in completions):
            return None

        rewards = self._compute_rewards(completions, expanded_answers)

        return RolloutBatch(
            step=step,
            prompts=prompts,
            completions=completions,
            token_ids=token_ids,
            rewards=rewards,
        )

    async def evaluate_accuracy(
        self, test_questions, test_answers, num_samples=100, step=None
    ):
        """Evaluate model accuracy on test dataset."""
        print(f"[EVAL] Starting evaluation on {num_samples} test samples")

        eval_questions = test_questions[:num_samples]
        eval_answers = test_answers[:num_samples]

        prompts = [
            f"{self.system_prompt}\n\nQuestion: {q}\n\nSolution:"
            for q in eval_questions
        ]

        gen_config = self.config.get("generation", {})
        completions, _ = await self.generate(
            prompts,
            request_version=self.checkpoint_version,
            max_tokens=gen_config.get("max_tokens", 512),
            temperature=0.1,  # Lower temperature for deterministic evaluation
            top_p=0.95,
        )

        if all(c == "" for c in completions):
            print(
                f"[EVAL] Request was stale (version {self.checkpoint_version}), skipping evaluation"
            )
            return None

        correct = 0
        total = len(eval_questions)

        for i, (completion, true_answer) in enumerate(zip(completions, eval_answers)):
            pred_answer = self._extract_answer(completion)
            true_value = self._extract_answer(true_answer)

            is_correct = pred_answer == true_value
            if is_correct:
                correct += 1

            if i < 3:
                # Debug: show end of completion where answer should be
                completion_end = (
                    completion[-300:].replace("\n", "\\n") if completion else "<empty>"
                )
                print(f"[EVAL DEBUG] Example {i+1} completion end: ...{completion_end}")
                print(
                    f"[EVAL] Example {i+1}: Predicted: {pred_answer}, True: {true_value}, Correct: {is_correct}"
                )

        accuracy = correct / total
        print(f"[EVAL] Accuracy: {correct}/{total} = {accuracy:.3f}")

        return {"accuracy": accuracy, "correct": correct, "total": total, "step": step}
