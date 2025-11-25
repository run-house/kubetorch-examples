import re


class SimpleMathAgent:
    """Math problem solver using vLLM."""

    def __init__(self, inference_service, checkpoint_version=0):
        self.inference_service = inference_service
        self.checkpoint_version = checkpoint_version
        self.system_prompt = (
            "You are a helpful math assistant. "
            "Solve the following problem step by step. "
            "End with '#### <answer>' where <answer> is just the number."
        )

    async def generate_batch(
        self, questions, answers, num_generations=4, step_num=None
    ):
        """Generate multiple completions per question and calculate rewards."""
        if step_num:
            print(
                f"[INFERENCE] Starting generation for step {step_num} (checkpoint v{self.checkpoint_version})"
            )

        # Expand for multiple generations
        expanded_questions = []
        expanded_answers = []
        for q, a in zip(questions, answers):
            expanded_questions.extend([q] * num_generations)
            expanded_answers.extend([a] * num_generations)

        # Format prompts
        prompts = [
            f"{self.system_prompt}\n\nQuestion: {q}\n\nSolution:"
            for q in expanded_questions
        ]

        # Generate completions with version tracking
        completions, token_ids = await self.inference_service.generate(
            prompts,
            request_version=self.checkpoint_version,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
        )

        if step_num:
            print(f"[INFERENCE] Completed generation for step {step_num}")

        # Check if request was ignored due to being stale
        if all(c == "" for c in completions):
            print(
                f"Request was stale (version {self.checkpoint_version}), skipping batch"
            )
            return None, None, None, None

        # Calculate rewards
        rewards = []
        for completion, true_answer in zip(completions, expanded_answers):
            # Extract predicted answer
            match = re.search(r"####\s*([-+]?\d*\.?\d+)", completion)
            pred_answer = match.group(1).strip() if match else None

            # Extract true answer
            true_match = re.search(r"####\s*([-+]?\d*\.?\d+)", true_answer)
            true_value = (
                true_match.group(1).strip() if true_match else true_answer.strip()
            )

            # Simple reward: 1.0 for correct, -0.2 for wrong
            reward = 1.0 if pred_answer == true_value else -0.2
            rewards.append(reward)

        print(f"[INFERENCE] Generated {len(completions)} samples.")
        return prompts, completions, token_ids, rewards
