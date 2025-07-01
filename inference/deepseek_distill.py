# Deploy via `kt deploy deepseek_distill.py` and then you can call inference against it -
# for instance, see main() for using the remote class as if it were local.
import os

import kubetorch as kt


img = kt.Image(image_id="vllm/vllm-openai:latest").set_env_vars(
    {"HF_TOKEN": os.environ["HF_TOKEN"]}
)


@kt.compute(gpus="1", image=img, name="deepseek_llama")
class DeepSeekDistillvLLM:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model_id = model_id
        self.model = None

    def load_model(self):
        from vllm import LLM

        print("loading model")
        import torch

        self.model = LLM(
            self.model_id,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,  # Reduces size of KV store
        )
        print("model loaded")

    def generate(
        self, queries, temperature=0.65, top_p=0.95, max_tokens=5120, min_tokens=32
    ):
        if self.model is None:
            self.load_model()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

        outputs = self.model.generate(queries, sampling_params)
        return [
            {
                "prompt": output.prompt,
                "outputs": [{"text": o.text} for o in output.outputs],
            }
            for output in outputs
        ]


if __name__ == "__main__":
    # Run inference remotely and print the results
    queries = [
        "What is the relationship between bees and a beehive compared to programmers and...?",
        "How many R's are in Strawberry?",
        """Roman numerals are formed by appending the conversions of decimal place values from highest to lowest.
        Converting a decimal place value into a Roman numeral has the following rules: If the value does not start with 4 or 9, select the symbol of the maximal value that can be subtracted from the input, append that symbol to the result, subtract its value, and convert the remainder to a Roman numeral.
        If the value starts with 4 or 9 use the subtractive form representing one symbol subtracted from the following symbol, for example, 4 is 1 (I) less than 5 (V): IV and 9 is 1 (I) less than 10 (X): IX. Only the following subtractive forms are used: 4 (IV), 9 (IX), 40 (XL), 90 (XC), 400 (CD) and 900 (CM). Only powers of 10 (I, X, C, M) can be appended consecutively at most 3 times to represent multiples of 10. You cannot append 5 (V), 50 (L), or 500 (D) multiple times. If you need to append a symbol 4 times use the subtractive form.
        Given an integer, write and return Python code to convert it to a Roman numeral.""",
    ]

    outputs = DeepSeekDistillvLLM.generate(queries, temperature=0.7)
    for output in outputs:
        prompt = output["prompt"]
        generated_text = output["outputs"][0]["text"]
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
