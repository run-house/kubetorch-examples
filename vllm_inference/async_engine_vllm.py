# # Async Engine vLLM
# The vLLM async engine is powerful as it allows for continuous batching of new requests

import asyncio
import os
import subprocess
import time

import kubetorch as kt


class SimplevLLM:
    """Simple OCR processor using DeepSeek-OCR with vLLM."""

    def __init__(self):
        # Kill any residual vLLM process in GPU memory upon restart
        self._clear_gpu_memory()
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        model_name = "Qwen/Qwen3-4B-FP8"
        engine_args = AsyncEngineArgs(
            model=model_name,
            enable_prefix_caching=True,
            dtype="auto",
            max_model_len=2048,
            max_num_seqs=64,
            gpu_memory_utilization=0.90,
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        time.sleep(2)  # Let engine fully initialize

    def _clear_gpu_memory(self):
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            for line in result.stdout.split("\n"):
                if "vllm" in line.lower() and "C" in line:
                    elems = line.split()
                    pid = elems[elems.index("C") - 1]
                    if pid.isdigit():
                        os.system(f"kill -9 {pid}")
        except Exception as e:
            print(e)
            pass

    async def run_inference(self, input: str):
        from uuid import uuid4

        from vllm import SamplingParams

        try:
            messages = [{"role": "user", "content": input}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0,
                max_tokens=2048,
                skip_special_tokens=False,
                presence_penalty=1,
            )

            results_generator = self.llm.generate(
                prompt,
                sampling_params,
                str(uuid4()),
            )

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            return final_output.outputs[0].text

        except Exception:
            import traceback

            traceback.print_exc()
            return None


async def main():
    """CLI with async downloads and continuous inference loop."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple vLLM inference with continuous batching"
    )
    parser.add_argument("--scale", type=int, default=2, help="Worker replicas")
    parser.add_argument("--concurrency", type=int, default=8, help="Batch size")
    parser.add_argument("--output", default="/tmp/results.jsonl", help="Not used")

    args = parser.parse_args()

    image = kt.Image(
        image_id="vllm/vllm-openai:nightly-97a01308e9ceef351c5ae36bbc9f59b0a03f808b"
    ).run_bash("sudo apt-get update && sudo apt-get install nvidia-utils-565 -y")

    compute = kt.Compute(gpus=1, image=image,).autoscale(
        max_scale=args.scale,
        min_scale=args.scale,
        concurrency=args.scale * args.concurrency,
        metric="concurrency",
    )
    service = kt.cls(SimplevLLM).to(compute, get_if_exists=True)

    service.async_ = True

    from datasets import load_dataset

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = [row["instruction"] for row in dataset.select(range(200))]
    # prompts = ["say hello", "say goodbye"]
    # print(prompts)
    semaphore = asyncio.Semaphore(args.scale * args.concurrency * 1.5)

    async def run_inference(input_text):
        async with semaphore:
            return await asyncio.wait_for(
                service.run_inference(input_text, stream_logs=True),
                timeout=600,
            )

    start_time = time.time()
    tasks = [run_inference(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    print(results[0])
    print(f"Completed {len(results)} in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
