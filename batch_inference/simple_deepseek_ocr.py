# # Parallel Batch Inference with DeepSeek OCR
# In this example, we launch a horizontal scaling OCR service (here, not autoscaled, but easily made so)
# * Downloads from Google blob storage (GCS) directly to memory
# * Automatic parallelism and load balancing
# * Built-in progress tracking and fault tolerance
#
# ::youtube[Simple DeepSeek OCR]{url="https://youtu.be/yJ3b6Gps9qI"}
#
# Compare against the examples in /ray/ray_ocr
#
# Run with:
# python simple_deepseek_ocr.py --scale 4 --input-dir gs://rh_demo_image_data/sample_images --creds-path <path to service account json>
# Optional: Run in a kube pod with `kt run python ...`

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Dict

import kubetorch as kt

app = kt.app(cpus=1)


class SimpleOCR:
    """Simple OCR processor using DeepSeek-OCR with vLLM."""

    def __init__(self, creds_path):
        # Kill any residual vLLM process in GPU memory upon restart
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

        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        engine_args = AsyncEngineArgs(
            model="deepseek-ai/DeepSeek-OCR",
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            dtype="auto",
            max_model_len=4096,
            max_num_seqs=64,
            gpu_memory_utilization=0.95,
            logits_processors=[NGramPerReqLogitsProcessor],
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        from google.cloud import storage

        self.client = storage.Client()

    async def download_and_infer(self, gcs_path: str) -> Dict:
        image = await self.download_to_memory(gcs_path)
        print("Downloaded ", gcs_path)

        result = await self.run_inference(gcs_path, image)
        print("Generated", gcs_path)

        self.write_to_gcs(result, gcs_path)
        return result

    async def download_to_memory(self, gcs_path: str):
        import io

        from PIL import Image

        bucket_name, blob_name = gcs_path[5:].split("/", 1)
        blob = self.client.bucket(bucket_name).blob(blob_name)
        image_bytes = await asyncio.to_thread(blob.download_as_bytes)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    def write_to_gcs(self, result: Dict, output_gcs_path: str):
        pass  # Implement something to write out results

    async def run_inference(self, gcs_path: str, image) -> Dict:
        """Run inference on PIL Image in memory - no disk I/O."""
        from uuid import uuid4

        from vllm import SamplingParams

        start_time = time.time()

        try:
            prompt = {
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                "image": image,
            }

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=4096,
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},
                ),
                skip_special_tokens=False,
            )

            results_generator = self.llm.generate(
                prompt,
                sampling_params,
                str(uuid4()),
            )

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            text = final_output.outputs[0].text
            result = {
                "file": gcs_path,
                "status": "success",
                "text": text,
                "processing_time": time.time() - start_time,
            }

            return result

        except Exception as e:
            print(f"Processing error for {gcs_path}: {e}")

            return {
                "file": gcs_path,
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time,
            }


def get_file_names(input_path):
    """Helper to get a list of files to process from a directory"""
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    from google.cloud import storage

    bucket_name, prefix = (
        input_path[5:].split("/", 1) if "/" in input_path[5:] else (input_path[5:], "")
    )
    client = storage.Client()
    blobs = client.bucket(bucket_name).list_blobs(prefix=prefix)

    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if Path(blob.name).suffix.lower() in extensions
    ]


async def main():
    """CLI with async downloads and continuous inference loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple OCR processor")
    parser.add_argument("--input-dir", required=True, help="gs:// path")
    parser.add_argument("--scale", type=int, default=4, help="Worker replicas")
    parser.add_argument("--concurrency", type=int, default=64, help="Batch size")
    parser.add_argument("--output", default="/tmp/results.jsonl", help="Not used")
    parser.add_argument(
        "--creds-path",
        default="~/.config/gcloud/",
        help="Service Account JSON path, locally to send to remote",
    )

    args = parser.parse_args()

    start_time = time.time()
    image = (
        kt.Image(
            image_id="vllm/vllm-openai:nightly-66a168a197ba214a5b70a74fa2e713c9eeb3251a"
        )
        .pip_install(
            [
                "transformers==4.57.1",
                "tokenizers==0.22.1",
                "einops",
                "addict",
                "easydict",
                "google-cloud-storage",
                "pillow",
            ]
        )
        .run_bash("sudo apt-get update && sudo apt-get install nvidia-utils-565 -y")
        # .run_bash("uv pip install --system flash-attn==2.7.3  --no-build-isolation")
    )

    compute = kt.Compute(
        gpus=1,
        image=image,
        secrets=[
            kt.Secret(
                name="gcp-dataaccess",
                path=args.creds_path,
            )
        ],
    ).autoscale(
        max_scale=args.scale,
        min_scale=args.scale,
        concurrency=args.scale * args.concurrency,
        metric="concurrency",
    )
    ocr = kt.cls(SimpleOCR).to(compute, init_args={"creds_path": args.creds_path})

    ocr.async_ = True

    # List all files
    all_files = get_file_names(args.input_dir)
    print(f"Found {len(all_files)} files")

    semaphore = asyncio.Semaphore(
        args.scale * args.concurrency * 1.5
    )  # Don't blow up with too many tasks, but saturate

    async def run_file(gcs_file):
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    ocr.download_and_infer(gcs_file, stream_logs=False),
                    timeout=600,
                )
            except Exception as e:
                return {"file": gcs_file, "status": "error", "error": str(e)}

    tasks = [run_file(file) for file in all_files]
    results = []

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        try:
            results.append(await coro)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(
                f"{i+1}/{len(all_files)} ({100*(i+1)/len(all_files):.1f}%) | {rate:.2f} files/s | ETA: {(len(all_files)-i-1)/rate:.0f}s"
            )

        except Exception as e:
            print(f"Task error: {e}")

    print(f"\nDone: {len(results)}/{len(all_files)} in {time.time()-start_time:.1f}s")
    # print(results[0])
    # ocr.teardown()


if __name__ == "__main__":
    asyncio.run(main())
