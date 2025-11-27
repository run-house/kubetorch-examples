# Ray Data with DeepSeek OCR
# In this example, we use Ray Data for:
# * Streaming downloads from Google blob storage (GCS) into Ray object store
# * Automatic parallelism and load balancing
# * Built-in progress tracking and fault tolerance
#
# Compare against the example for /batch_inference/simple_deepseek_ocr.py
#
#
# Run with:
# python ray_data_ocr.py --scale 4 --input-dir gs://rh_demo_image_data/sample_images --creds-path <path to service account json>
# Optional: Run in a kube pod with `kt run python ...`
#

import os
import time
from pathlib import Path
from typing import Dict, List

import kubetorch as kt
import ray
import ray.data
from ray.data import ActorPoolStrategy

app = kt.app(cpus=1)


class OCRInferenceActor:
    """GPU actor for OCR inference using vLLM."""

    def __init__(self):
        from vllm import LLM
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        self.llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            dtype="auto",
            max_model_len=4096,
            max_num_seqs=64,
            gpu_memory_utilization=0.95,
            logits_processors=[NGramPerReqLogitsProcessor],
        )

    def __call__(self, batch: Dict[str, List]) -> Dict[str, List]:
        from vllm import SamplingParams

        images = batch["image"]
        paths = batch["path"]

        # Prepare batch inputs
        model_inputs = [
            {
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                "image": img,
            }
            for img in images
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
            extra_args=dict(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}),
            skip_special_tokens=False,
        )

        results = {"path": [], "text": [], "status": [], "error": []}
        try:
            outputs = self.llm.generate(model_inputs, sampling_params)
            for output, path in zip(outputs, paths):
                try:
                    results["path"].append(path)
                    results["text"].append(output.outputs[0].text)
                    results["status"].append("success")
                    results["error"].append(None)
                except Exception as e:
                    results["path"].append(path)
                    results["text"].append(None)
                    results["status"].append("error")
                    results["error"].append(str(e))
        except Exception as e:
            for path in paths:
                results["path"].append(path)
                results["text"].append(None)
                results["status"].append("error")
                results["error"].append(f"Batch error: {str(e)}")

        return results


def load_image_from_bytes(row: Dict) -> Dict:
    """Convert bytes to PIL Image (streaming from Ray's read_binary_files)."""
    import io

    from PIL import Image

    try:
        image = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
        return {"path": row["path"], "image": image, "status": "loaded"}
    except Exception as e:
        return {"path": row["path"], "image": None, "status": "error", "error": str(e)}


def write_to_gcs(result: Dict, output_dir: str):
    """Implement something to write / postprocess results"""
    pass


def process(input_dir, scale, batch_size, creds_path, output_dir):
    print(os.path.expanduser(creds_path))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(creds_path)
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"GOOGLE_APPLICATION_CREDENTIALS": os.path.expanduser(creds_path)}},
    )

    print(f"Streaming from {input_dir}...")
    ds = ray.data.read_binary_files(
        input_dir,
        include_paths=True,
        partition_filter=lambda paths: [
            p for p in paths if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        ],
    )

    # Convert bytes to images (streaming)
    ds = ds.map(load_image_from_bytes, num_cpus=0.5).filter(lambda r: r["status"] == "loaded")

    # Run inference (batched, parallel GPU actors)
    ds = ds.map_batches(
        OCRInferenceActor,
        batch_size=batch_size,
        num_gpus=1,
        concurrency=scale,
        compute=ActorPoolStrategy(size=scale),
    )

    successful = failed = 0
    start_time = time.time()

    for batch in ds.iter_batches(batch_size=None, batch_format="pandas"):
        for i in range(len(batch)):
            row = batch.iloc[i]
            result = {
                "path": row["path"],
                "status": row["status"],
                "text": row["text"],
                "error": row["error"],
            }
            write_to_gcs(result, output_dir)
            if row["status"] == "success":
                successful += 1
            else:
                failed += 1

        total = successful + failed
        rate = total / (time.time() - start_time)
        print(f"{total} files | Success: {successful} | Failed: {failed} | {rate:.2f} files/s")


def main():
    """Run OCR pipeline with Ray Data."""
    import argparse

    parser = argparse.ArgumentParser(description="Ray Data OCR processor")
    parser.add_argument("--input-dir", required=True, help="GCS path (gs://...)")
    parser.add_argument("--scale", type=int, default=4, help="Number of GPU workers")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--output", default="/tmp/ocr_results.jsonl", help="Output file")
    parser.add_argument(
        "--creds-path",
        default="~/.config/gcloud/",
        help="Service Account JSON path, locally",
    )

    args = parser.parse_args()

    # Define an image, mutating the base image without Docker rebuild
    image = (
        kt.Image(image_id="vllm/vllm-openai:nightly-66a168a197ba214a5b70a74fa2e713c9eeb3251a")
        .pip_install(
            [
                "transformers==4.57.1",
                "tokenizers==0.22.1",
                "einops",
                "addict",
                "easydict",
                "google-cloud-storage",
                "pillow",
                "ray[data]==2.50.1",
            ]
        )
        .run_bash("sudo apt-get update && sudo apt-get install nvidia-utils-565 -y")  # convenience for nvidia-smi
    )

    # Initialize Ray compute with kubetorch
    compute = kt.Compute(
        gpus=1,
        image=image,
        secrets=[kt.Secret(name="gcp-dataaccess", path=args.creds_path)],  # See the kt.secret() api
    ).distribute("ray", workers=args.scale)

    ray_processing = kt.fn(process).to(compute)

    # Call the Ray program on remote (argparse used for convenience, but not necessary)
    start_time = time.time()
    ray_processing(args.input_dir, args.scale, args.batch_size, args.creds_path, args.output)
    print(f"Pipeline Complete! Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
