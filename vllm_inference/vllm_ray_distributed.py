# # Inference with Tensor and Pipeline Parallelism
# This example demonstrates distributed inference with parallelism using vLLM + Ray.
# Tensor parallelism is best done when there is interconnect between the GPUs as it splits a single layer
# over multiple devices; therefore, TP is usually best set to number of GPUs on your node.
# Pipeline parallelism splits the model layers across multiple GPUs/nodes, enabling larger models to be served;
# so should be the number of nodes that you have. Generally, TP is preferred to PP for vLLM due to more
# hardening (anecdotally, more snags with PP), and you need to be wary about splitting layers.
#
# Run with: python vllm_ray_distributed.py --workers 2 as an example of 1 x GPU on 2 nodes
# Run with: python vllm_ray_distributed.py --tensor-parallel 2 as an example of 2 x GPU on 1 node.
# And TP_size * PP_size = total GPUs used.
#
# This script both launches and runs inference. You can always launch inference, and then get the running
# inference service by name to call from within any other Python process (line 152). We use async engine here,
# which is equally good for online and offline use cases.

import asyncio
import subprocess

import kubetorch as kt
from openai import AsyncOpenAI


class PipelineParallelvLLM:
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=1024,
        max_num_seqs=256,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        port=8000,
    ):
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port

        self.server_process = None
        self.client = None

    async def download_model(self):
        """Download model weights before starting server."""
        cmd = ["huggingface-cli", "download", self.model_id]
        print(f"Downloading model: {cmd}")
        subprocess.run(cmd, capture_output=True, text=True)

    async def start_engine(self):
        """Start vLLM server using CLI command."""
        if self.server_process:
            return

        cmd = [
            "vllm", "serve", self.model_id,
            "--pipeline-parallel-size", str(self.pipeline_parallel_size),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--dtype", self.dtype,
            "--enforce-eager",
            "--trust-remote-code",
            "--distributed-executor-backend", "ray",
            "--enable-chunked-prefill",
            "--max-num-seqs", str(self.max_num_seqs),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--port", str(self.port),  # fmt: skip
        ]

        print("Starting vLLM server with: \n", cmd)
        self.server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        await self._wait_for_server()
        self.client = AsyncOpenAI(
            base_url=f"http://localhost:{self.port}/v1", api_key="dummy"
        )

    async def _wait_for_server(self, max_wait=1200):
        import aiohttp

        for _ in range(max_wait // 5):
            try:
                async with aiohttp.ClientSession() as session:
                    if (
                        await session.get(f"http://localhost:{self.port}/health")
                    ).status == 200:
                        return
            except:
                pass
            await asyncio.sleep(5)

    async def generate_batch(self, queries, **sampling_kwargs):
        """Generate completions for multiple queries in parallel."""
        tasks = [self.generate(query, **sampling_kwargs) for query in queries]
        return await asyncio.gather(*tasks)

    async def generate(
        self, query, max_tokens=100, temperature=0.9, top_p=0.1, **kwargs
    ):
        """Generate completion using OpenAI client."""
        if not self.client:
            raise RuntimeError("Server not started. Call start_engine() first.")

        response = await self.client.completions.create(
            model=self.model_id,
            prompt=query,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return response.choices[0].text

    async def shutdown(self):
        """Shutdown the vLLM server."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
        self.client = None


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-14B")
    parser.add_argument(
        "--workers", type=int, default=1, help="Pipeline parallel nodes"
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=1, help="Tensor parallel GPUs per node"
    )
    args = parser.parse_args()

    # Setup image and compute
    image = kt.Image(image_id="rayproject/ray:nightly-py311-cu128").pip_install(
        ["vllm", "openai", "aiohttp"]
    )
    compute = kt.Compute(
        gpus=args.tensor_parallel,
        image=image,
        launch_timeout=1200,
        secrets=["huggingface"],
    ).distribute("ray", workers=args.workers)

    # Deploy our inference class to compute, getting it by name if it already exists (to not redeploy)
    remote_inference = kt.cls(PipelineParallelvLLM).to(
        compute,
        init_args={
            "model_id": args.model_id,
            "pipeline_parallel_size": args.workers,
            "tensor_parallel_size": args.tensor_parallel,
        },
        get_if_exists=True,
    )

    # Set to async and download / start the engine (if not already started)
    remote_inference.async_ = True
    await remote_inference.download_model()
    await remote_inference.start_engine()

    # Call remote service for inference. You can run this from within any Python process that is authenticated
    # to Kubernetes, such as within a FastAPI application or calling batch execution.
    result = await remote_inference.generate_batch(
        queries=[
            "Who is Randy Jackson 2?",
            "Why does Camus say Sisyphus is happy",
            "Why is the sky blue",
        ],
        max_tokens=256,
        temperature=0.9,
        top_p=0.1,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
