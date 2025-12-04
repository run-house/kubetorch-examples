"""SGLang-based inference engine for LLaDA2 models.

This module provides an SGLang wrapper for efficient async inference with LLaDA2 models.
It supports hot-swapping of full model weights without restarting the engine, making it
ideal for reinforcement learning scenarios where model weights are frequently updated.

Key Features:
- Uses SGLang's runtime for efficient async inference
- Supports hot-swapping of full model weights without restarting
- Implements version tracking to filter stale requests after checkpoint updates
- Caches the engine across redeployments for fast checkpoint switching
- Reads from local disk for RL weight reloading

Based on the fork: https://github.com/ClawSeven/sglang/tree/dev-dllm
"""

from typing import Any, Dict, List, Optional, Tuple
import os 
import kubetorch as kt


class SGLang:
    """SGLang wrapper with hot-swapping support for LLaDA2 models.

    This class wraps SGLang's runtime engine for efficient async inference.
    It's designed to work with the LLaDA2 model architecture and supports
    hot-swapping of checkpoints for reinforcement learning workflows.

    Args:
        model_id: Base model ID or local path to model
        model_checkpoint: Path to checkpoint for hot-swapping
        checkpoint_version: Version number for tracking checkpoint updates
        kt_cached_state: Cached state from previous deployment (for hot-swapping)
        config: Configuration dictionary with compute and training settings
    """

    def __init__(
        self,
        model_id: str = "inclusionAI/LLaDA2.0-mini-preview",
        model_checkpoint: Optional[str] = None,
        checkpoint_version: int = 0,
        kt_cached_state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoint_version = checkpoint_version
        self.model_id = model_id
        self.server_process = None
        self.server_url = None

        # Reuse cached server if available (for hot-swapping)
        if kt_cached_state and kt_cached_state.get("server_process") is not None:
            print(
                f"Reusing SGLang server from cache (version {self.checkpoint_version})"
            )
            self.server_process = kt_cached_state["server_process"]
            self.server_url = kt_cached_state.get(
                "server_url", "http://127.0.0.1:30000"
            )

            if self._check_health():
                return
            else:
                print("Failed to find active SGLang on reload, starting from scratch")
        
        # Create new server if not cached
        print(f"Creating new SGLang server (version {self.checkpoint_version})")

        # Use the checkpoint path as model if provided, otherwise download base model
        # This ensures we read from local disk for RL scenarios
        if model_checkpoint:
            model_path = model_checkpoint
            print(f"Using checkpoint: {model_path}")
        else:
            # Download model from HuggingFace if not already cached
            import os
            from huggingface_hub import snapshot_download

            if os.path.exists(model_id) and os.path.isdir(model_id):
                model_path = model_id
            else:
                model_path = snapshot_download(repo_id=model_id, local_dir=model_id)

        # NOTE: Merge disabled - SGLang's dev-dllm fork doesn't support merged MoE weights
        # The SGLang implementation expects individual expert keys (experts.0.gate_proj, etc.)
        # not merged tensors (experts.gate_proj with shape [num_experts, ...])
        # import sys
        # parent_dir = os.path.dirname(os.path.dirname(__file__))
        # if parent_dir not in sys.path:
        #     sys.path.insert(0, parent_dir)
        # from moe_converter import ensure_moe_merged
        # model_path = ensure_moe_merged(model_path, rank=0)

        # Configure engine using config values
        compute_config = config.get("compute", {}) if config else {}

        import subprocess
        import sys

        # Build server command
        port = 30000
        host = "127.0.0.1"
        cmd = [ # fmt: off
            sys.executable, "-m", "sglang.launch_server", "--model-path", model_path, 
            "--tokenizer-path", model_id,  # Use base model's tokenizer for LLaDA2
            "--host", host, 
            "--port", str(port), 
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--mem-fraction-static", str(compute_config.get("inference_gpu_memory_utilization", 0.9)),
            "--max-total-tokens", str(compute_config.get("inference_max_model_len", 2048)),
            "--context-length", str(compute_config.get("inference_max_model_len", 1024)),
            "--dllm-block-size", str(compute_config.get("diffusion_block_size", 128)),
            "--dllm-algorithm", str(compute_config.get("diffusion_algorithm", "LowConfidence")),
            # " > sglang_server.log"
        ]

        print(f"Starting SGLang HTTP server on {host}:{port}...")
        print(f"Command: {' '.join(cmd)}")

        # Launch server and check for up
        self.server_process = subprocess.Popen(
            cmd #, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.server_url = f"http://{host}:{port}"

        import time

        for i in range(50):
            if self._check_health():
                print(f"SGLang server ready at {self.server_url}")
                break
            print('Retrying')
            time.sleep(15)
        else:
            raise RuntimeError("SGLang server failed to start")

        print(f"SGLang server initialized with model: {model_path}")

    def _check_health(self):
        import httpx
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.server_url}/health", timeout=15.0)
                print(f"Health check response: {response.status_code} - {response.text}")
                return response.status_code == 200
        except Exception as e:
            print('Failed check', e)
            return False

    def __kt_cached_state__(self) -> Dict[str, Any]:
        """Return state to be cached by Kubetorch across reloads.

        This method is called by Kubetorch before reloading the class.
        The returned dictionary will be passed to the new instance's __init__
        via the kt_cached_state parameter, enabling hot-swapping of checkpoints
        without restarting the server.

        Returns:
            Dictionary containing the server info to preserve
        """
        # Preserve the SGLang server process for hot-swapping
        return {
            "server_process": self.server_process,
            "server_url": self.server_url,
        }

    async def generate(
        self, prompts: List[str], request_version: Optional[int] = None, **kwargs
    ) -> Tuple[List[str], List[List[int]]]:
        # if request_version is not None and request_version != self.checkpoint_version:
        #     print(
        #         f"Ignoring stale request from version {request_version} "
        #         f"(current: {self.checkpoint_version})"
        #     )
        #     return [""] * len(prompts), [[]] * len(prompts)

        sampling_params = {
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 0.95),
            "max_new_tokens": kwargs.get("max_tokens", 512),
        }

        print(
            f"Processing {len(prompts)} prompts with SGLang engine v{self.checkpoint_version} (batch mode)"
        )

        import httpx
        request_data = {
            "text": prompts if isinstance(prompts, list) else [prompts],
            "sampling_params": {
                "temperature": sampling_params.get("temperature", 0.6),
                "top_p": sampling_params.get("top_p", 0.95),
                "max_new_tokens": sampling_params.get("max_new_tokens", 1024),
            },
        }

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.server_url}/generate", json=request_data
            )

            # Debug: Log response details if there's an error
            if response.status_code != 200:
                print(f"ERROR: SGLang returned {response.status_code}")
                print(f"Request payload: {request_data}")
                try:
                    error_detail = response.json()
                    print(f"Response JSON: {error_detail}")
                except:
                    print(f"Response text: {response.text}")

            response.raise_for_status()
            results = response.json()

        print(f"Batch generation completed: {len(results)} results")

        # Extract completions from HTTP response
        completions = []
        token_ids_list = []

        for idx, result in enumerate(results):
            completion = result.get("text", "")
            print("DEBUG: ", completion)
            completions.append(completion)

            # Token IDs from output_ids field (actual token IDs list)
            tokens = result.get("output_ids", [])
            print(
                f"DEBUG tokens type: {type(tokens)}, value: {tokens[:10] if len(tokens) > 10 else tokens}..."
            )

            if not isinstance(tokens, list):
                raise ValueError(
                    f"Expected output_ids to be a list, got {type(tokens)}: {tokens}"
                )

            token_ids_list.append(tokens)

        print(f"SGLang generation completed: {len(completions)} completions")
        return completions, token_ids_list

    async def update_weights_from_tensor(
        self, state_dict: Dict[str, Any], new_version: int, load_format: str = "dtensor"
    ):
        """TODO"""
        pass

    async def update_weights_from_disk(self, key: str, new_version: int, checkpoint_subdir: str):
        """Update model weights from disk without restarting the engine.

        Args:
            key: The key used in kt.vput/kt.get
            new_version: Version number for the checkpoint
            checkpoint_subdir: The subdirectory name created by kt.get (e.g., "checkpoint-v1-step100")
        """

        import os

        # Ensure destination directory exists before syncing
        dest_path = key + f"_{new_version}"
        os.makedirs(dest_path, exist_ok=True)

        await kt.get_async(
            key=key,
            dest=dest_path,
            seed_data=False,
        )
        print('Gotten the file!')
        # Construct full model path including subdirectory
        full_model_path = os.path.join(dest_path, checkpoint_subdir)

        import httpx
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.server_url}/update_weights_from_disk",
                    json={"model_path": full_model_path},
                )
                response.raise_for_status()
        except Exception as e:
            print(f"FATAL: Weight update HTTP request failed: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(
                f"Failed to update weights: {e}"
            ) from e

        # Verify server is still healthy after weight update
        print("Verifying SGLang server health after weight update...")
        import asyncio
        for i in range(10):
            await asyncio.sleep(3)
            if self._check_health():
                break
        else:
            raise RuntimeError("Server health check timed out after weight update")

        # Update version tracking
        old_version = self.checkpoint_version
        self.checkpoint_version = new_version

        print(f"Successfully updated weights from v{old_version} to v{new_version}")

        # Clean up the previous version's destination directory after successful sync
        import shutil
        if old_version > 0:
            old_dest_path = key + f"_{old_version}"
            if os.path.exists(old_dest_path):
                try:
                    shutil.rmtree(old_dest_path)
                    print(f"Cleaned up old checkpoint: {old_dest_path}")
                except Exception as e:
                    print(f"Warning: Failed to clean up old checkpoint {old_dest_path}: {e}")

        return True


async def main(): 
    img = (
        kt.Image(image_id="lmsysorg/sglang:v0.5.6")
        # .run_bash("apt-get update && apt-get install -y libnuma-dev")
        # .run_bash("uv pip install --system --break-system-packages sgl_kernel")
        .run_bash(
            "uv pip install --break-system-packages --system 'git+https://github.com/ClawSeven/sglang.git@dev-dllm#subdirectory=python'"
        )
        .pip_install(["huggingface_hub"])
        .set_env_vars({"HF_TOKEN": os.environ["HF_TOKEN"]})
    )
    compute = kt.Compute(gpus=1, memory = "150Gi", image=img, allowed_serialization=["json", "pickle"])

    infer = kt.cls(SGLang).to(compute, get_if_exists=True)
    infer._async = True 
    # result = await infer.generate(["Why does camus say sisyphus is happy?"], max_tokens = 512)
    # print(result)
    
    key = "model_v2"
    result = await infer.update_weights_from_disk(key, 2, "checkpoint-v2-step0")
    print(result)

    result = await infer.generate(["Why does camus say sisyphus is happy?"], max_tokens = 512)
    print(result)


if __name__ == "__main__":
    import asyncio 
    asyncio.run(main())