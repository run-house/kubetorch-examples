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
        self.engine = None
        self.tokenizer = None
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
            self.engine = None
            self.tokenizer = None
            try:
                with httpx.Client() as client:
                    client.get(f"{self.server_url}/health", timeout=5.0)
                return
            except:
                print("Failed to find active server on reload, starting from scratch")

        # Create new server if not cached
        print(f"Creating new SGLang server (version {self.checkpoint_version})")

        # Use the checkpoint path as model if provided, otherwise use base model
        # This ensures we read from local disk for RL scenarios
        model_path = model_checkpoint if model_checkpoint else model_id

        # NOTE: Merge disabled - SGLang's dev-dllm fork doesn't support merged MoE weights
        # The SGLang implementation expects individual expert keys (experts.0.gate_proj, etc.)
        # not merged tensors (experts.gate_proj with shape [num_experts, ...])
        #
        # # Merge MoE experts for optimized inference (if needed)
        # # The merged format enables faster Triton kernels and better SGLang performance
        # import sys
        # parent_dir = os.path.dirname(os.path.dirname(__file__))
        # if parent_dir not in sys.path:
        #     sys.path.insert(0, parent_dir)
        # from moe_converter import ensure_moe_merged
        # model_path = ensure_moe_merged(model_path, rank=0)

        # Configure engine using config values
        compute_config = config.get("compute", {}) if config else {}

        # Start SGLang HTTP server (avoids event loop conflicts)
        # The server runs in its own process with continuous batching
        import subprocess
        import sys

        # Build server command
        port = 30000
        host = "127.0.0.1"

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--tokenizer-path",
            model_id,  # Use base model's tokenizer for LLaDA2
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--mem-fraction-static",
            str(compute_config.get("inference_gpu_memory_utilization", 0.9)),
            "--max-total-tokens",
            str(compute_config.get("inference_max_model_len", 1024)),
            "--context-length",
            str(compute_config.get("inference_max_model_len", 1024)),
            "--diffusion-block-size",
            str(compute_config.get("diffusion_block_size", 128)),
            "--diffusion-algorithm",
            str(compute_config.get("diffusion_algorithm", "LowConfidence")),
            # " > sglang_server.log 2>&1"
        ]

        print(f"Starting SGLang HTTP server on {host}:{port}...")
        print(f"Command: {' '.join(cmd)}")

        # Launch server as subprocess
        self.server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.server_url = f"http://{host}:{port}"

        import time

        # Wait for server to be ready
        import httpx

        for i in range(50):
            try:
                with httpx.Client() as client:
                    client.get(f"{self.server_url}/health", timeout=10.0)
                print(f"SGLang server ready at {self.server_url}")
                break
            except:
                time.sleep(5)
        else:
            raise RuntimeError("SGLang server failed to start")

        # Store server info
        self.engine = None  # Not using direct engine anymore
        self.tokenizer = None  # Tokenization handled by server

        print(f"SGLang server initialized with model: {model_path}")

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
        """Generate completions for the given prompts.

        This method processes prompts asynchronously and returns both text completions
        and token IDs. It supports version tracking to filter stale requests from
        old checkpoint versions.

        Args:
            prompts: List of prompt strings to generate from
            request_version: Version number of the checkpoint that created this request.
                           If it doesn't match current checkpoint_version, the request
                           is ignored (returns empty results).
            **kwargs: Additional sampling parameters (temperature, top_p, max_tokens, etc.)

        Returns:
            Tuple of (completions, token_ids) where:
                - completions: List of generated text strings
                - token_ids: List of token ID lists
        """
        # Check if this request is from an old checkpoint version
        if request_version is not None and request_version != self.checkpoint_version:
            print(
                f"Ignoring stale request from version {request_version} "
                f"(current: {self.checkpoint_version})"
            )
            # Return empty results for stale requests
            return [""] * len(prompts), [[]] * len(prompts)

        # Extract sampling parameters for SGLang
        sampling_params = {
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 0.95),
            "max_new_tokens": kwargs.get("max_tokens", 1024),
        }

        # Process all prompts as a batch
        print(
            f"Processing {len(prompts)} prompts with SGLang engine v{self.checkpoint_version} (batch mode)"
        )

        # Use HTTP API to avoid event loop conflicts
        import httpx

        # Prepare request payload
        request_data = {
            "text": prompts if isinstance(prompts, list) else [prompts],
            "sampling_params": {
                "temperature": sampling_params.get("temperature", 0.6),
                "top_p": sampling_params.get("top_p", 0.95),
                "max_new_tokens": sampling_params.get("max_new_tokens", 1024),
            },
        }

        # Make async HTTP request to SGLang server
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.server_url}/generate", json=request_data
            )
            response.raise_for_status()
            results = response.json()

        print(f"Batch generation completed: {len(results)} results")

        # Extract completions from HTTP response
        # SGLang HTTP API returns: [{"text": "...", "meta_info": {...}}, ...]
        completions = []
        token_ids_list = []

        for idx, result in enumerate(results):
            completion = result.get("text", "")
            print("DEBUG: ", completion)
            completions.append(completion)

            # Token IDs from output_ids field (actual token IDs list)
            # Note: completion_tokens in meta_info is just a count (int), not the IDs
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

    async def update_weights_from_disk(self, checkpoint_path: str, new_version: int):
        """Update model weights from disk without restarting the engine.

        This method loads weights from disk, which is slower than update_weights_from_tensor
        but useful when weights are only available on disk.

        For RL workflows with frequent updates, prefer update_weights_from_tensor instead.

        Args:
            checkpoint_path: Local path to the checkpoint directory
            new_version: New checkpoint version number

        Raises:
            RuntimeError if weight update fails or server becomes unhealthy
        """
        print(
            f"Updating weights from {checkpoint_path} (v{self.checkpoint_version} -> v{new_version})"
        )

        import time

        # Use HTTP API to update weights
        import httpx

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.server_url}/update_weights_from_disk",
                    json={"model_path": checkpoint_path},
                )
                response.raise_for_status()
        except Exception as e:
            print(f"FATAL: Weight update HTTP request failed: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(
                f"Failed to update weights from {checkpoint_path}: {e}"
            ) from e

        # Verify server is still healthy after weight update
        print("Verifying SGLang server health after weight update...")
        for i in range(10):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    health_response = await client.get(f"{self.server_url}/health")
                    if health_response.status_code == 200:
                        print(f"Server healthy after weight update (attempt {i+1})")
                        break
            except Exception as e:
                print(f"Server health check failed (attempt {i+1}/10): {e}")
                if i == 9:
                    raise RuntimeError(
                        f"Server became unhealthy after weight update: {e}"
                    ) from e
                time.sleep(2)
        else:
            raise RuntimeError("Server health check timed out after weight update")

        # Update version tracking
        old_version = self.checkpoint_version
        self.checkpoint_version = new_version

        print(f"Successfully updated weights from v{old_version} to v{new_version}")
        return True


if __name__ == "__main__":
    img = (
        kt.Image(image_id="lmsysorg/sglang:latest")
        # .run_bash("apt-get update && apt-get install -y libnuma-dev")
        # .run_bash("uv pip install --system --break-system-packages sgl_kernel")
        .run_bash(
            "uv pip install --break-system-packages --system 'git+https://github.com/ClawSeven/sglang.git@dev-dllm#subdirectory=python'"
        )
    )
    compute = kt.Compute(gpus=1, image=img)

    infer = kt.cls(SGLang).to(compute)
    result = infer.generate(["Why does camus say sisyphus is happy?"])
    print(result)
