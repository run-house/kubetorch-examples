import os


class vLLM:
    """vLLM wrapper with hot-swapping support using AsyncLLMEngine."""

    def __init__(
        self,
        model_id,
        lora_checkpoint=None,
        checkpoint_version=0,
        kt_cached_state=None,
        engine_config=None,
    ):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.current_lora_request = None
        self.checkpoint_version = checkpoint_version
        self.model_id = model_id

        if kt_cached_state and kt_cached_state.get("model") is not None:
            print(
                f"Reusing AsyncLLMEngine from cache (version {self.checkpoint_version})"
            )
            self.model = kt_cached_state["model"]
            if lora_checkpoint and os.path.exists(lora_checkpoint):
                self.load_lora_adapter(lora_checkpoint)
            return

        print(f"Creating new AsyncLLMEngine (version {self.checkpoint_version})")

        # Default engine config
        default_config = {
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.95,
            "max_model_len": 2048,
            "enforce_eager": False,
            "enable_lora": True,
            "max_lora_rank": 64,
        }
        if engine_config:
            default_config.update(engine_config)

        engine_args = AsyncEngineArgs(
            model=model_id,
            **default_config,
        )

        self.model = AsyncLLMEngine.from_engine_args(engine_args)

        if lora_checkpoint and os.path.exists(lora_checkpoint):
            self.load_lora_adapter(lora_checkpoint)

    def __kt_cached_state__(self):
        """Return state to be cached by Kubetorch across reloads."""
        return {"model": self.model}

    def load_lora_adapter(self, lora_path):
        """Hot-swap LoRA adapter without restarting."""
        from vllm.lora.request import LoRARequest

        lora_id = f"adapter_{hash(lora_path)}"
        self.current_lora_request = LoRARequest(
            lora_name=lora_id,
            lora_int_id=hash(lora_id) % 100000,
            lora_local_path=lora_path,
        )
        print(f"LoRA adapter loaded from {lora_path}")

    async def generate(self, prompts, request_version=None, **kwargs):
        import asyncio
        import uuid

        from vllm import SamplingParams

        if request_version is not None and request_version != self.checkpoint_version:
            print(
                f"Ignoring stale request from version {request_version} "
                f"(current: {self.checkpoint_version})"
            )
            return [""] * len(prompts), [[]] * len(prompts)

        sampling_params = SamplingParams(**kwargs)

        async def process_single_prompt(prompt):
            request_id = str(uuid.uuid4())
            result_generator = self.model.generate(
                prompt,
                sampling_params,
                request_id,
                lora_request=self.current_lora_request
                if self.current_lora_request
                else None,
            )

            async for output in result_generator:
                if output.finished:
                    return output
            return None

        tasks = [process_single_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        completions = []
        token_ids = []
        for result in results:
            if result:
                completions.append(result.outputs[0].text)
                token_ids.append(result.outputs[0].token_ids)
            else:
                completions.append("")
                token_ids.append([])

        return completions, token_ids


class vLLMSync:
    """Synchronous vLLM wrapper with LoRA hot-swap support."""

    def __init__(
        self,
        model_id,
        lora_checkpoint=None,
        kt_cached_state=None,
    ):
        from vllm import LLM

        self.model_id = model_id
        self.base_model_id = model_id
        self.current_lora_request = None

        if kt_cached_state and "model" in kt_cached_state:
            print("Reusing existing vLLM engine from Kubetorch cached state")
            self.model = kt_cached_state["model"]

            if lora_checkpoint and os.path.exists(lora_checkpoint):
                print(f"Hot-swapping LoRA adapter to: {lora_checkpoint}")
                self.load_lora_adapter(lora_checkpoint)
            return

        print("Creating new vLLM engine with LoRA support")
        self.model = LLM(
            self.model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,
            enforce_eager=True,
            enable_lora=True,
            max_lora_rank=32,
            max_loras=4,
        )

        if lora_checkpoint and os.path.exists(lora_checkpoint):
            print(f"Loading LoRA checkpoint: {lora_checkpoint}")
            self.load_lora_adapter(lora_checkpoint)

    def __kt_cached_state__(self):
        """Return state to be cached by Kubetorch across reloads."""
        return {"model": self.model}

    def load_lora_adapter(self, lora_path, lora_id=None):
        """Load a LoRA adapter without restarting the server."""
        import re

        from vllm.lora.request import LoRARequest

        if lora_id is None:
            version_match = re.search(r"v(\d+)", lora_path)
            if version_match:
                lora_id = f"adapter_v{version_match.group(1)}"
            else:
                lora_id = f"adapter_{hash(lora_path)}"

        print(f"Hot-swapping LoRA adapter from {lora_path} with ID {lora_id}")

        self.current_lora_request = LoRARequest(
            lora_name=lora_id,
            lora_int_id=hash(lora_id) % 100000,
            lora_local_path=lora_path,
        )
        self.model_id = lora_path
        print(f"LoRA adapter hot-swapped with ID {lora_id}")

    def generate(
        self,
        queries,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
    ):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            min_p=min_p,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        print(f"Generating response with model_id: {self.model_id}")

        if self.current_lora_request:
            all_outputs = self.model.generate(
                queries, sampling_params, lora_request=self.current_lora_request
            )
        else:
            all_outputs = self.model.generate(queries, sampling_params)

        completions = [
            output.text for outputs in all_outputs for output in outputs.outputs
        ]
        token_ids = [
            output.token_ids for outputs in all_outputs for output in outputs.outputs
        ]
        return completions, token_ids
