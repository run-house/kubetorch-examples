"""
vLLM inference wrapper with GPU tensor LoRA loading.

Hijack is adapted from verl (https://github.com/volcengine/verl).
"""

import json
import os
import threading
import time
from contextlib import contextmanager


class PriorityLock:
    """Lock that gives LoRA updates priority over inference."""

    def __init__(self):
        self._cond = threading.Condition()
        self._held = False
        self._priority_waiting = False

    @contextmanager
    def _acquire(self, priority=False):
        with self._cond:
            if priority:
                self._priority_waiting = True
            while self._held or (not priority and self._priority_waiting):
                self._cond.wait()
            self._held = True
            if priority:
                self._priority_waiting = False
        try:
            yield
        finally:
            with self._cond:
                self._held = False
                self._cond.notify_all()

    def inference(self):
        return self._acquire(priority=False)

    def lora_update(self):
        return self._acquire(priority=True)


class vLLM:
    """vLLM wrapper with GPU tensor LoRA loading support."""

    _lock = PriorityLock()
    _initialized = False
    _TensorLoRARequest = None
    _LoRARequest = None

    @classmethod
    def _init_vllm(cls):
        """One-time vLLM setup: env vars, imports, and monkey-patching."""
        if cls._initialized:
            return

        # Must set before any vLLM import
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        from msgspec import field
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.lora.request import LoRARequest
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

        cls._LoRARequest = LoRARequest

        # TensorLoRARequest accepts GPU tensors directly instead of file paths
        class TensorLoRARequest(LoRARequest):
            peft_config: dict = field(default=None)
            lora_tensors: dict = field(default=None)

        cls._TensorLoRARequest = TensorLoRARequest

        # Patch vLLM to support TensorLoRARequest
        def patched_load_adapter(self, lora_request):
            supported = self._adapter_manager.supported_lora_modules
            packed = self._adapter_manager.packed_modules_mapping
            expected = []
            for mod in supported:
                expected.extend(packed.get(mod, [mod]))
            expected = list(set(expected))

            model = self._adapter_manager.model
            mapper = getattr(model, "hf_to_vllm_mapper", None)

            if isinstance(lora_request, TensorLoRARequest):
                peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
                peft_helper.validate_legal(self.lora_config)

                # Normalize PEFT names: .lora_A.default.weight -> .lora_A.weight
                tensors = {
                    k.replace(".lora_A.default.", ".lora_A.").replace(
                        ".lora_B.default.", ".lora_B."
                    ): v
                    for k, v in lora_request.lora_tensors.items()
                }

                return self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=tensors,
                    peft_helper=peft_helper,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=self.vocab_size
                    + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=mapper,
                )
            else:
                from vllm.lora.utils import get_adapter_absolute_path

                path = get_adapter_absolute_path(lora_request.lora_path)
                peft_helper = PEFTHelper.from_local_dir(
                    path, self.max_position_embeddings
                )
                peft_helper.validate_legal(self.lora_config)

                lora = self._lora_model_cls.from_local_checkpoint(
                    path,
                    expected,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    target_embedding_padding=self.vocab_size
                    + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=mapper,
                )

                if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                    raise ValueError(
                        f"LoRA vocab size {lora.extra_vocab_size} > "
                        f"lora_extra_vocab_size {self.lora_config.lora_extra_vocab_size}"
                    )
                return lora

        LRUCacheWorkerLoRAManager._load_adapter = patched_load_adapter
        cls._initialized = True

    def __init__(
        self,
        model_id: str,
        lora_checkpoint: str = None,
        checkpoint_version: int = 0,
        kt_cached_state: dict = None,
        engine_config: dict = None,
        peft_config: dict = None,
    ):
        self._init_vllm()
        from vllm import LLM

        self.model_id = model_id
        self.peft_config = peft_config
        self.checkpoint_version = checkpoint_version
        self.current_lora_request = None
        self._lora_int_id = 1
        self._cached_metadata = None

        # Reuse cached engine if available
        if kt_cached_state and kt_cached_state.get("inference_engine"):
            print(f"Reusing cached vLLM engine (v{checkpoint_version})")
            self.engine = kt_cached_state["inference_engine"]
            self.tokenizer = kt_cached_state["tokenizer"]
            return

        print("Creating vLLM engine (single-process mode)")
        config = {
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.80,
            "max_model_len": 2048,
            "enforce_eager": True,
            "enable_lora": True,
            "max_lora_rank": 64,
            **(engine_config or {}),
        }

        self.engine = LLM(model=model_id, **config)
        self.tokenizer = self.engine.get_tokenizer()

        if lora_checkpoint and os.path.exists(lora_checkpoint):
            self.load_lora_adapter(lora_checkpoint)

        # Start checkpoint poller (will auto-discover metadata from store)
        self.start_checkpoint_poller()

    def __kt_cached_state__(self):
        """State to cache across kubetorch reloads."""
        return {"inference_engine": self.engine, "tokenizer": self.tokenizer}

    def load_lora_adapter(self, lora_path: str):
        """Load LoRA adapter from file path."""
        lora_id = f"adapter_{hash(lora_path)}"
        self.current_lora_request = self._LoRARequest(
            lora_name=lora_id,
            lora_int_id=hash(lora_id) % 100000,
            lora_local_path=lora_path,
        )
        with self._lock.lora_update():
            self.engine.llm_engine.add_lora(self.current_lora_request)
        print(f"Loaded LoRA from {lora_path}")

    def load_lora_from_tensors(self, lora_tensors: dict, new_version: int = None):
        """Load LoRA weights directly from GPU tensors."""
        if self.peft_config is None:
            raise ValueError("peft_config required for tensor-based LoRA loading")

        if new_version is not None:
            self.checkpoint_version = new_version

        with self._lock.lora_update():
            if self.current_lora_request:
                try:
                    self.engine.llm_engine.remove_lora(
                        self.current_lora_request.lora_int_id
                    )
                except Exception:
                    pass

            self._lora_int_id += 1
            self.current_lora_request = self._TensorLoRARequest(
                lora_name=f"adapter_v{self.checkpoint_version}",
                lora_int_id=self._lora_int_id,
                lora_path="N/A",
                peft_config=self.peft_config,
                lora_tensors=lora_tensors,
            )
            self.engine.llm_engine.add_lora(self.current_lora_request)

    def load_lora_from_store(
        self, key: str, metadata: dict = None, new_version: int = None
    ):
        """Load LoRA weights from kubetorch data store."""
        import kubetorch as kt
        import torch

        # Cache metadata for future polling
        if metadata is not None:
            self._cached_metadata = metadata
        elif self._cached_metadata is not None:
            metadata = self._cached_metadata
        else:
            raise ValueError("metadata required for first load")

        # # Clear old LoRA tensors to free GPU memory
        # if self.current_lora_request is not None and hasattr(self.current_lora_request, 'lora_tensors'):
        #     del self.current_lora_request.lora_tensors
        #     torch.cuda.empty_cache()

        dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
        }
        dest = {
            name: torch.empty(
                info["shape"],
                dtype=dtype_map.get(info["dtype"], torch.bfloat16),
                device="cuda",
            )
            for name, info in metadata.items()
        }

        total_bytes = sum(t.numel() * t.element_size() for t in dest.values())
        print(
            f"[LOAD] Fetching {len(dest)} tensors, {total_bytes/1e6:.2f} MB from '{key}'"
        )

        t0 = time.time()
        kt.get(key=key, dest=dest, verbose=True)
        print(
            f"[LOAD] kt.get() took {time.time()-t0:.3f}s ({total_bytes/(time.time()-t0)/1e6:.1f} MB/s)"
        )

        t1 = time.time()
        self.load_lora_from_tensors(dest, new_version=new_version)
        print(f"[LOAD] load_lora_from_tensors() took {time.time()-t1:.3f}s")

    def start_checkpoint_poller(self, poll_interval: float = 10.0):
        """Start background thread polling for new lora/v{N} checkpoints via kt.ls()."""
        import re

        import kubetorch as kt

        def poll_loop():
            while not getattr(self, "_stop_poller", False):
                try:
                    # First, discover metadata if not cached
                    if self._cached_metadata is None:
                        keys = kt.ls("lora/")
                        if any(
                            item.get("name", "").startswith("metadata") for item in keys
                        ):
                            # Fetch metadata JSON file to /tmp directory
                            kt.get(key="lora/metadata", dest="/tmp/")
                            with open("/tmp/metadata/lora_metadata.json") as f:
                                self._cached_metadata = json.load(f)
                            print(
                                f"[POLLER] Discovered metadata ({len(self._cached_metadata)} tensors)"
                            )
                        else:
                            time.sleep(poll_interval)
                            continue

                    # Poll for new versions
                    keys = kt.ls("lora/")
                    versions = []
                    for item in keys:
                        match = re.match(r"v(\d+)/?$", item.get("name", ""))
                        if match:
                            versions.append(int(match.group(1)))
                    if versions:
                        latest = max(versions)
                        if latest > self.checkpoint_version:
                            print(f"[POLLER] v{latest} available, loading...")
                            t0 = time.time()
                            self.load_lora_from_store(
                                f"lora/v{latest}", new_version=latest
                            )
                            load_time = time.time() - t0
                            print(f"[POLLER] v{latest} loaded in {load_time:.3f}s")
                            # Clean up old versions
                            t1 = time.time()
                            cleaned = 0
                            for v in versions:
                                if v < latest:
                                    try:
                                        kt.rm(f"lora/v{v}", recursive=True)
                                        cleaned += 1
                                    except Exception:
                                        pass
                            if cleaned:
                                print(
                                    f"[POLLER] Cleaned up {cleaned} old versions in {time.time()-t1:.3f}s"
                                )
                except Exception as e:
                    print(f"[POLLER] Error: {e}")
                time.sleep(poll_interval)

        self._stop_poller = False
        threading.Thread(target=poll_loop, daemon=True).start()
        print(f"[POLLER] Started (interval={poll_interval}s)")

    def generate(self, prompts: list, request_version: int = None, **kwargs):
        """Generate completions for prompts."""
        from vllm import SamplingParams

        with self._lock.inference():
            outputs = self.engine.generate(
                prompts,
                SamplingParams(**kwargs),
                lora_request=self.current_lora_request,
            )

        return (
            [o.outputs[0].text for o in outputs],
            [list(o.outputs[0].token_ids) for o in outputs],
        )
