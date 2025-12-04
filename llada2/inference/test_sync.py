import kubetorch as kt
from sglang_engine import SGLang
class SyncFrom:
    def __init__(self):
        # self.inference_service = inference_service
        self.local_path = "/llada2_weights"
        self.key = "model-v1/weights"
        self.inference_service = None 
    def download_weights(self, model_id = "inclusionAI/LLaDA2.0-mini-preview", key = "model-v1/weights"):
        """Download weights from HuggingFace"""
        from huggingface_hub import snapshot_download
        import os

        print("Downloading model weights from HuggingFace...")
        
        # Download to a local cache directory
        os.makedirs(self.local_path, exist_ok=True)

        self.local_path = snapshot_download(
            repo_id=model_id,
            local_dir=self.local_path,
        )

        print(f"Model weights downloaded to: {self.local_path}")
        kt.put(key=key, src=self.local_path)
        print("Put!")
        return key

async def main(): 
    img = (
        kt.Image(image_id="lmsysorg/sglang:latest")
        # .run_bash("apt-get update && apt-get install -y libnuma-dev")
        # .run_bash("uv pip install --system --break-system-packages sgl_kernel")
        .run_bash(
            "uv pip install --break-system-packages --system 'git+https://github.com/ClawSeven/sglang.git@dev-dllm#subdirectory=python'"
        )
        .pip_install(["huggingface_hub"])
    )
    compute = kt.Compute(gpus=1, image=img, allowed_serialization=["json", "pickle"])

    infer = kt.cls(SGLang).to(compute, get_if_exists=False)
    infer._async = True 
    result = await infer.generate(["Why does camus say sisyphus is happy?"], max_tokens = 512)
    print(result)
    
    key = "model_v1"
    result = await infer.update_weights_from_disk(key, 2)
    print(result)

    result = await infer.generate(["Why does camus say sisyphus is happy?"], max_tokens = 512)
    print(result)


if __name__ == "__main__": 
    import asyncio 
    asyncio.run(main())
