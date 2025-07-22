# ### Load RAG LLM Inference Service
# Deploy an open LLM, Llama 3 in this case, to 1 or more GPUs in the cloud.
# We will use vLLM to serve the model due to it's high performance.
#
import kubetorch as kt


@kt.compute(
    gpus="1",
    image=kt.Image(image_id="nvcr.io/nvidia/pytorch:24.08-py3").run_bash(
        "uv pip install --system --break-system-packages vllm==0.9.0"
    ),
    shared_memory_limit="2Gi",  # Recommended by vLLM: https://docs.vllm.ai/en/v0.6.4/serving/deploying_with_k8s.html
    launch_timeout=1200,  # Need more time to load the model
    secrets=["huggingface"],
)
@kt.autoscale(initial_replicas=1, min_replicas=0, max_replicas=4, concurrency=1000)
class LlamaModel:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        from vllm import LLM

        self.model = LLM(
            model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,  # Reduces size of KV store
            enforce_eager=True,
        )

    def generate(
        self, queries, temperature=0.65, top_p=0.95, max_tokens=5120, min_tokens=32
    ):
        """Generate text with proper error handling and model loading"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

        req_output = self.model.generate(queries, sampling_params)
        return [output.outputs[0].text for output in req_output]


# Uncomment to test the LlamaModel service locally.
if __name__ == "__main__":
    # LlamaModel.deploy()
    res = LlamaModel.generate(["Who is Randy Jackson?"], max_tokens=100)
    print(res)
