# ## Hello World: Deploy an LLM Inference Service
# Our inference `Hello, World` has you deploying an open LLM (Llama 3 in this case) to 1 or more GPUs in the cloud,
# using vLLM to serve the model due to it's high performance. All you have to do is call `kt deploy llama.py` to
# deploy the service, and then you can run `python llama.py` to see how you can call the remote service from
# your local machine.

# ### Decorate a Regular Inference Class
# We start with a regular inference class called `LlamaModel` which has a `generate()` method that runs inference with vLLM.
# Then, we apply our decorators which will allow this service to be deployed with `kt deploy.`
#
# In the `kt.compute` decorator, we request 1 GPU for each replica of the inference class; our cluster is configured with an L4, but you can easily
# specify the exact compute requirements here, including a specific GPU type, CPU, memory, etc. We also specify that we want
# it to start with a public PyTorch image with vllm installed additionally; you can launch the service with your team's base
# Docker image, but optionally (especially during development iteration) lightly modify that image without having to rebuild
# that image fully.
#
# We also use the `kt.autoscale` decorator to specify the autoscale conditions for this service ; this service will
# scale to 0, have a maximum scale of 5, and autoscale up when there are 100 concurrent requests. There are many more
# flexible options you have here as well.
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
@kt.autoscale(initial_scale=1, min_scale=0, max_scale=5, concurrency=100)
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
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

        req_output = self.model.generate(queries, sampling_params)
        return [output.outputs[0].text for output in req_output]


# ### Call the Inference
# Once we call `kt deploy llama.py,` an autoscaling service is stood up that we can use directly in our programs.
# We are calling the service in our script here, but you could identically call the service from within your FastAPI app,
# an orchestrator for batch inference, or anywhere else.
if __name__ == "__main__":
    # LlamaModel.deploy() # Uncomment to test the LlamaModel service locally.
    res = LlamaModel.generate(["Who is Randy Jackson?"], max_tokens=100)
    print(res)
