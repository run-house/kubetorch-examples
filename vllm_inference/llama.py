# # Hello World: Deploy an LLM Inference Service
# This inference _Hello, World_ example walks through deploying an open LLM (Llama 3) to one or more GPUs in the cloud.
# We'll use [vLLM](https://github.com/vllm-project/vllm) to serve the model due to it's high performance.
# To deploy the service to your cloud, you'll can simply run `kt deploy llama.py`. Then, run `python llama.py`
# to see how you can call the remote service from your local machine.
#
# ::youtube[Llama Inference with vLLM]{url="https://www.youtube.com/watch?v=8slAR7459X4"}
#
# ## Decorate a Regular Inference Class
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
    image=kt.Image(image_id="nvcr.io/nvidia/pytorch:24.08-py3").pip_install(
        [
            "--no-build-isolation flash-attn==2.7.3",
            "vllm==0.9.0",
            "'transformers<4.54.0'",
        ]
    ),
    shared_memory_limit="2Gi",  # Recommended by vLLM: https://docs.vllm.ai/en/v0.6.4/serving/deploying_with_k8s.html
    launch_timeout=1200,  # Need more time to load the model
    secrets=["huggingface"],
)
@kt.autoscale(
    initial_scale=1,
    min_scale=1,
    max_scale=5,
)
class LlamaModel:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        from vllm import LLM

        print("Loading model in vLLM:", model_id)
        self.model_id = model_id
        self.model = LLM(
            self.model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,  # Reduces size of KV store
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


# ## Call the Inference
# Once we call `kt deploy llama.py,` an autoscaling service is stood up that we can use directly in our programs.
# We are calling the service in our script here, but you could identically call the service from within your FastAPI app,
# an orchestrator for batch inference, or anywhere else.


def run_inference():
    # LlamaModel.deploy() # Uncomment to test the LlamaModel service locally.
    res = LlamaModel.generate(["Who is Randy Jackson?"], max_tokens=100)
    print(res)


if __name__ == "__main__":
    run_inference()
