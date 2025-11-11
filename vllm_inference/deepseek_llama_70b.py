# # Launch DeepSeek-R1 on Your Own Cloud Account
#
# The Llama-70B distill of DeepSeek R1 is the most powerful and largest of the
# DeepSeek distills, but still easily fits on a single node of GPUs - even 8 x L4s with minor optimizations.
# Of course, inference speed will improve if you change to A100s or H100s on your cloud provider,
# but the L4s are cost-effective for experimentation or low throughput, latency-agnostic workloads, and also
# fairly available on spot meaning you can serve this model for as low as ~$4/hour. This is the full model,
# not a quantization of it.
#
# On benchmarks, this Llama70B distillation meets or exceeds the performance of GPT-4o-0513,
# Claude-3.5-Sonnet-1022, and o1-mini across most quantitative and coding tasks. Real world
# quality of output depends on your use case. It will take some time to download the model
# to the remote machine on the first run. Further iterations will take no time at all.
#
# To deploy the service, simply call `kt deploy deepseek_llama_70b.py` on this file and
# we create a proper Kubernetes service with scale to zero and autoscaling for concurrency.
# Then, you can import and use this class as is, which we show below in the main function.
import os

import kubetorch as kt
from vllm import LLM, SamplingParams

# ## Launch Compute and Create Service
# We will define compute using Kubetorch and send our inference class to the remote compute.
# First, we define an image with torch and vllm and 8 x L4 with 1 node.
# Then, we send our inference class to the remote compute and instantiate a the remote inference class
# with the name `deepseek` which we can access by name below, or from any other
# Python process which imports DeepSeekDistillLlama70BvLLM. This also creates a proper service
# in Kubernetes that you can call over HTTP.
img = kt.Image(image_id="vllm/vllm-openai:latest").sync_secrets(["huggingface"])


@kt.compute(gpus="8", gpu_type="L4", image=img, name="deepseek_llama")
@kt.autoscale(initial_scale=1, min_scale=0, max_scale=5, concurrency=100)
class DeepSeekDistillLlama70BvLLM:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"):
        self.model_id = model_id
        self.model = None

    def load_model(self):
        print("loading model")
        self.model = LLM(
            self.model_id,
            tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,  # Reduces size of KV store
        )
        print("model loaded")

    def generate(self, queries, temperature=0.65, top_p=0.95, max_tokens=5120, min_tokens=32):
        if self.model is None:
            self.load_model()

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

        outputs = self.model.generate(queries, sampling_params)
        return outputs


# ## Call the Inference
# Once we call `kt deploy llama.py,` an autoscaling service is stood up that we can use directly in our programs.
# We are calling the service in our script here, but you could identically call the service from within your FastAPI app,
# an orchestrator for batch inference, or anywhere else.

if __name__ == "__main__":
    # Run inference remotely and print the results
    queries = [
        "What is the relationship between bees and a beehive compared to programmers and...?",
        "How many R's are in Strawberry?",
        """Roman numerals are formed by appending the conversions of decimal place values from highest to lowest.
        Converting a decimal place value into a Roman numeral has the following rules: If the value does not start with 4 or 9, select the symbol of the maximal value that can be subtracted from the input, append that symbol to the result, subtract its value, and convert the remainder to a Roman numeral.
        If the value starts with 4 or 9 use the subtractive form representing one symbol subtracted from the following symbol, for example, 4 is 1 (I) less than 5 (V): IV and 9 is 1 (I) less than 10 (X): IX. Only the following subtractive forms are used: 4 (IV), 9 (IX), 40 (XL), 90 (XC), 400 (CD) and 900 (CM). Only powers of 10 (I, X, C, M) can be appended consecutively at most 3 times to represent multiples of 10. You cannot append 5 (V), 50 (L), or 500 (D) multiple times. If you need to append a symbol 4 times use the subtractive form.
        Given an integer, write and return Python code to convert it to a Roman numeral.""",
    ]

    outputs = DeepSeekDistillLlama70BvLLM.generate(queries, temperature=0.7)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")
