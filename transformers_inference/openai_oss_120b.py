# # Deploy OpenAI GPT OSS 120B as an Inference Service
# This inference example walks through deploying OpenAI's new open LLM (120B) to 8 GPUs in the cloud.
# We'll use [Transformers' AutoModel](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html)
# to serve the model due to the simplicity of deploying the model across multiple GPUs.
# To deploy the service to your cloud, you'll can simply run `kt deploy openai_oss_120b.py`,
# and then you can run `python openai_oss_120b.py` to run main and see how you can call the remote service from
# any Python script that imports the inference class.
#
#
# ## Decorate a Regular Inference Class
# We start with a regular inference class called `OpenAIOSSInference` which has a `generate()` method
# that runs inference with vLLM.
#
# Then, we apply our decorators which will allow this service to be deployed with `kt deploy.`
# In the `kt.compute` decorator, we request 8 GPU for each replica of the inference class;
# our cluster is configured with an A100s, but you can easily specify the exact compute requirements here,
# including a specific GPU type, CPU, memory, etc. We also specify that we want
# it to start with a public NVIDIA CUDA-Python and install a few packages; you can launch the service with your team's base
# Docker image, but optionally (especially during development iteration) lightly modify that image without having to rebuild
# that image fully.
#
# We also could use the `kt.autoscale` decorator to specify the autoscale conditions for this service; this service will
# scale to 0, have a maximum scale of 5, and autoscale up when there are 100 concurrent requests. There are many more
# flexible options you have here as well.

import kubetorch as kt
from transformers import AutoModelForCausalLM, AutoTokenizer

img = (
    kt.Image(image_id="nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.6")
    .run_bash("pip install --upgrade pip setuptools wheel")
    .pip_install(["torch", "transformers", "accelerate", "kernels"])
    # For H100 / H200, which support MXFP4 quantization
    # .pip_install(['triton>=3.4.0'])
    # .run_bash('git clone --branch main https://github.com/triton-lang/triton.git'
    # .run_bash('uv pip install --system ./triton/python/triton_kernels')
)


@kt.compute(gpus="8", image=img, name="openai_oss", secrets=["huggingface"])
@kt.autoscale(initial_scale=1, min_scale=0, max_scale=5, concurrency=100)
class OpenAIOSSInference:
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        import torch
        from accelerate import infer_auto_device_map, init_empty_weights

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16
            )

        device_map = infer_auto_device_map(
            self.model,
            max_memory={i: "30GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["GptOssDecoderLayer"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="bfloat16", device_map=device_map
        )

    def generate(
        self, input, add_generation_prompt=True, max_tokens=200, temperature=0.7
    ):
        if self.model is None:
            self.load_model()

        messages = [
            {"role": "user", "content": input},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature
        )
        decoded_output = self.tokenizer.decode(outputs[0])
        print(decoded_output)
        return decoded_output


# ## Call the Inference
# Once we call `kt deploy openai_oss_120b.py,` an autoscaling service is stood up that we can use directly in our programs.
# We are calling the service in our script here, but you could identically call the service from within your FastAPI app,
# an orchestrator for batch inference, or anywhere else. Since the Inference class is now a proper Kubernetes
# service, you could also call it over HTTP from within your VPC, if you set the `visibility` to `vpc` in the
# compute decorator.

if __name__ == "__main__":
    prompt = "What is the best chocolate chip cookie?"

    output = OpenAIOSSInference.generate(
        prompt, max_tokens=200, add_generation_prompt=False
    )
    print(f"Prompt: {prompt}, Generated text: {output}")
