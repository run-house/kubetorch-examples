# # Online Embedding Inference with Triton Server

# This example demonstrates how to deploy a high-performance embedding service using NVIDIA's Triton.
# Specifically, we deploy an embedding model (BGE-Large-EN-v1.5) by downloading
# a pre-built ONNX model, configuring Triton server, and exposing a simple
# API for text embedding generation.
#
# Kubetorch allows you to use Python to deploy Triton Inference Server as an
# autoscaling managed service simply by calling `kt deploy embedding.py`.
# Then, we can call against that inference service directly in code as we demonstrate in
# `main` with 30 threads in a pool, and those calls propagate to my service. You can call
# this service from anywhere, either using Kubetorch as a client library (simply import the decorated
# class), or by regular HTTP posts from within your VPC.
import os
import shlex
import subprocess
import time

import kubetorch as kt

# Triton model configuration for serving the ONNX model
TRITON_MODEL_CONFIG = """
name: "embedding_model"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  }
]

output [
  {
    name: "last_hidden_state"
    data_type: TYPE_FP32
    dims: [ -1, -1, 1024 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
"""
MODEL_URL = "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/onnx/model.onnx"

# ## Image and Compute Setup
# We use the official NVIDIA Triton image (including CUDA) and install the necessary Python dependencies
# for the embedding service. Then, we define the required compute, which simply requesting 1 GPU
# per replica, while the `autoscale` API allows us to define the min (scale to zero) and max (5 replicas)
# copies of our service. Finally the Class we define is helpful, because it enables you to implement rich
# lazy loading logic or input pre-processing, as we lightly do here.
triton_img = kt.Image(image_id="nvcr.io/nvidia/tritonserver:25.06-py3").run_bash(
    "uv pip install --system --break-system-packages transformers==4.53.1 torch==2.7.1 tritonclient[grpc]==2.59.0"
)


@kt.compute(gpus="1", image=triton_img)
@kt.autoscale(initial_replicas=1, min_replicas=0, max_replicas=4, concurrency=1000)
class Embedder:
    def __init__(self, model_name_or_path="BAAI/bge-large-en-v1.5"):
        import tritonclient.grpc as grpcclient

        from transformers import AutoTokenizer

        # Set up model paths and server configuration
        self.model_dir = "/models"
        self.triton_model_dir = "/models/embedding_model"
        self.server_url = "localhost:8001"  # gRPC port

        # Download and initialize the tokenizer
        print("Downloading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Create Triton model directory structure
        os.makedirs(f"{self.triton_model_dir}/1", exist_ok=True)

        # Download the pre-built ONNX model if not already present
        onnx_path = f"{self.triton_model_dir}/1/model.onnx"
        if not os.path.exists(onnx_path):
            print("Downloading pre-built ONNX model...")
            import urllib.request

            urllib.request.urlretrieve(MODEL_URL, onnx_path)
        else:
            print("ONNX model already exists, skipping download.")

        # Create Triton model configuration if not already present
        config_path = f"{self.triton_model_dir}/config.pbtxt"
        with open(config_path, "w") as f:
            f.write(TRITON_MODEL_CONFIG)

        # Check if the triton server is already running, or starting it again will fail
        server_start_cmd = f"tritonserver --model-repository {self.model_dir} --http-port 8000 --grpc-port 8001 --log-verbose 1"  # Set log verbosity level
        try:
            self.client = grpcclient.InferenceServerClient(url=self.server_url)
            if self.client.is_server_ready():
                print("Triton server is already running.")
                return
        except grpcclient.InferenceServerException:
            print(f"Starting Triton server with command `{server_start_cmd}`")

        # Start Triton server with gRPC enabled
        self.triton_process = subprocess.Popen(shlex.split(server_start_cmd), text=True)

        # Wait for server to be ready with health checks
        print("Waiting for Triton server to be ready...")
        max_retries = 30
        self.client = None
        for i in range(max_retries):
            try:
                self.client = grpcclient.InferenceServerClient(url=self.server_url)
                if self.client.is_server_ready():
                    print("Triton server is ready!")
                    break
            except Exception:
                if i == max_retries - 1:
                    raise Exception("Triton server failed to start")
                time.sleep(2)

    def embed(self, text, **embed_kwargs):
        import numpy as np
        import tritonclient.grpc as grpcclient

        # Handle single text or list of texts
        texts = [text] if isinstance(text, str) else text

        # Tokenize the input texts with padding and truncation
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Convert PyTorch tensors to numpy arrays for Triton
        input_ids = inputs["input_ids"].numpy().astype(np.int64)
        attention_mask = inputs["attention_mask"].numpy().astype(np.int64)
        token_type_ids = (
            inputs.get("token_type_ids", np.zeros_like(input_ids))
            .numpy()
            .astype(np.int64)
        )

        # Prepare inputs for Triton client
        triton_inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
            grpcclient.InferInput("token_type_ids", token_type_ids.shape, "INT64"),
        ]

        # Set input data
        triton_inputs[0].set_data_from_numpy(input_ids)
        triton_inputs[1].set_data_from_numpy(attention_mask)
        triton_inputs[2].set_data_from_numpy(token_type_ids)

        # Prepare output specification
        triton_outputs = [grpcclient.InferRequestedOutput("last_hidden_state")]

        # Run inference using gRPC client
        response = self.client.infer(
            "embedding_model", triton_inputs, outputs=triton_outputs
        )

        # Get output data as numpy array
        embeddings = response.as_numpy("last_hidden_state")

        # Apply mean pooling using attention mask to convert token embeddings to sentence embeddings
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]

        # Reshape attention mask for broadcasting
        attention_mask_expanded = attention_mask.reshape(batch_size, sequence_length, 1)

        # Apply weighted mean pooling: sum(embeddings * attention_mask) / sum(attention_mask)
        embeddings = (embeddings * attention_mask_expanded).sum(
            axis=1
        ) / attention_mask_expanded.sum(axis=1)

        # Optional L2 normalization for cosine similarity calculations
        if embed_kwargs.get("normalize_embeddings", False):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        # Return single embedding or list based on input type
        return embeddings[0].tolist() if isinstance(text, str) else embeddings.tolist()


# ## Testing and Benchmarking
# The following code demonstrates how to test the embedding service and measure its performance
# under concurrent load. This is useful for understanding the throughput capabilities of your
# Triton-based embedding service.

# Sample text for testing (simulated log data)
SAMPLE_TEXT = """INFO | 2025-07-10 04:08:03 | kubetorch.servers.http.http_server:360 | Rsyncing over updated code with command: rsync -av rsync://10.4.0.205:873/data/ .
INFO | 2025-07-10 04:08:03 | kubetorch.servers.http.http_server:67 | Clearing callables cache.
INFO | 2025-07-10 04:08:03 | kubetorch.servers.http.http_server:93 | Running image setup steps:
INFO | 2025-07-10 04:08:03 | kubetorch.servers.http.http_server:124 | Running image setup with: uv pip install --system --break-system-packages transformers requests numpy torch
INFO | 2025-07-10 04:08:36 | kubetorch.servers.http.http_server:116 | Setting env var KT_FILE_PATH
INFO | 2025-07-10 04:08:36 | kubetorch.servers.http.http_server:116 | Setting env var KT_CALLABLE_TYPE
INFO | 2025-07-10 04:08:36 | kubetorch.servers.http.http_server:116 | Setting env var KT_DISTRIBUTED_CONFIG
INFO | 2025-07-10 04:08:36 | kubetorch.servers.http.http_server:173 | Cleared cache and reloaded callable from metadata
INFO | 2025-07-10 04:08:36 | kubetorch.servers.http.http_server:295 | Reloading in module embedding
Downloading tokenizer...
{"asctime": "2025-07-10 04:08:47", "name": "print_redirect", "levelname": "INFO", "message": "Downloading tokenizer...", "request_id": "244bffd0b9", "pod": "donny-embedder-00001-deployment-d768988cc-nrks7"}
Downloading pre-built ONNX model...
{"asctime": "2025-07-10 04:08:48", "name": "print_redirect", "levelname": "INFO", "message": "Downloading pre-built ONNX model...", "request_id": "244bffd0b9", "pod": "donny-embedder-00001-deployment-d768988cc-nrks7"}
Starting Triton server...
{"asctime": "2025-07-10 04:08:56", "name": "print_redirect", "levelname": "INFO", "message": "Starting Triton server...", "request_id": "244bffd0b9", "pod": "donny-embedder-00001-deployment-d768988cc-nrks7"}
Waiting for Triton server to be ready...
{"asctime": "2025-07-10 04:08:56", "name": "print_redirect", "levelname": "INFO", "message": "Waiting for Triton server to be ready...", "request_id": "244bffd0b9", "pod": "donny-embedder-00001-deployment-d768988cc-nrks7"}
Triton server is ready!
{"asctime": "2025-07-10 04:09:00", "name": "print_redirect", "levelname": "INFO", "message": "Triton server is ready!", "request_id": "244bffd0b9", "pod": "donny-embedder-00001-deployment-d768988cc-nrks7"}
{"asctime": "2025-07-10 04:09:01", "name": "uvicorn.access", "levelname": "INFO", "message": "10.4.0.61:0 - \"POST /Embedder/embed HTTP/1.1\" 200", "request_id": "-", "pod": "donny-embedder-00001-deployment-d768988cc-nrks7"}"""
TEST_TEXT_LINES = [SAMPLE_TEXT.split("\n")] * 40

if __name__ == "__main__":
    # You can call `kt deploy embedding.py` or uncomment the below
    # Embedder.deploy()

    # Test single embedding generation
    res = Embedder.embed("This is a test sentence.", normalize_embeddings=True)
    print(res)

    # ## Performance Benchmarking
    # Test concurrent embedding generation to measure throughput
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    from tqdm import tqdm

    # Create a partial function for embedding with logging disabled
    embed_thread = partial(Embedder.embed, stream_logs=False)

    # Run concurrent embedding requests to measure throughput
    with ThreadPoolExecutor(max_workers=30) as executor:
        start = time.time()
        list(
            tqdm(
                executor.map(embed_thread, TEST_TEXT_LINES),
                total=len(TEST_TEXT_LINES),
                desc="Processing embeddings",
            )
        )
    print(
        f"Processed {(len(TEST_TEXT_LINES) * len(TEST_TEXT_LINES[0])) / (time.time() - start):.2f} inferences / second"
    )
