# # Offline Batch Inference
# We will use BGE to embed a large amount of text. We start with a regular class
# that defines a few methods to load, tokenize, and embed the datasets. Then, we take this class
# and dispatch it to remote compute as an autoscaling service, and call that service in parallel
# to process our data. In practice, you'd want to re-implement the load_data()
# and save_embeddings() methods; we use a public dataset from HuggingFace for convenience.

import asyncio

import kubetorch as kt


class BGEEmbedder:
    def __init__(self, model_id="BAAI/bge-large-en-v1.5"):
        self.model_id = model_id
        self.model = None

    def load_model(self):
        from vllm import LLM

        self.model = LLM(model=self.model_id, task="embed")

    def load_data(
        self,
        dataset_name,
        text_column_name,
        split=None,
        data_files=None,
        batch_size=None,
    ):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        dataset = load_dataset(
            dataset_name,
            data_files=data_files,
            split=split,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name],
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            num_proc=7,
            batched=True,
            remove_columns=dataset.column_names,
            # batch_size=5000,
        )
        inputs = tokenized_dataset["input_ids"]
        return [{"prompt_token_ids": inp} for inp in inputs]

    def embed_dataset(self, dataset_name, text_column_name, split, data_files):
        if self.model is None:
            self.load_model()

        data = self.load_data(
            dataset_name,
            text_column_name=text_column_name,
            split=split,
            data_files=data_files,
        )
        data = data[0:100]
        try:
            results = self.model.embed(data)
            self.save_embeddings([result.outputs.embedding for result in results])
            return {"file": data_files, "status": "success"}
        except Exception as e:
            print(e)
            return {"file": data_files, "status": "failure"}

    def save_embeddings(self, results):
        pass


# ## Defining Compute and Launching the Service
# Here, we will create an autoscaling service, where each replica runs on 1 GPU + 7 CPUs. Then,
# we define it such that each replica has a concurrency of 1 (processes 1 file at a time), and set
# a min / max scale limit for the number of files we run in parallel. In other inference examples, we show a decorator pattern
# but in this example, we will dispatch the embedding using the regular `.to()` since we may want the deployment
# of this embedder service to exist within a pipeline proper.
if __name__ == "__main__":

    replicas = 4

    compute = kt.Compute(
        gpus="1",
        cpus="7",
        memory="25Gi",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:24.08-py3").run_bash(
            "uv pip install --system --break-system-packages vllm==0.9.0 transformers==4.53.0 datasets"
        ),
        launch_timeout=600,
    ).autoscale(min_scale=0, max_scale=replicas, concurrency=1)

    embedder = kt.cls(BGEEmbedder).to(compute)
    embedder.async_ = True

    # Illustrative; you'd manage this elsewhere.
    data_files_list = [
        "20231101.en/train-00000-of-00041.parquet",
        "20231101.en/train-00001-of-00041.parquet",
        "20231101.en/train-00002-of-00041.parquet",
        "20231101.en/train-00003-of-00041.parquet",
        "20231101.en/train-00004-of-00041.parquet",
        "20231101.en/train-00005-of-00041.parquet",
    ]  # ETC

    async def process_files():
        tasks = [
            embedder.embed_dataset("wikimedia/wikipedia", "text", "train", data_file)
            for data_file in data_files_list
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(process_files())
    print(results)
