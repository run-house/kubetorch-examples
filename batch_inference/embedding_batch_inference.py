# # Offline Batch Inference
# We will use BGE to embed a large amount of text.
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

    def embed_dataset(
        self, dataset_name, text_column_name, split, data_files, batch_size=None
    ):
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


# ## Defining compute and the launching embeddings service
# Here, we will create an autoscaling service that
if __name__ == "__main__":

    replicas = 4

    compute = kt.Compute(
        gpus="1",
        cpus="7",
        memory="25Gi",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:24.08-py3")
        .pip_install(["vllm"])
        .run_bash(["pip uninstall pyarrow -y", "pip install pyarrow datasets"]),
        launch_timeout=600,
        inactivity_ttl="2h",
        concurrency=1,
    ).autoscale(min_scale=0, max_scale=replicas)

    embedder = kt.cls(BGEEmbedder).to(compute)
    embedder.load_model()

    data_files_list = [
        "20231101.en/train-00000-of-00041.parquet",
        "20231101.en/train-00001-of-00041.parquet",
        "20231101.en/train-00002-of-00041.parquet",
        "20231101.en/train-00003-of-00041.parquet",
        "20231101.en/train-00004-of-00041.parquet",
        "20231101.en/train-00005-of-00041.parquet",
    ]  # ETC

    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    with ThreadPoolExecutor(max_workers=replicas) as executor:
        embed_file = partial(
            embedder.embed_dataset,
            "wikimedia/wikipedia",
            "text",
            "train",
        )
        results = list(executor.map(embed_file, data_files_list))

    print(results)
