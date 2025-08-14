import re

import datasets


def download_model(model_name, local_download_path):
    """Download the model from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading model {model_name} to {local_download_path}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_download_path,
    )


# From https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py
def download_data(
    data_source="openai/gsm8k",
    train_path="/data/gsm8k/train.parquet",
    val_path="/data/gsm8k/test.parquet",
):

    dataset = datasets.load_dataset(data_source, "main")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        'Let\'s think step by step and output the final answer after "####".'
    )

    def make_map_fn(split):
        def extract_solution(solution_str):
            solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
            assert solution is not None
            final_solution = solution.group(0)
            final_solution = final_solution.split("#### ")[1].replace(",", "")
            return final_solution

        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)
