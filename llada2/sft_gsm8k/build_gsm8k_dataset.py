import json
import os

from datasets import load_dataset


def format_gsm8k_example_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def main():
    output_dir = "./gsm8k_datasets"

    if os.path.exists(output_dir):
        print(f"Dataset already exists at {output_dir}, skipping.")
        return

    print("Loading gsm8k dataset...")
    raw_dataset = load_dataset("openai/gsm8k", "main")

    print("Converting to messages format...")
    transformed_dataset = raw_dataset.map(
        format_gsm8k_example_to_messages, remove_columns=["question", "answer"]
    )

    os.makedirs(output_dir, exist_ok=True)

    for split, dataset_split in transformed_dataset.items():
        output_filename = os.path.join(output_dir, f"gsm8k_{split}.jsonl")
        with open(output_filename, "w", encoding="utf-8") as f:
            for record in dataset_split:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {split} split to {output_filename}")


if __name__ == "__main__":
    main()
