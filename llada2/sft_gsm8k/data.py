import json

import torch


def apply_chat_template(messages, tokenizer, max_seq_len):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return input_ids, labels


def sft_noise_transition(input_ids, labels, noise_range, mask_token_id):
    noise_level = (
        torch.rand(1).item() * (noise_range[1] - noise_range[0]) + noise_range[0]
    )

    # If labels is None, treat all tokens as valid for masking (RL use case)
    # Otherwise, only mask tokens that aren't padding (SFT use case)
    if labels is None:
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        valid_mask = labels != -100

    num_valid = valid_mask.sum().item()
    num_to_mask = int(num_valid * noise_level)

    noisy_input_ids = input_ids.clone()
    if num_to_mask > 0:
        valid_indices = torch.where(valid_mask)[0]
        perm = torch.randperm(len(valid_indices))[:num_to_mask]
        mask_indices = valid_indices[perm]
        noisy_input_ids[mask_indices] = mask_token_id

    return noisy_input_ids


def process_example(example, tokenizer, max_seq_len, noise_range, mask_token_id):
    messages = example["messages"]
    input_ids, labels = apply_chat_template(messages, tokenizer, max_seq_len)
    noisy_input_ids = sft_noise_transition(
        input_ids, labels, noise_range, mask_token_id
    )

    return {
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "labels": labels,
    }


class LLaDA2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len, noise_range, mask_token_id):
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.noise_range = noise_range
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return process_example(
            self.data[idx],
            self.tokenizer,
            self.max_seq_len,
            self.noise_range,
            self.mask_token_id,
        )


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "noisy_input_ids": torch.stack([x["noisy_input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }
