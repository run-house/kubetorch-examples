"""Shared data processing utilities for LLaDA2 diffusion training.

This module contains data utilities used across different training modes (SFT, RL, etc.):
- Noise transition functions for diffusion process
- Chat template application
- Dataset and collate functions
"""

import json

import torch


def apply_chat_template(messages, tokenizer, max_seq_len):
    """Apply chat template and create labels.

    Args:
        messages: List of chat messages
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length

    Returns:
        Tuple of (input_ids, labels)
    """
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
    """Apply noise transition for diffusion training.

    Randomly masks tokens according to a noise level sampled from noise_range.
    This is used in both SFT and RL training to create the noisy input sequence.

    Args:
        input_ids: Input token IDs [seq_len]
        labels: Labels for masking (None for RL, tensor for SFT)
        noise_range: Tuple of (min_noise, max_noise) e.g., (0.3, 0.8)
        mask_token_id: Token ID to use for masking

    Returns:
        noisy_input_ids: Input IDs with random tokens masked
    """
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
    """Process a single training example for diffusion SFT.

    Args:
        example: Dictionary with 'messages' key
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length
        noise_range: Noise level range for masking
        mask_token_id: Token ID for masking

    Returns:
        Dictionary with input_ids, noisy_input_ids, and labels
    """
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
    """Dataset for LLaDA2 diffusion training.

    Loads JSONL data and applies noise transition for diffusion process.
    Each line should contain a dictionary with a 'messages' key.
    """

    def __init__(self, data_path, tokenizer, max_seq_len, noise_range, mask_token_id):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            noise_range: Tuple of (min_noise, max_noise)
            mask_token_id: Token ID for masking
        """
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
    """Collate function for DataLoader.

    Args:
        batch: List of dictionaries from dataset

    Returns:
        Dictionary with stacked tensors
    """
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "noisy_input_ids": torch.stack([x["noisy_input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }
