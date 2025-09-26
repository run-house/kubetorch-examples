"""
Dataset processing utilities for LeetCode problems.
Contains functions for loading, processing, and manipulating dataset fields.
"""

import json
from typing import List, Tuple


def extract_dataset_fields(
    dataset, indices: List[int]
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Extract required fields from dataset for given indices"""
    contexts = [dataset[int(idx)]["prompt"] for idx in indices]
    tasks = [dataset[int(idx)]["query"] for idx in indices]
    tests = [dataset[int(idx)]["test"] for idx in indices]
    entrypoints = [dataset[int(idx)]["entry_point"] for idx in indices]
    return contexts, tasks, tests, entrypoints


def extract_test_dataset_fields(
    test_dataset,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Extract all fields from test dataset"""
    contexts = [test_dataset[i]["prompt"] for i in range(len(test_dataset))]
    tasks = [test_dataset[i]["query"] for i in range(len(test_dataset))]
    tests = [test_dataset[i]["test"] for i in range(len(test_dataset))]
    entrypoints = [test_dataset[i]["entry_point"] for i in range(len(test_dataset))]
    return contexts, tasks, tests, entrypoints


def expand_for_multiple_generations(
    contexts: List[str],
    tasks: List[str],
    tests: List[str],
    entrypoints: List[str],
    num_generations: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Expand data for multiple generations per sample"""
    expanded_contexts = []
    expanded_tasks = []
    expanded_tests = []
    expanded_entrypoints = []

    for context, task, test, entrypoint in zip(contexts, tasks, tests, entrypoints):
        expanded_contexts.extend([context] * num_generations)
        expanded_tasks.extend([task] * num_generations)
        expanded_tests.extend([test] * num_generations)
        expanded_entrypoints.extend([entrypoint] * num_generations)

    return expanded_contexts, expanded_tasks, expanded_tests, expanded_entrypoints


def load_leetcode_dataset(dataset_name: str = "newfacade/LeetCodeDataset"):
    """Load the LeetCode dataset"""
    from datasets import load_dataset

    print(f"Loading {dataset_name}...")
    train_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")
    return train_dataset, test_dataset


def save_debug_data(data: List[tuple], filename: str = "debug_data.json"):
    """Save debug data to file for inspection"""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def get_batch_data(
    dataset, indices: List[int], start_idx: int, batch_size: int
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Get a batch of data from the dataset"""
    end_idx = min(start_idx + batch_size, len(indices))
    batch_indices = indices[start_idx:end_idx]
    return extract_dataset_fields(dataset, batch_indices)


class DatasetProcessor:
    """Helper class for processing dataset batches"""

    def __init__(self, dataset, test_dataset):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self._test_data_cache = None

    def get_test_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Get test data, cached for efficiency"""
        if self._test_data_cache is None:
            self._test_data_cache = extract_test_dataset_fields(self.test_dataset)
        return self._test_data_cache

    def get_batch(
        self, indices: List[int], start_idx: int, batch_size: int
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Get a batch from the training dataset"""
        return get_batch_data(self.dataset, indices, start_idx, batch_size)

    def get_expanded_batch(
        self, indices: List[int], start_idx: int, batch_size: int, num_generations: int
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Get an expanded batch for multiple generations"""
        contexts, tasks, tests, entrypoints = self.get_batch(
            indices, start_idx, batch_size
        )
        return expand_for_multiple_generations(
            contexts, tasks, tests, entrypoints, num_generations
        )
