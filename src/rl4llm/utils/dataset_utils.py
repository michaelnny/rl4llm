import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


def shard_dataset(
    dataset: Dataset, num_shards: int, shard_index: int
) -> Dataset:
    """
    Shard a dataset across multiple ranks.

    This function supports both Hugging Face Datasets and PyTorch Datasets.

    Args:
        dataset (Dataset): The dataset to shard.
        num_shards (int): Total number of shards.
        shard_index (int): The current rank (shard index).

    Returns:
        Dataset: The sharded dataset.
    """

    # Manually create a subset.
    total_length = len(dataset)
    shard_size = total_length // num_shards
    start_index = shard_index * shard_size
    # Last shard takes any extra samples
    end_index = (
        total_length
        if shard_index == num_shards - 1
        else start_index + shard_size
    )
    indices = list(range(start_index, end_index))
    return torch.utils.data.Subset(dataset, indices)
