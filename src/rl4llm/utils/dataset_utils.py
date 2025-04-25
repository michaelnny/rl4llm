import logging
from typing import Any, Dict, List, Sequence, TypeVar, Union

import torch
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


# Define a TypeVar to represent the input dataset type, making the return type more precise
T = TypeVar('T', bound=Union[Dataset, Sequence[Any]])


def shard_dataset(
    dataset: T, num_shards: int, shard_index: int
) -> Union[T, Subset]:
    """
    Shard a dataset (e.g., PyTorch Dataset, list) across multiple ranks.

    Distributes the items of the dataset into num_shards partitions and returns
    the partition corresponding to shard_index. The last shard receives any
    extra items if the dataset size is not perfectly divisible by num_shards.

    Args:
        dataset (Union[Dataset, Sequence]): The dataset to shard. Must support
            len() and indexing (__getitem__). Examples: torch.utils.data.Dataset, list.
        num_shards (int): Total number of shards (must be a positive integer).
        shard_index (int): The current rank/shard index (must be >= 0 and < num_shards).

    Returns:
        Union[T, Subset]: The original dataset if num_shards is 1,
                          otherwise a torch.utils.data.Subset representing the
                          assigned shard.

    Raises:
        TypeError: If dataset doesn't support len() or __getitem__ implicitly.
        ValueError: If num_shards is not positive, or if shard_index is out of
                    the valid range [0, num_shards-1].
    """
    # --- Input Validation ---
    if not isinstance(num_shards, int) or num_shards <= 0:
        # Allow num_shards == 1, as it means no sharding, handle below
        if num_shards == 1:
            if not isinstance(shard_index, int) or shard_index != 0:
                raise ValueError(
                    f"If num_shards is 1, shard_index must be 0, got {shard_index}"
                )
            # Fall through to return the original dataset
        else:
            raise ValueError(
                f"num_shards must be a positive integer, got {num_shards}"
            )

    # Only validate shard_index if actual sharding occurs (num_shards > 1)
    # If num_shards == 1, the check above already ensures shard_index is 0.
    if num_shards > 1:
        if not isinstance(shard_index, int) or not (
            0 <= shard_index < num_shards
        ):
            raise ValueError(
                f"shard_index must be an integer in the range [0, {num_shards - 1}], got {shard_index}"
            )

    # --- Handle No Sharding Case ---
    if num_shards == 1:
        return dataset

    # --- Sharding Logic ---
    try:
        total_length = len(dataset)
    except TypeError:
        raise TypeError('Input dataset must support len()')

    if total_length == 0:
        return Subset(dataset, [])

    # Calculate shard size and indices
    shard_size = total_length // num_shards
    start_index = shard_index * shard_size
    # Last shard takes any extra samples
    end_index = (
        total_length
        if shard_index == num_shards - 1
        else start_index + shard_size
    )

    indices = list(range(start_index, end_index))

    return Subset(dataset, indices)
