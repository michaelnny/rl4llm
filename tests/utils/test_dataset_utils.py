# import pytest
# import torch
# from torch.utils.data import Subset, Dataset
# from typing import List, Dict, Any

# from rl4llm.utils.dataset_utils import shard_dataset


# @pytest.fixture
# def list_dataset_even() -> List[Dict[str, Any]]:
#     """Dataset with 10 items."""
#     return [{"id": i} for i in range(10)]


# @pytest.fixture
# def list_dataset_odd() -> List[Dict[str, Any]]:
#     """Dataset with 11 items."""
#     return [{"id": i} for i in range(11)]


# @pytest.fixture
# def empty_list_dataset() -> List[Dict[str, Any]]:
#     """Empty dataset."""
#     return []


# class SimpleTorchDataset(Dataset):
#     def __init__(self, data: List[Dict[str, Any]]):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# # --- Updated Test Cases ---

# # 1. Tests for Valid Behavior


# def test_sharding_num_shards_one(list_dataset_even, empty_list_dataset):
#     """Test case where num_shards == 1, should return original dataset."""
#     original_dataset = list_dataset_even
#     num_shards = 1
#     shard_index = 0
#     sharded = shard_dataset(original_dataset, num_shards, shard_index)
#     # Expect the original object back if num_shards == 1
#     assert sharded is original_dataset
#     assert len(sharded) == len(original_dataset)

#     # Test with empty dataset
#     original_empty = empty_list_dataset
#     sharded_empty = shard_dataset(original_empty, num_shards, shard_index)
#     assert sharded_empty is original_empty
#     assert len(sharded_empty) == 0


# def test_empty_dataset_sharding(empty_list_dataset):
#     """Test sharding an empty dataset when num_shards > 1."""
#     original_dataset = empty_list_dataset
#     num_shards = 3
#     shard_index = 1  # Any valid index
#     sharded = shard_dataset(original_dataset, num_shards, shard_index)

#     # Should return an empty Subset
#     assert isinstance(sharded, Subset)
#     assert len(sharded) == 0
#     assert list(sharded) == []  # Convert Subset to list to check content


# @pytest.mark.parametrize(
#     "num_shards, shard_index, expected_ids",
#     [
#         (5, 0, [0, 1]),  # 10 items / 5 shards = 2 items/shard
#         (5, 1, [2, 3]),
#         (5, 2, [4, 5]),
#         (5, 3, [6, 7]),
#         (5, 4, [8, 9]),  # Last shard
#         (2, 0, [0, 1, 2, 3, 4]),  # 10 items / 2 shards = 5 items/shard
#         (2, 1, [5, 6, 7, 8, 9]),  # Last shard
#         (10, 0, [0]),  # 10 items / 10 shards = 1 item/shard
#         (10, 9, [9]),  # Last shard
#     ],
# )
# def test_perfect_division(list_dataset_even, num_shards, shard_index, expected_ids):
#     """Test sharding where dataset size is perfectly divisible by num_shards."""
#     original_dataset = list_dataset_even
#     sharded = shard_dataset(original_dataset, num_shards, shard_index)

#     assert isinstance(sharded, Subset)
#     assert len(sharded) == len(expected_ids)
#     sharded_items = list(sharded)
#     actual_ids = [item["id"] for item in sharded_items]
#     assert actual_ids == expected_ids


# @pytest.mark.parametrize(
#     "num_shards, shard_index, expected_ids",
#     [
#         (3, 0, [0, 1, 2]),  # 11 items / 3 shards = 3 items/shard (base)
#         (3, 1, [3, 4, 5]),
#         (3, 2, [6, 7, 8, 9, 10]),  # Last shard gets remainder (11 - 3*2 = 5 items)
#         (4, 0, [0, 1]),  # 11 items / 4 shards = 2 items/shard (base)
#         (4, 1, [2, 3]),
#         (4, 2, [4, 5]),
#         (4, 3, [6, 7, 8, 9, 10]),  # Last shard gets remainder (11 - 2*3 = 5 items)
#         (11, 0, [0]),  # 11 items / 11 shards = 1 item/shard
#         (11, 10, [10]),  # Last shard
#     ],
# )
# def test_imperfect_division(list_dataset_odd, num_shards, shard_index, expected_ids):
#     """Test sharding where dataset size is not perfectly divisible by num_shards."""
#     original_dataset = list_dataset_odd
#     sharded = shard_dataset(original_dataset, num_shards, shard_index)

#     assert isinstance(sharded, Subset)
#     assert len(sharded) == len(expected_ids)
#     sharded_items = list(sharded)
#     actual_ids = [item["id"] for item in sharded_items]
#     assert actual_ids == expected_ids


# def test_with_torch_dataset(list_dataset_even):
#     """Verify it works with a torch Dataset object as input."""
#     torch_dataset = SimpleTorchDataset(list_dataset_even)
#     num_shards = 4  # 10 items / 4 shards = 2 base, last gets 10 - 2*3 = 4
#     shard_index = 3  # Last shard

#     sharded = shard_dataset(torch_dataset, num_shards, shard_index)
#     assert isinstance(sharded, Subset)
#     assert len(sharded) == 4  # Last shard gets 4 items. Indices 6, 7, 8, 9.

#     sharded_items = list(sharded)
#     actual_ids = [item["id"] for item in sharded_items]
#     assert actual_ids == [6, 7, 8, 9]


# # 2. Tests for Invalid Inputs (Error Handling)


# @pytest.mark.parametrize("invalid_num_shards", [0, -1, -10, 1.5, "abc", None, []])
# def test_invalid_num_shards(list_dataset_even, invalid_num_shards):
#     """Test invalid values for num_shards."""
#     with pytest.raises(ValueError, match="num_shards must be a positive integer"):
#         shard_dataset(list_dataset_even, invalid_num_shards, 0)


# @pytest.mark.parametrize(
#     "num_shards, invalid_shard_index",
#     [
#         (3, -1),  # Negative index
#         (3, 3),  # Index equal to num_shards
#         (3, 4),  # Index greater than num_shards
#         (5, 5),
#         (5, -2),
#         (2, 1.0),  # Float index
#         (2, "0"),  # String index
#         (2, None),  # None index
#     ],
# )
# def test_invalid_shard_index(list_dataset_even, num_shards, invalid_shard_index):
#     """Test invalid values for shard_index when num_shards > 1."""
#     expected_error_msg = (
#         f"shard_index must be an integer in the range \[0, {num_shards - 1}\]"
#     )
#     # Use pytest.raises context manager
#     with pytest.raises(ValueError, match=expected_error_msg):
#         shard_dataset(list_dataset_even, num_shards, invalid_shard_index)
