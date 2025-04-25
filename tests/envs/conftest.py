import random
from typing import Any, Dict, List
from unittest.mock import Mock

import datasets
import pytest
import torch

from rl4llm.core.base_env import (
    BaseRewardFunction,
    EnvState,
)


class DummyTokenizer:
    """A dummy tokenizer for testing."""

    def __init__(
        self,
        pad_token='<pad>',
        eos_token='<eos>',
        pad_token_id=0,
        eos_token_id=1,
    ):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = 'right'

    def __call__(
        self, texts, truncation=False, max_length=None, padding=False, **kwargs
    ):
        if isinstance(texts, list):
            input_ids = []
            attention_mask = []
            for text in texts:
                # Create dummy tokens based on each word's length
                tokens = [len(word) + 1 for word in text.split()]
                if truncation and max_length is not None:
                    tokens = tokens[:max_length]
                input_ids.append(tokens)
                attention_mask.append([1] * len(tokens))
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            tokens = [len(word) + 1 for word in texts.split()]
            if truncation and max_length is not None:
                tokens = tokens[:max_length]
            return {'input_ids': tokens, 'attention_mask': [1] * len(tokens)}

    def pad(
        self,
        encoded_inputs,
        padding,
        padding_side,
        return_tensors,
        return_attention_mask,
        **kwargs,
    ):
        sequences = encoded_inputs['input_ids']
        max_len = max(len(seq) for seq in sequences)
        padded = []
        attention_masks = []
        for seq in sequences:
            pad_length = max_len - len(seq)
            if padding_side == 'left':
                padded_seq = [self.pad_token_id] * pad_length + seq
                mask = [0] * pad_length + [1] * len(seq)
            else:
                padded_seq = seq + [self.pad_token_id] * pad_length
                mask = [1] * len(seq) + [0] * pad_length
            padded.append(padded_seq)
            attention_masks.append(mask)
        return {
            'input_ids': torch.tensor(padded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }

    def batch_decode(
        self,
        sequences,
        skip_special_tokens,
        clean_up_tokenization_spaces,
        **kwargs,
    ):

        texts = []
        for seq in sequences:
            words = [
                str(token)
                for token in seq
                if not (
                    skip_special_tokens
                    and token in [self.pad_token_id, self.eos_token_id]
                )
            ]
            texts.append(' '.join(words))
        return texts


class DummyRewardFunction(BaseRewardFunction):
    """A dummy reward function that always returns 1.0 for testing."""

    def __init__(self, name='mock_reward_function'):
        super().__init__(name)

    def __call__(self, completions, ground_truths):
        return [1.0 for _ in completions]


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return DummyTokenizer()


@pytest.fixture
def mock_reward_function():
    """Create a mock reward function."""
    return DummyRewardFunction()
