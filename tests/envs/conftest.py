import random
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from rl4llm.core.base_env import (
    BaseMDPEnv,
    BaseRewardFunction,
    ChatMessage,
    EnvState,
    EpisodeData,
    SampleState,
)

# --- Fixtures ---


@pytest.fixture
def mock_tokenizer():
    """Provides a mock PreTrainedTokenizer."""
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '<eos>'
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    # Store the word map globally within the mock instance for consistency
    tokenizer.word_map = {}
    tokenizer.next_word_id = 100

    def _get_word_id(word):
        if word not in tokenizer.word_map:
            tokenizer.word_map[word] = tokenizer.next_word_id
            tokenizer.next_word_id += 1
        return tokenizer.word_map[word]

    def _mock_apply_chat_template_v2(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        tools=None,
    ):
        tokens = []
        full_text = ''
        role_map = {'user': 10, 'assistant': 11, 'system': 12, 'tool': 13}
        # Reset map for each call to simulate fresh tokenization *if needed*
        # For these tests, let's keep the map persistent across calls within a single test setup
        # unless explicitly reset by the test.

        for i, msg in enumerate(messages):
            # Use .get() for safer dictionary access if msg is a dict
            role = msg.get('role') if isinstance(msg, dict) else msg.role
            content = (
                msg.get('content') if isinstance(msg, dict) else msg.content
            )

            role_token = role_map.get(role, 99)
            tokens.append(role_token)
            full_text += f"{role}: {content}\n"
            content_tokens = []
            if content:  # Handle potential None content
                for word in content.split():
                    content_tokens.append(_get_word_id(word))
            tokens.extend(content_tokens)
            tokens.append(tokenizer.eos_token_id)  # Add EOS after each message

        if add_generation_prompt:
            tokens.append(role_map['assistant'])
            full_text += 'assistant:\n'

        if tokenize:
            return tokens
        else:
            return full_text

    def _mock_encode_v2(text, add_special_tokens=False):
        tokens = []
        if text:  # Handle potential None text
            for word in text.split():
                tokens.append(_get_word_id(word))
        return tokens

    tokenizer.apply_chat_template = MagicMock(
        side_effect=_mock_apply_chat_template_v2
    )
    tokenizer.encode = MagicMock(side_effect=_mock_encode_v2)

    # Needed for _setup_tokenizer check
    tokenizer.chat_template = 'mock_template'  # Indicate a template exists

    return tokenizer


@pytest.fixture
def mock_reward_function():
    """Provides a simple mock BaseRewardFunction."""

    class MockReward(BaseRewardFunction):
        def __init__(self, name='mock_reward', reward_value=1.0):
            super().__init__(name)
            self.reward_value = reward_value

        def __call__(
            self,
            batch_messages: List[List[ChatMessage]],
            batch_ground_truths: List[Union[str, float, int]],
            **kwargs: Any,
        ) -> List[float]:
            return [self.reward_value] * len(batch_messages)

    return MockReward()


@pytest.fixture
def mock_reward_function_alt():
    """Provides a second mock BaseRewardFunction with a different value."""

    class MockRewardAlt(BaseRewardFunction):
        def __init__(self, name='mock_reward_alt', reward_value=0.5):
            super().__init__(name)
            self.reward_value = reward_value

        def __call__(
            self,
            batch_messages: List[List[ChatMessage]],
            batch_ground_truths: List[Union[str, float, int]],
            **kwargs: Any,
        ) -> List[float]:
            return [self.reward_value] * len(batch_messages)

    return MockRewardAlt()


@pytest.fixture
def sample_raw_data():
    """Provides sample raw data mimicking dataset rows."""
    return [
        {
            'messages': [{'role': 'user', 'content': 'Hello there'}],
            'ground_truth': 'General Kenobi',
        },
        {
            'messages': [
                {'role': 'user', 'content': 'Explain RLHF'},
                {'role': 'system', 'content': 'Be concise'},
            ],
            'ground_truth': 'Reinforcement Learning from Human Feedback',
        },
    ]


@pytest.fixture
def mock_dataset(sample_raw_data):
    """Provides a mock datasets.Dataset."""
    return Dataset.from_list(sample_raw_data)
