# --- Fixtures ---

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import torch
from datasets import Dataset
from pydantic import ValidationError
from transformers import PreTrainedModel, PreTrainedTokenizer

# Objects under test
from rl4llm.envs.hf_env import (
    HfMDPEnv,
    BaseMDPEnv,
    BaseRewardFunction,
    ChatMessage,
    EpisodeData,
    EnvState,
)


@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer instance."""
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.padding_side = "left"
    tokenizer.apply_chat_template = MagicMock(return_value="formatted_prompt")
    tokenizer.encode = MagicMock(return_value=[10, 20])  # Mock content encoding
    tokenizer.__call__ = MagicMock(
        return_value={
            "input_ids": torch.tensor([[0, 0, 2, 3, 4]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        }
    )
    tokenizer.batch_decode = MagicMock(return_value=["decoded response"])
    return tokenizer


@pytest.fixture
def mock_reward_function():
    """Provides a mock reward function instance."""
    reward_fn = MagicMock(spec=BaseRewardFunction)
    reward_fn.name = "mock_reward"
    reward_fn.__call__ = MagicMock(
        return_value=[1.0]
    )  # Default return for single item batch
    return reward_fn


@pytest.fixture
def mock_reward_functions(mock_reward_function):
    """Provides a list containing one mock reward function."""
    return [mock_reward_function]


@pytest.fixture
def hf_env_instance(minimal_config):
    """Provides an instance of HfMDPEnv."""
    return HfMDPEnv(**minimal_config)


# Test HfMDPEnv
def test_hf_env_run_interaction_loop(hf_env_instance, mock_llm, mock_tokenizer):
    """Tests the HfMDPEnv interaction loop."""
    # Prepare a simple initial state
    initial_state = EnvState(
        batch_messages=[[ChatMessage(role="user", content="Input prompt")]],
        batch_ground_truth=["Output"],
        batch_init_prompt_size=[1],
    )
    sampling_params = {"max_new_tokens": 5, "temperature": 0.7}

    # Configure mocks for this specific interaction
    prompt_str = "formatted_prompt_for_hf"
    input_tokens = torch.tensor([[0, 10, 11, 12]])  # Mock tokenized prompt
    output_tokens = torch.tensor(
        [[0, 10, 11, 12, 20, 21, 1]]
    )  # Prompt + generated + eos
    decoded_response = "generated text"

    mock_tokenizer.apply_chat_template.return_value = prompt_str
    mock_tokenizer.__call__.return_value = {
        "input_ids": input_tokens.to(mock_llm.device),
        "attention_mask": torch.ones_like(input_tokens).to(mock_llm.device),
    }
    mock_llm.generate.return_value = output_tokens.to(mock_llm.device)
    mock_tokenizer.batch_decode.return_value = [decoded_response]

    final_state = hf_env_instance._run_interaction_loop(
        initial_state, mock_llm, sampling_params
    )

    # Verify tokenizer calls
    mock_tokenizer.apply_chat_template.assert_called_once()
    mock_tokenizer.__call__.assert_called_once_with(
        [prompt_str], padding=True, padding_side="left", return_tensors="pt"
    )

    # Verify LLM call
    mock_llm.generate.assert_called_once()
    call_args, call_kwargs = mock_llm.generate.call_args
    assert torch.equal(call_kwargs["input_ids"], input_tokens.to(mock_llm.device))
    assert call_kwargs["max_new_tokens"] == 5
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["pad_token_id"] == mock_tokenizer.pad_token_id
    assert call_kwargs["eos_token_id"] == mock_tokenizer.eos_token_id

    # Verify decoding call
    expected_generated_ids = output_tokens[:, input_tokens.shape[1] :]  # [[20, 21, 1]]
    mock_tokenizer.batch_decode.assert_called_once_with(
        expected_generated_ids, skip_special_tokens=True
    )

    # Verify final state
    assert len(final_state.batch_messages) == 1
    assert len(final_state.batch_messages[0]) == 2  # Initial user + new assistant
    assert final_state.batch_messages[0][0].role == "user"
    assert final_state.batch_messages[0][1].role == "assistant"
    assert final_state.batch_messages[0][1].content == decoded_response
    assert final_state.batch_ground_truth == initial_state.batch_ground_truth
    assert final_state.batch_init_prompt_size == initial_state.batch_init_prompt_size
