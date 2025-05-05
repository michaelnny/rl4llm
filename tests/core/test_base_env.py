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
from rl4llm.core.base_env import (
    BaseMDPEnv,
    BaseRewardFunction,
    ChatMessage,
    EpisodeData,
    EnvState,
    find_subsequence,
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


# --- Fixtures ---


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
def sample_dataset_dict():
    """Provides a sample dataset dictionary."""
    return {
        "messages": [
            [{"role": "user", "content": "Hello"}],
            [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
        ],
        "ground_truth": ["World", "There"],
    }


@pytest.fixture
def sample_dataset(sample_dataset_dict):
    """Provides a sample Hugging Face Dataset."""
    return Dataset.from_dict(sample_dataset_dict)


@pytest.fixture
def minimal_config(sample_dataset, mock_tokenizer, mock_reward_functions):
    """Provides minimal configuration for BaseMDPEnv."""
    return {
        "dataset": sample_dataset,
        "tokenizer": mock_tokenizer,
        "reward_functions": mock_reward_functions,
        "batch_size": 1,
        "group_size": 1,
        "max_steps": 1,
        "rank": 0,
        "world_size": 1,
    }


# --- Concrete Subclass for Testing BaseMDPEnv ---


class ConcreteTestEnv(BaseMDPEnv):
    """A minimal concrete subclass for testing BaseMDPEnv."""

    def _run_interaction_loop(
        self,
        initial_state: EnvState,
        llm: Any,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        # Minimal implementation: just add a dummy assistant message
        final_messages = []
        for msg_list in initial_state.batch_messages:
            final_messages.append(
                msg_list + [ChatMessage(role="assistant", content="dummy response")]
            )

        return EnvState(
            batch_messages=final_messages,
            batch_ground_truth=initial_state.batch_ground_truth,
            batch_init_prompt_size=initial_state.batch_init_prompt_size,
        )


@pytest.fixture
def base_env_instance(minimal_config):
    """Provides an instance of the concrete test environment."""
    return ConcreteTestEnv(**minimal_config)


@pytest.fixture
def mock_llm():
    """Provides a mock Hugging Face PreTrainedModel."""
    llm = MagicMock(spec=PreTrainedModel)
    llm.device = torch.device("cpu")
    # Mock generate to return input_ids + some generated tokens
    llm.generate = MagicMock(
        return_value=torch.tensor([[0, 0, 2, 3, 4, 5, 6, 1]])
    )  # Includes original + generated + eos
    return llm


# --- Test Cases ---


# Test Pydantic Models (ChatMessage, EpisodeData, EnvState)
def test_chat_message_valid():
    """Tests valid ChatMessage creation."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


@pytest.mark.parametrize("role", ["invalid_role", "", None])
def test_chat_message_invalid_role(role):
    """Tests ChatMessage validation for invalid roles."""
    with pytest.raises(ValidationError):
        ChatMessage(role=role, content="Test")


def test_episode_data_valid():
    """Tests valid EpisodeData creation."""
    data = EpisodeData(
        states=torch.tensor([1, 2]),
        actions=torch.tensor([2, 3]),
        loss_mask=torch.tensor([True, False]),
        terminal_reward=0.5,
        ground_truth="answer",
        reward_dict={"r1": 0.5},
        chat_history=[
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ],
        prompt_length=1,
        completion_length=1,
    )
    assert data.prompt_length == 1


def test_episode_data_shape_mismatch():
    """Tests EpisodeData validation for tensor shape mismatch."""
    with pytest.raises(ValidationError):
        EpisodeData(
            states=torch.tensor([1, 2]),
            actions=torch.tensor([2, 3, 4]),  # Mismatched shape
            loss_mask=torch.tensor([True, False]),
            terminal_reward=0.5,
            ground_truth="answer",
            reward_dict={"r1": 0.5},
            chat_history=[],
            prompt_length=1,
            completion_length=1,
        )


def test_env_state_valid():
    """Tests valid EnvState creation."""
    state = EnvState(
        batch_messages=[[ChatMessage(role="user", content="Hi")]],
        batch_ground_truth=["Bye"],
        batch_init_prompt_size=[1],
    )
    assert len(state.batch_messages) == 1


# Test BaseRewardFunction
class ConcreteReward(BaseRewardFunction):
    """Concrete reward for testing BaseRewardFunction."""

    def __call__(self, batch_messages, batch_ground_truths, **kwargs):
        return [1.0] * len(batch_messages)


@pytest.mark.parametrize(
    "name, should_raise",
    [
        ("valid-name_1", False),
        ("", True),
        ("invalid name", True),
        (None, True),
        (123, True),
    ],
)
def test_base_reward_function_init_validation(name, should_raise):
    """Tests BaseRewardFunction name validation during initialization."""
    if should_raise:
        with pytest.raises((ValueError, TypeError)):
            ConcreteReward(name=name)
    else:
        reward = ConcreteReward(name=name)
        assert reward.name == name


# Test BaseMDPEnv Initialization
def test_init_success(minimal_config):
    """Tests successful initialization of BaseMDPEnv."""
    env = ConcreteTestEnv(**minimal_config)
    assert env.batch_size == minimal_config["batch_size"]
    assert env.group_size == minimal_config["group_size"]
    assert env.tokenizer == minimal_config["tokenizer"]
    assert len(env.sharded_dataset) == len(minimal_config["dataset"])  # world_size=1


@pytest.mark.parametrize(
    "param, value, error",
    [
        ("batch_size", 0, ValueError),
        ("group_size", 0, ValueError),
        ("max_steps", 0, ValueError),
        ("reward_functions", [], ValueError),
        ("reward_functions", [lambda x: x], ValueError),  # Not BaseRewardFunction
        ("dataset", [], TypeError),  # Not a Dataset
    ],
)
def test_init_invalid_params(minimal_config, param, value, error):
    """Tests BaseMDPEnv initialization with invalid parameters."""
    config = minimal_config.copy()
    config[param] = value
    with pytest.raises(error):
        ConcreteTestEnv(**config)


def test_init_missing_dataset_columns(minimal_config):
    """Tests BaseMDPEnv init fails if dataset misses required columns."""
    config = minimal_config.copy()
    config["dataset"] = Dataset.from_dict({"wrong_col": [1]})
    with pytest.raises(ValueError, match="needs 'messages' and 'ground_truth'"):
        ConcreteTestEnv(**config)


def test_init_invalid_messages_format(minimal_config):
    """Tests BaseMDPEnv init fails if 'messages' column has wrong format."""
    config = minimal_config.copy()
    config["dataset"] = Dataset.from_dict(
        {"messages": ["not a list of dicts"], "ground_truth": ["ok"]}
    )
    with pytest.raises(ValueError, match="'messages' column should contain lists"):
        ConcreteTestEnv(**config)


def test_init_multiple_rewards_no_transform(minimal_config, mock_reward_function):
    """Tests BaseMDPEnv init fails with multiple rewards but no transform function."""
    config = minimal_config.copy()
    config["reward_functions"] = [mock_reward_function, ConcreteReward("reward2")]
    with pytest.raises(ValueError, match="Multiple reward functions provided without"):
        ConcreteTestEnv(**config)


def test_init_tokenizer_no_pad_token(minimal_config):
    """Tests tokenizer setup handles missing pad token by using eos token."""
    config = minimal_config.copy()
    mock_tokenizer_no_pad = MagicMock(spec=PreTrainedTokenizer)
    mock_tokenizer_no_pad.pad_token = None
    mock_tokenizer_no_pad.eos_token = "<eos>"
    mock_tokenizer_no_pad.pad_token_id = None
    mock_tokenizer_no_pad.eos_token_id = 1
    mock_tokenizer_no_pad.padding_side = "left"
    mock_tokenizer_no_pad.apply_chat_template = MagicMock(return_value="test")
    config["tokenizer"] = mock_tokenizer_no_pad

    env = ConcreteTestEnv(**config)
    assert env.tokenizer.pad_token == env.tokenizer.eos_token
    assert env.tokenizer.pad_token_id == env.tokenizer.eos_token_id
    assert env.pad_token_id == env.tokenizer.eos_token_id


def test_init_tokenizer_no_chat_template(minimal_config):
    """Tests BaseMDPEnv init fails if tokenizer lacks a chat template."""
    config = minimal_config.copy()
    mock_tokenizer_no_template = MagicMock(spec=PreTrainedTokenizer)
    mock_tokenizer_no_template.pad_token = "<pad>"
    mock_tokenizer_no_template.eos_token = "<eos>"
    mock_tokenizer_no_template.pad_token_id = 0
    mock_tokenizer_no_template.eos_token_id = 1
    mock_tokenizer_no_template.padding_side = "left"
    mock_tokenizer_no_template.apply_chat_template = MagicMock(
        side_effect=Exception("No template")
    )
    config["tokenizer"] = mock_tokenizer_no_template

    with pytest.raises(ValueError, match="Tokenizer must have a chat template"):
        ConcreteTestEnv(**config)


# Test BaseMDPEnv Core Logic
def test_collate_fn(base_env_instance):
    """Tests the simple collation function."""
    batch_list = [{"a": 1, "b": [2]}, {"a": 3, "b": [4]}]
    collated = base_env_instance._collate_fn(batch_list)
    assert collated == {"a": [1, 3], "b": [[2], [4]]}
    assert base_env_instance._collate_fn([]) == {}


def test_prepare_initial_state(base_env_instance, sample_dataset_dict):
    """Tests the preparation of the initial environment state."""
    base_env_instance.group_size = 2
    raw_batch = sample_dataset_dict  # Use the dict directly
    initial_state = base_env_instance._prepare_initial_state(raw_batch)

    assert len(initial_state.batch_messages) == len(raw_batch["messages"]) * 2
    assert len(initial_state.batch_ground_truth) == len(raw_batch["ground_truth"]) * 2
    assert len(initial_state.batch_init_prompt_size) == len(raw_batch["messages"]) * 2

    # Check content replication and init prompt size (message count)
    assert initial_state.batch_messages[0][0].content == "Hello"
    assert initial_state.batch_messages[1][0].content == "Hello"
    assert initial_state.batch_init_prompt_size[0] == 1  # First sample has 1 message
    assert initial_state.batch_init_prompt_size[1] == 1

    assert initial_state.batch_messages[2][0].content == "Be helpful"
    assert initial_state.batch_messages[3][0].content == "Be helpful"
    assert initial_state.batch_init_prompt_size[2] == 2  # Second sample has 2 messages
    assert initial_state.batch_init_prompt_size[3] == 2

    assert isinstance(initial_state.batch_messages[0][0], ChatMessage)


def test_prepare_initial_state_invalid_message_format(base_env_instance):
    """Tests error handling for invalid message format during state preparation."""
    raw_batch = {
        "messages": [[{"role": "user", "content": "ok"}], [{"invalid": "format"}]],
        "ground_truth": ["gt1", "gt2"],
    }
    with pytest.raises(ValueError, match="Invalid message format"):
        base_env_instance._prepare_initial_state(raw_batch)


def test_calculate_rewards(base_env_instance):  # Removed mock_reward_function from args
    """Tests reward calculation with a single reward function."""
    # Get the actual mock function used by the instance
    reward_fn_in_env = base_env_instance.reward_functions[0]
    # Configure its return value for this specific test
    reward_fn_in_env.return_value = [0.8]
    # Reset mock state before the call
    reward_fn_in_env.reset_mock()

    batch_messages = [[ChatMessage(role="user", content="Q")]]
    batch_ground_truths = ["A"]

    terminal_rewards, rewards_dict = base_env_instance._calculate_rewards(
        batch_messages, batch_ground_truths
    )

    # Assert call on the mock instance *retrieved from the environment*
    # Use explicit comparison if assert_called_once_with fails mysteriously
    reward_fn_in_env.assert_called_once()  # Check it was called
    actual_call = reward_fn_in_env.call_args
    expected_call = call(batch_messages, batch_ground_truths)  # Use unittest.mock.call

    # Compare arguments explicitly for clarity
    assert actual_call == expected_call, (
        f"Expected call {expected_call} but got {actual_call}"
    )

    # Final checks
    assert torch.equal(terminal_rewards, torch.tensor([0.8]))
    assert rewards_dict == {reward_fn_in_env.name: [0.8]}  # Use the actual name


def test_calculate_rewards_multiple(minimal_config):
    """Tests reward calculation with multiple functions and a transform."""
    # Create fresh mocks specifically for this test to avoid fixture side effects
    mock_reward_fn_1 = MagicMock(spec=BaseRewardFunction)
    mock_reward_fn_1.name = "reward1"
    mock_reward_fn_1.return_value = [0.8]

    mock_reward_fn_2 = MagicMock(spec=BaseRewardFunction)
    mock_reward_fn_2.name = "reward2"
    mock_reward_fn_2.return_value = [0.5]

    transform_fn = MagicMock(return_value=torch.tensor([0.65]))

    config = minimal_config.copy()
    # Use the fresh mocks created locally for this test
    config["reward_functions"] = [mock_reward_fn_1, mock_reward_fn_2]
    config["reward_transform_fn"] = transform_fn
    # Create the env instance with these specific mocks
    env = ConcreteTestEnv(**config)

    batch_messages = [[ChatMessage(role="user", content="Q")]]
    batch_ground_truths = ["A"]

    terminal_rewards, rewards_dict = env._calculate_rewards(
        batch_messages, batch_ground_truths
    )

    # --- Assertions for mock_reward_fn_1 ---
    mock_reward_fn_1.assert_called_once()
    actual_call_1 = mock_reward_fn_1.call_args
    expected_call = call(batch_messages, batch_ground_truths)
    assert actual_call_1 == expected_call, (
        f"Fn1: Expected call {expected_call} but got {actual_call_1}"
    )

    # --- Assertions for mock_reward_fn_2 ---
    mock_reward_fn_2.assert_called_once()
    actual_call_2 = mock_reward_fn_2.call_args
    assert actual_call_2 == expected_call, (
        f"Fn2: Expected call {expected_call} but got {actual_call_2}"
    )

    # --- Assertions for transform_fn ---
    transform_fn.assert_called_once_with({"reward1": [0.8], "reward2": [0.5]})

    # --- Final assertions ---
    assert torch.equal(terminal_rewards, torch.tensor([0.65]))
    assert rewards_dict == {"reward1": [0.8], "reward2": [0.5]}


def test_calculate_rewards_function_error(base_env_instance, mock_reward_function):
    """Tests reward calculation handles errors in a reward function."""
    batch_messages = [[ChatMessage(role="user", content="Q")]]
    batch_ground_truths = ["A"]
    mock_reward_function.__call__.side_effect = ValueError("Reward calculation failed")

    terminal_rewards, rewards_dict = base_env_instance._calculate_rewards(
        batch_messages, batch_ground_truths
    )

    # Should return default reward (0.0) and log error (check logs if needed)
    assert torch.equal(terminal_rewards, torch.tensor([0.0]))
    assert rewards_dict == {"mock_reward": [0.0]}


def test_transform_rewards_error(minimal_config, mock_reward_function):
    """Tests reward transformation fallback when transform function fails."""
    transform_fn = MagicMock(side_effect=ValueError("Transform failed"))
    reward_fn_2 = ConcreteReward("reward2")
    reward_fn_2.__call__ = MagicMock(return_value=[0.5])

    config = minimal_config.copy()
    config["reward_functions"] = [mock_reward_function, reward_fn_2]
    config["reward_transform_fn"] = transform_fn
    env = ConcreteTestEnv(**config)

    rewards_dict = {"mock_reward": [0.8], "reward2": [0.5]}
    terminal_rewards = env._transform_rewards(rewards_dict)

    # Should fall back to the first reward function's output
    assert torch.equal(terminal_rewards, torch.tensor([0.8], dtype=torch.float32))


def test_convert_batch_message_to_prompt(base_env_instance, mock_tokenizer):
    """Tests conversion of messages to a chat-formatted prompt string."""
    # Reset mock calls before the test execution
    mock_tokenizer.reset_mock()
    mock_tokenizer.apply_chat_template.side_effect = [
        "prompt1",
        "prompt2",
    ]  # Different return for each call

    batch_messages = [
        [ChatMessage(role="user", content="Hello")],
        [
            ChatMessage(role="system", content="Sys"),
            ChatMessage(role="user", content="Hi"),
        ],
    ]

    prompts = base_env_instance._convert_batch_message_to_prompt(batch_messages)

    # Now expect exactly 2 calls from within the tested function
    assert mock_tokenizer.apply_chat_template.call_count == 2
    # Check the first call
    call_args_1, call_kwargs_1 = mock_tokenizer.apply_chat_template.call_args_list[0]
    assert call_args_1[0] == [
        {"role": "user", "content": "Hello", "tool_calls": None, "tool_call_id": None}
    ]
    assert call_kwargs_1["tokenize"] is False
    assert call_kwargs_1["add_generation_prompt"] is True
    # Check the second call
    call_args_2, call_kwargs_2 = mock_tokenizer.apply_chat_template.call_args_list[1]
    assert call_args_2[0] == [
        {"role": "system", "content": "Sys", "tool_calls": None, "tool_call_id": None},
        {"role": "user", "content": "Hi", "tool_calls": None, "tool_call_id": None},
    ]
    assert call_kwargs_2["tokenize"] is False
    assert call_kwargs_2["add_generation_prompt"] is True

    assert prompts == ["prompt1", "prompt2"]


def test_find_subsequence():
    """Tests the helper function find_subsequence."""
    assert find_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2
    assert find_subsequence([1, 2, 3, 4, 5], [1]) == 0
    assert find_subsequence([1, 2, 3, 4, 5], [5]) == 4
    assert find_subsequence([1, 2, 3, 4, 5], [6]) == -1
    assert find_subsequence([1, 2, 3], [1, 2, 3, 4]) == -1
    assert find_subsequence([], [1]) == -1
    assert find_subsequence([1], []) == -1
    assert find_subsequence([1, 1, 2, 1, 2, 3], [1, 2, 3]) == 3


# Test _convert_to_episodes (Complex Case)
@pytest.fixture
def final_state_for_conversion(mock_reward_function):
    """Provides a final EnvState suitable for testing _convert_to_episodes."""
    # Mock reward function to return specific values for this state
    mock_reward_function.__call__ = MagicMock(return_value=[0.7])

    return EnvState(
        batch_messages=[
            [  # Sample 1
                ChatMessage(role="user", content="Question?"),  # Prompt
                ChatMessage(role="assistant", content="Answer."),  # Completion
            ]
        ],
        batch_ground_truth=["Correct Answer"],
        batch_init_prompt_size=[1],  # Prompt is the first message
    )


@pytest.fixture
def mock_tokenizer_for_conversion(mock_tokenizer):
    """Configures mock tokenizer specifically for _convert_to_episodes tests."""

    # Mock apply_chat_template to return different token lists based on input length
    def template_side_effect(messages, tokenize, add_generation_prompt):
        if len(messages) == 1:  # Just the prompt
            return [101, 10, 11, 102]  # e.g., [CLS] Q tokens [SEP]
        elif len(messages) == 2:  # Prompt + Answer
            return [
                101,
                10,
                11,
                102,
                20,
                21,
                22,
                103,
            ]  # e.g., [CLS] Q [SEP] A tokens [EOS]
        else:
            return []  # Should not happen in this test case

    mock_tokenizer.apply_chat_template = MagicMock(side_effect=template_side_effect)
    # Mock encode for the content "Answer."
    mock_tokenizer.encode = MagicMock(
        return_value=[20, 21, 22]
    )  # Tokens for "Answer." content

    return mock_tokenizer


def test_convert_to_episodes_basic(
    base_env_instance,
    final_state_for_conversion,
    mock_tokenizer_for_conversion,
    mock_reward_function,  # Add mock_reward_function fixture
):
    """Tests the basic conversion of a final state to EpisodeData."""
    # Configure the mock reward function on the instance FOR THIS TEST
    # Ensure the instance uses the correct mock object
    base_env_instance.reward_functions = [mock_reward_function]
    mock_reward_function.reset_mock()  # Reset any previous calls/configs
    mock_reward_function.return_value = [0.7]  # Set the desired return value

    base_env_instance.tokenizer = mock_tokenizer_for_conversion
    episodes = base_env_instance._convert_to_episodes(final_state_for_conversion)

    assert len(episodes) == 1
    ep = episodes[0]

    assert isinstance(ep, EpisodeData)
    assert ep.ground_truth == "Correct Answer"
    # Check the reward value set specifically for this test
    assert round(ep.terminal_reward, 1) == 0.7
    assert ep.reward_dict == {"mock_reward": 0.7}
    assert len(ep.chat_history) == 2
    assert ep.chat_history[0].role == "user"
    assert ep.chat_history[1].role == "assistant"

    # --- Assertions for tokenization and masking ---
    assert ep.prompt_length == 4
    assert torch.equal(ep.states, torch.tensor([101, 10, 11, 102, 20, 21, 22]))
    assert torch.equal(ep.actions, torch.tensor([10, 11, 102, 20, 21, 22, 103]))
    expected_mask = torch.tensor([False, False, False, True, True, True, False])
    assert torch.equal(ep.loss_mask, expected_mask)
    assert ep.completion_length == 3  # Sum of loss mask


def test_convert_to_episodes_no_completion(
    base_env_instance, mock_tokenizer_for_conversion
):
    """Tests conversion when there's only a prompt (no assistant message)."""
    final_state = EnvState(
        batch_messages=[[ChatMessage(role="user", content="Question?")]],  # Only prompt
        batch_ground_truth=["Correct Answer"],
        batch_init_prompt_size=[1],
    )
    base_env_instance.tokenizer = mock_tokenizer_for_conversion
    episodes = base_env_instance._convert_to_episodes(final_state)

    assert len(episodes) == 1
    ep = episodes[0]

    # Expected sequence: [101, 10, 11, 102]
    # Prompt length: 4
    # States: [101, 10, 11]
    # Actions: [10, 11, 102]
    # Loss Mask: [F, F, F]
    assert ep.prompt_length == 4
    assert torch.equal(ep.states, torch.tensor([101, 10, 11]))
    assert torch.equal(ep.actions, torch.tensor([10, 11, 102]))
    assert torch.equal(ep.loss_mask, torch.tensor([False, False, False]))
    assert ep.completion_length == 0


def test_convert_to_episodes_short_sequence(
    base_env_instance, mock_tokenizer_for_conversion
):
    """Tests conversion handles sequences too short to form state/action pairs."""
    # Mock tokenizer to return a very short sequence
    mock_tokenizer_for_conversion.apply_chat_template = MagicMock(
        return_value=[101]
    )  # Length 1

    final_state = EnvState(
        batch_messages=[[ChatMessage(role="user", content="Q")]],
        batch_ground_truth=["A"],
        batch_init_prompt_size=[1],
    )
    base_env_instance.tokenizer = mock_tokenizer_for_conversion
    episodes = base_env_instance._convert_to_episodes(final_state)

    assert len(episodes) == 0  # Should skip the sample


# Test BaseMDPEnv Reset and Rollout
def test_reset_successful(base_env_instance, sample_dataset_dict):
    """Tests the reset mechanism to get a new batch."""
    # Mock the dataloader iterator
    mock_iterator = MagicMock()
    mock_iterator.__next__.return_value = sample_dataset_dict
    base_env_instance.dataset_iterator = mock_iterator

    initial_state = base_env_instance._reset()

    assert initial_state is not None
    assert (
        len(initial_state.batch_messages)
        == len(sample_dataset_dict["messages"]) * base_env_instance.group_size
    )
    mock_iterator.__next__.assert_called_once()


@pytest.mark.parametrize(
    "exception,log_message",
    [
        (RuntimeError("Batch error"), "Error getting batch"),
        (ValueError("State error"), "Error preparing initial state from batch"),
    ],
)
def test_reset_handles_exceptions(base_env_instance, mocker, exception, log_message):
    """Tests exception handling in reset method."""
    if log_message == "Error getting batch":
        mocker.patch.object(base_env_instance, "_get_next_batch", side_effect=exception)
    else:
        mocker.patch.object(
            base_env_instance, "_prepare_initial_state", side_effect=exception
        )
    with pytest.raises(type(exception)) as exc_info:
        base_env_instance._reset()
    assert str(exc_info.value) == str(exception)


def test_rollout_basic_flow(
    base_env_instance, mock_llm, mock_tokenizer_for_conversion, sample_dataset_dict
):
    """Tests the basic orchestration flow of the rollout method."""
    # Mock reset to return a valid initial state
    initial_state = base_env_instance._prepare_initial_state(sample_dataset_dict)
    base_env_instance._reset = MagicMock(return_value=initial_state)

    # Mock interaction loop (already implemented in ConcreteTestEnv)
    # Mock conversion (use the specific tokenizer)
    base_env_instance.tokenizer = mock_tokenizer_for_conversion

    episodes = base_env_instance.rollout(
        llm=mock_llm, sampling_params={"max_new_tokens": 10}
    )

    base_env_instance._reset.assert_called_once()
    # _run_interaction_loop is called implicitly by rollout via the concrete class
    # _convert_to_episodes is called implicitly by rollout
    assert len(episodes) > 0  # Should produce episodes based on sample_dataset_dict


def test_rollout_reset_returns_none(base_env_instance, mock_llm):
    """Tests rollout returns empty list if reset fails."""
    base_env_instance._reset = MagicMock(return_value=None)
    episodes = base_env_instance.rollout(llm=mock_llm, sampling_params={})
    assert episodes == []
    base_env_instance._reset.assert_called_once()


@patch.object(
    ConcreteTestEnv,
    "_run_interaction_loop",
    side_effect=Exception("Interaction failed"),
)
def test_rollout_interaction_error(
    mock_interaction, base_env_instance, mock_llm, sample_dataset_dict
):
    """Tests rollout returns empty list if interaction loop fails."""
    initial_state = base_env_instance._prepare_initial_state(sample_dataset_dict)
    base_env_instance._reset = MagicMock(return_value=initial_state)

    episodes = base_env_instance.rollout(llm=mock_llm, sampling_params={})

    assert episodes == []
    base_env_instance._reset.assert_called_once()
    mock_interaction.assert_called_once()


@patch.object(
    ConcreteTestEnv, "_convert_to_episodes", side_effect=Exception("Conversion failed")
)
def test_rollout_conversion_error(
    mock_conversion, base_env_instance, mock_llm, sample_dataset_dict
):
    """Tests rollout returns empty list if conversion fails."""
    initial_state = base_env_instance._prepare_initial_state(sample_dataset_dict)
    base_env_instance._reset = MagicMock(return_value=initial_state)
    # Need to let interaction run to get to conversion
    base_env_instance._run_interaction_loop = MagicMock(
        wraps=base_env_instance._run_interaction_loop
    )

    episodes = base_env_instance.rollout(llm=mock_llm, sampling_params={})

    assert episodes == []
    base_env_instance._reset.assert_called_once()
    base_env_instance._run_interaction_loop.assert_called_once()
    mock_conversion.assert_called_once()

