import random
from typing import Any, Dict, List
from unittest.mock import Mock

import datasets
import pytest
import torch

from rl4llm.envs.explore_env import (
    BaseRewardFunction,
    EnvState,
    ExploreInferenceEnv,
)


# Mock classes and dependencies
class EnvState:
    def __init__(self, prompt: List[str], ground_truth: List[str]):
        self.prompt = prompt
        self.ground_truth = ground_truth


class InferenceClient:
    def generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        custom_logit_processor: str = None,
    ) -> List[Dict[str, Any]]:
        pass


@pytest.fixture
def dummy_dataset():
    """Returns a dummy dataset."""
    data = {
        'prompt': ['hello world', 'foo bar'],
        'ground_truth': ['expected output one', 'expected output two'],
    }
    return datasets.Dataset.from_dict(data)


@pytest.fixture
def dummy_reward():
    dummy_fn = Mock(spec=BaseRewardFunction)
    dummy_fn.name = 'dummy'
    return dummy_fn


# Fixtures
@pytest.fixture
def inference_env(dummy_dataset, dummy_reward, mock_tokenizer):
    """Create an ExploreInferenceEnv instance with mock dependencies."""

    env = ExploreInferenceEnv(
        temperatures=torch.tensor([1.0, 1.0]),
        explore_steps=0,
        explore_top_k=20,
        explore_skip_n=0,
        explore_decay_rate=0.9,
        continue_special_tokens=['<CONT>'],
        continue_max_retry=2,
        continue_prob=0.5,
        dataset=dummy_dataset,
        reward_functions=[dummy_reward],
        batch_size=2,
        group_size=2,
        tokenizer=mock_tokenizer,
    )
    env.accuracy_fn = dummy_reward  # Assign accuracy_fn explicitly
    return env


@pytest.fixture
def mock_llm():
    """Create a mock InferenceClient."""
    return Mock(spec=InferenceClient)


@pytest.fixture
def env_state():
    """Create an EnvState with sample prompts and ground truths."""
    return EnvState(
        prompt=['Prompt 1', 'Prompt 2'], ground_truth=['Answer 1', 'Answer 2']
    )


def test_generate_completions_with_retries(inference_env, mock_llm, env_state):
    """Test generation with retries for incorrect outputs."""
    inference_env.continue_prob = 1.0
    inference_env.continue_max_retry = 2
    inference_env.accuracy_fn.side_effect = [
        [0.0],  # Pass 1, idx 0 ("Wrong 1") -> Incorrect
        [1.0],  # Pass 1, idx 1 ("Correct 2") -> Correct
        [1.0],  # Pass 2, idx 0 ("Wrong 1<CONT>Fixed 1") -> Correct
    ]
    mock_llm.generate.side_effect = [
        [
            {
                'text': 'Wrong 1',
                'meta_info': {'finish_reason': {'type': 'length'}},
            },
            {
                'text': 'Correct 2',
                'meta_info': {'finish_reason': {'type': 'stop'}},
            },
        ],  # First pass
        [
            {
                'text': 'Fixed 1',
                'meta_info': {'finish_reason': {'type': 'length'}},
            },
        ],  # Retry for item 1
    ]
    texts, tokens, lengths = inference_env._generate_completions(
        mock_llm, {}, env_state
    )
    assert texts == ['Wrong 1<CONT>Fixed 1', 'Correct 2']
    assert all(isinstance(t, torch.Tensor) for t in tokens)


# Tests for _should_retry
@pytest.mark.parametrize(
    'is_correct,retry_attempts_left,continue_prob,expected',
    [
        (True, 1, 0.5, False),  # Correct, no retry
        (False, -1, 0.5, False),  # No retries left
        (False, 0, 0.0, False),  # Retries disabled
    ],
)
def test_should_retry_no_retry_cases(
    inference_env, is_correct, retry_attempts_left, continue_prob, expected
):
    """Test cases where retry should not occur."""
    inference_env.continue_prob = continue_prob
    random.seed(42)  # Control randomness
    result = inference_env._should_retry(is_correct, retry_attempts_left)
    assert result == expected


def test_should_retry_with_probability(inference_env):
    """Test retry decision with probability check passing."""
    inference_env.continue_prob = 1.0  # Force probability check to pass
    result = inference_env._should_retry(False, 1)
    assert result is True


# Tests for _generate_completions
def test_generate_completions_single_pass_no_retries(
    inference_env, mock_llm, env_state
):
    """Test generation with retries disabled completes in one pass."""
    inference_env.continue_prob = 0.0
    mock_llm.generate.return_value = [
        {'text': 'Output 1', 'meta_info': {'finish_reason': {'type': 'stop'}}},
        {'text': 'Output 2', 'meta_info': {'finish_reason': {'type': 'stop'}}},
    ]
    texts, tokens, lengths = inference_env._generate_completions(
        mock_llm, {}, env_state
    )
    assert texts == ['Output 1', 'Output 2']
    assert all(isinstance(t, torch.Tensor) for t in tokens)
    assert lengths == [3, 3]  # DummyTokenizer: 2 words + EOS


def test_generate_completions_no_accuracy_fn(
    inference_env, mock_llm, env_state
):
    """Test generation stops after one pass when no accuracy function is set."""
    inference_env.accuracy_fn = None
    mock_llm.generate.return_value = [
        {'text': 'Output 1', 'meta_info': {}},
        {'text': 'Output 2', 'meta_info': {}},
    ]
    texts, tokens, lengths = inference_env._generate_completions(
        mock_llm, {}, env_state
    )
    assert texts == ['Output 1', 'Output 2']
    assert mock_llm.generate.call_count == 1
