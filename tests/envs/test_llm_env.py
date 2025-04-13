import random

import datasets
import numpy as np
import pytest
import torch

from rl4llm.envs.llm_env import (
    BaseRewardFunction,
    EnvState,
    EpisodeData,
    LocalLLMEnv,
)

# Dummy implementations for testing


class DummyModel:
    """A dummy model for testing generation."""

    def __init__(self, device='cpu'):
        self.device = device

    def generate(self, input_ids, attention_mask, **gen_args):
        batch_size = input_ids.shape[0]
        # Create a dummy completion: a tensor of constant tokens (e.g. token '2')
        completion = torch.full((batch_size, 5), 2, dtype=torch.long)
        sequences = torch.cat([input_ids, completion], dim=1)

        class DummyOutput:
            pass

        out = DummyOutput()
        out.sequences = sequences
        return out


# Fixtures


@pytest.fixture
def dummy_dataset():
    """Returns a dummy dataset."""
    data = {
        'prompt': ['hello world', 'foo bar'],
        'ground_truth': ['expected output one', 'expected output two'],
    }
    return datasets.Dataset.from_dict(data)


@pytest.fixture
def dummy_model():
    """Returns a dummy model."""
    return DummyModel()


# Tests


def test_initialization(dummy_dataset, mock_tokenizer, mock_reward_function):
    """Tests LocalLLMEnv initializes correctly with valid parameters."""
    env = LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=2,
        group_size=3,
    )
    assert env.batch_size == 2
    assert env.group_size == 3


def test_setup_tokenizer(dummy_dataset, mock_tokenizer, mock_reward_function):
    """Tests tokenizer setup assigns pad_token if missing."""
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = '<eos>'
    mock_tokenizer.eos_token_id = 1
    LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=1,
        group_size=1,
    )
    assert mock_tokenizer.pad_token == '<eos>'
    assert mock_tokenizer.pad_token_id == 1


def test_collate_fn(dummy_dataset, mock_tokenizer, mock_reward_function):
    """Tests collate function pads inputs correctly."""
    env = LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=1,
        group_size=1,
    )
    batch = [
        {
            'input_ids': [2, 3, 4],
            'attention_mask': [1, 1, 1],
            'ground_truth': 'gt1',
            'prompt': 'p1',
        },
        {
            'input_ids': [5, 6],
            'attention_mask': [1, 1],
            'ground_truth': 'gt2',
            'prompt': 'p2',
        },
    ]
    collated = env._collate_fn(batch)
    assert collated['input_ids'].shape == (2, 3)
    assert collated['ground_truth'] == ['gt1', 'gt2']


def test_reset_and_grouped_state(
    dummy_dataset, mock_tokenizer, mock_reward_function
):
    """Tests reset returns an EnvState with repeated prompts."""
    env = LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=2,
        group_size=2,
    )
    state = env._reset()
    assert state is not None
    assert len(state.prompt) == 4
    expected_prompts = [p for p in dummy_dataset['prompt'] for _ in range(2)]
    assert state.prompt == expected_prompts


def test_rollout(
    dummy_dataset, mock_tokenizer, mock_reward_function, dummy_model
):
    """Tests rollout returns valid EpisodeData objects."""
    env = LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=2,
        group_size=2,
    )
    gen_args = {'max_new_tokens': 5}
    episodes = env.rollout(dummy_model, gen_args)
    assert len(episodes) == 4
    episode = episodes[0]
    assert hasattr(episode, 'prompt_text')
    assert hasattr(episode, 'completion_text')
    assert hasattr(episode, 'reward_dict')


def test_invalid_gen_args(
    dummy_dataset, mock_tokenizer, mock_reward_function, dummy_model
):
    """Tests rollout raises error when using num_return_sequences > 1."""
    env = LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=1,
        group_size=1,
    )
    gen_args = {'num_return_sequences': 2}
    with pytest.raises(ValueError) as excinfo:
        env.rollout(dummy_model, gen_args)
    assert 'Set group_size during initialization' in str(excinfo.value)


def test_rollout_no_pad_tokens(
    dummy_dataset, mock_tokenizer, mock_reward_function, dummy_model
):
    """Tests rollout returns data with no pad tokens in prompt or completion."""
    env = LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=2,
        group_size=2,
    )
    gen_args = {'max_new_tokens': 5}
    episodes = env.rollout(dummy_model, gen_args)
    for ep in episodes:
        prompt_tokens = (
            ep.prompt_tokens.tolist()
            if isinstance(ep.prompt_tokens, torch.Tensor)
            else list(ep.prompt_tokens)
        )
        completion_tokens = (
            ep.completion_tokens.tolist()
            if isinstance(ep.completion_tokens, torch.Tensor)
            else list(ep.completion_tokens)
        )
        # Check that pad token (id 0) does not appear in prompt tokens
        assert 0 not in prompt_tokens
        # Check that pad token (id 0) does not appear in completion tokens
        assert 0 not in completion_tokens
        # If EOS token (id 1) appears, allow it only at the very end of completions.
        if 1 in completion_tokens:
            if completion_tokens[-1] == 1:
                assert all(token != 1 for token in completion_tokens[:-1])
            else:
                pytest.fail(
                    'EOS token found in the middle of completion tokens.'
                )


def test_rollout_rewards(
    dummy_dataset, mock_tokenizer, mock_reward_function, dummy_model
):
    """Tests rollout returns expected rewards from dummy reward function."""
    env = LocalLLMEnv(
        dataset=dummy_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=2,
        group_size=2,
    )
    gen_args = {'max_new_tokens': 5}
    episodes = env.rollout(dummy_model, gen_args)
    for ep in episodes:
        assert ep.reward_dict.get('mock_reward_function') == 1.0
