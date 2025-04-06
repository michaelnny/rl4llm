import random

import datasets
import numpy as np
import pytest
import torch

from rl4llm.envs.hf_env import (
    BaseRewardFunction,
    EnvState,
    EpisodeData,
    HFEnv,
)

# Dummy implementations for testing


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

    def __call__(self, texts, truncation=False, max_length=None, padding=False):
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
        self, sequences, skip_special_tokens, clean_up_tokenization_spaces
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


class DummyRewardFunction(BaseRewardFunction):
    """A dummy reward function that always returns 1.0 for testing."""

    def __init__(self, name='dummy_reward'):
        super().__init__(name)

    def __call__(self, completions, ground_truths):
        return [1.0 for _ in completions]


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
def dummy_tokenizer():
    """Returns a dummy tokenizer."""
    return DummyTokenizer()


@pytest.fixture
def dummy_reward():
    """Returns a dummy reward function."""
    return DummyRewardFunction()


@pytest.fixture
def dummy_model():
    """Returns a dummy model."""
    return DummyModel()


# Tests


def test_initialization(dummy_dataset, dummy_tokenizer, dummy_reward):
    """Tests HFEnv initializes correctly with valid parameters."""
    env = HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
        batch_size=2,
        group_size=3,
    )
    assert env.batch_size == 2
    assert env.group_size == 3


@pytest.mark.parametrize(
    'batch_size, group_size, reward_functions, dataset, error_msg',
    [
        (
            0,
            1,
            [DummyRewardFunction()],
            datasets.Dataset.from_dict(
                {'prompt': ['test'], 'ground_truth': ['gt']}
            ),
            'Batch size must be at least 1',
        ),
        (
            1,
            0,
            [DummyRewardFunction()],
            datasets.Dataset.from_dict(
                {'prompt': ['test'], 'ground_truth': ['gt']}
            ),
            'Group size must be at least 1',
        ),
        (
            1,
            1,
            [],
            datasets.Dataset.from_dict(
                {'prompt': ['test'], 'ground_truth': ['gt']}
            ),
            'reward_functions must be a non-empty list',
        ),
        (
            1,
            1,
            [DummyRewardFunction()],
            'not a dataset',
            'dataset must be a datasets.Dataset instance.',
        ),
        (
            1,
            1,
            [DummyRewardFunction()],
            datasets.Dataset.from_dict({'prompt': ['test']}),
            "Dataset must contain 'prompt' and 'ground_truth' columns.",
        ),
    ],
)
def test_invalid_initialization(
    batch_size,
    group_size,
    reward_functions,
    dataset,
    error_msg,
    dummy_tokenizer,
):
    """Tests HFEnv initialization errors with invalid parameters."""
    with pytest.raises(Exception) as excinfo:
        HFEnv(
            dataset=dataset,
            tokenizer=dummy_tokenizer,
            reward_functions=reward_functions,
            batch_size=batch_size,
            group_size=group_size,
        )
    assert error_msg in str(excinfo.value)


def test_setup_tokenizer(dummy_dataset, dummy_tokenizer, dummy_reward):
    """Tests tokenizer setup assigns pad_token if missing."""
    dummy_tokenizer.pad_token = None
    dummy_tokenizer.eos_token = '<eos>'
    dummy_tokenizer.eos_token_id = 1
    HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
        batch_size=1,
        group_size=1,
    )
    assert dummy_tokenizer.pad_token == '<eos>'
    assert dummy_tokenizer.pad_token_id == 1


def test_collate_fn(dummy_dataset, dummy_tokenizer, dummy_reward):
    """Tests collate function pads inputs correctly."""
    env = HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
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


def test_reset_and_grouped_state(dummy_dataset, dummy_tokenizer, dummy_reward):
    """Tests reset returns an EnvState with repeated prompts."""
    env = HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
        batch_size=2,
        group_size=2,
    )
    state = env._reset()
    assert state is not None
    assert len(state.prompt) == 4
    expected_prompts = [p for p in dummy_dataset['prompt'] for _ in range(2)]
    assert state.prompt == expected_prompts


def test_rollout(dummy_dataset, dummy_tokenizer, dummy_reward, dummy_model):
    """Tests rollout returns valid EpisodeData objects."""
    env = HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
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
    dummy_dataset, dummy_tokenizer, dummy_reward, dummy_model
):
    """Tests rollout raises error when using num_return_sequences > 1."""
    env = HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
        batch_size=1,
        group_size=1,
    )
    gen_args = {'num_return_sequences': 2}
    with pytest.raises(ValueError) as excinfo:
        env.rollout(dummy_model, gen_args)
    assert 'Set group_size during initialization' in str(excinfo.value)


def test_rollout_no_pad_tokens(
    dummy_dataset, dummy_tokenizer, dummy_reward, dummy_model
):
    """Tests rollout returns data with no pad tokens in prompt or completion."""
    env = HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
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
    dummy_dataset, dummy_tokenizer, dummy_reward, dummy_model
):
    """Tests rollout returns expected rewards from dummy reward function."""
    env = HFEnv(
        dataset=dummy_dataset,
        tokenizer=dummy_tokenizer,
        reward_functions=[dummy_reward],
        batch_size=2,
        group_size=2,
    )
    gen_args = {'max_new_tokens': 5}
    episodes = env.rollout(dummy_model, gen_args)
    for ep in episodes:
        assert ep.reward_dict.get('dummy_reward') == 1.0
