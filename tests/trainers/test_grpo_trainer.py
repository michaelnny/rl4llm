import math
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.trainers.grpo_trainer import (
    EpisodeData,
    GRPOConfig,
    GRPOTrainer,
    HFEnv,
    TransitionData,
)


class DummyPolicyEngine:
    """Dummy policy engine with minimal implementation."""

    def zero_optimization_stage(self):
        return 0  # Dummy value that avoids triggering any branch

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=input_ids.float() + 0.1)

    def backward(self, loss):
        self.last_loss = loss

    def step(self):
        pass

    def is_gradient_accumulation_boundary(self):
        return True

    def get_lr(self):
        return [0.001]

    def bfloat16_enabled(self):
        return True

    def float16_enabled(self):
        return False


class DummyLogger:
    def __init__(self):
        self.scalars = {}
        self.messages = []

    def log_scalar(self, key, value):
        self.scalars.setdefault(key, []).append(value)

    def warning(self, msg):
        self.messages.append(('warning', msg))

    def error(self, msg):
        self.messages.append(('error', msg))


class DummyEnv:
    def rollout(self, model, gen_kwargs, **custom_kwargs):
        return [DummyEpisode(0.5, [2, 3], [4, 5], 2, 2)]


class DummyEpisode:
    def __init__(
        self,
        reward,
        prompt_tokens,
        completion_tokens,
        prompt_length,
        completion_length,
    ):
        self.reward_dict = {'accuracy_reward': reward}
        self.prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
        self.completion_tokens = torch.tensor(
            completion_tokens, dtype=torch.long
        )
        self.prompt_length = prompt_length
        self.completion_length = completion_length


class DummyDistributedManager:
    """Dummy distributed manager with required attributes and methods."""

    def __init__(self, world_size=1, device=torch.device('cpu')):
        self.world_size = world_size
        self.global_rank = 0
        self.device = device
        self.is_master = True

    def barrier(self):
        pass


def dummy_compute_logprobs_from_logits(logits, actions):
    return torch.full_like(actions, 0.2, dtype=torch.float32)


def dummy_masked_whiten(x, mask):
    return x


def dummy_masked_mean(x, mask, dim=None):
    m = mask.float()
    return (x * m).sum(dim=dim) / (m.sum(dim=dim) + 1e-8)


def dummy_clean_up():
    pass


class DummyContextManager:
    def __enter__(self):
        return SimpleNamespace(
            forward=lambda **kwargs: SimpleNamespace(
                logits=kwargs['input_ids'].float() + 0.1
            )
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Fixtures


@pytest.fixture
def dummy_config():
    """Fixture returning a dummy GRPOConfig."""

    return GRPOConfig.construct(
        xml_format=False,
        group_temperature=False,
        min_temperature=0.6,
        max_temperature=1.2,
        explore_init_epsilon=0.9,
        explore_min_epsilon=0.1,
        explore_decay_steps=10,
        explore_steps=0,
        explore_top_k=0,
        explore_decay_rate=0.8,
        replace_max_per_seq=0,
        replace_prob=0.0,
        group_size=10,
        train_micro_batch_size=2,
        normalize_advantages=True,
        clip_eps=0.2,
        entropy_loss_coef=0.01,
        kl_loss_coef=0.5,
        num_updates=1,
        train_rollout_size=4,
        eval_rollout_size=4,
        eval_batch_size=1,
        max_completion_tokens=5,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        normalize_rewards=True,
    )


@pytest.fixture
def dummy_tokenizer():
    """Fixture returning a dummy tokenizer."""
    return SimpleNamespace(pad_token_id=0, eos_token_id=1)


@pytest.fixture
def dummy_policy_engine():
    """Fixture returning a dummy policy engine."""
    return MagicMock()


@pytest.fixture
def dummy_dist_manager():
    """Fixture returning a dummy distributed manager with barrier method."""
    return DummyDistributedManager(world_size=1, device=torch.device('cpu'))


@pytest.fixture
def dummy_logger():
    """Fixture returning a dummy logger."""
    return MagicMock()


@pytest.fixture
def dummy_train_env():
    """Fixture returning a dummy training environment."""
    return MagicMock()


@pytest.fixture
def dummy_eval_env():
    """Fixture returning a dummy evaluation environment."""
    return MagicMock()


@pytest.fixture
def dummy_trainer(
    dummy_config,
    dummy_tokenizer,
    dummy_policy_engine,
    dummy_dist_manager,
    dummy_logger,
    dummy_train_env,
):
    """Fixture returning a dummy GRPOTrainer instance."""

    with tempfile.TemporaryDirectory() as tmp_artifacts_path:
        trainer = GRPOTrainer(
            config=dummy_config,
            tokenizer=dummy_tokenizer,
            policy_engine=dummy_policy_engine,
            dist_manager=dummy_dist_manager,
            logger=dummy_logger,
            artifacts_path=tmp_artifacts_path,
            train_env=dummy_train_env,
            eval_env=None,
            seed=42,
        )
        trainer.device = torch.device('cpu')
        trainer.torch_dtype = torch.float32
        trainer.global_step = 0
        trainer.policy_update_count = 0
        trainer.compute_logprobs_from_logits = (
            dummy_compute_logprobs_from_logits
        )
        trainer.masked_whiten = dummy_masked_whiten
        trainer.masked_mean = dummy_masked_mean
        trainer.clean_up = dummy_clean_up
        trainer.unwrapped_model_for_generation = lambda: DummyContextManager()
        return trainer


# Tests for initialize_trainer


def test_initialize_trainer(dummy_trainer):
    """Test that initialize_trainer sets group_reward_std_threshold and explore_epsilon."""
    dummy_trainer.initialize_trainer()
    expected_dummy = torch.tensor(
        [1.0] + [0.0] * (dummy_trainer.config.group_size - 1),
        dtype=torch.float32,
    )
    expected_threshold = torch.std(expected_dummy, unbiased=False)
    assert torch.allclose(
        dummy_trainer.group_reward_std_threshold, expected_threshold
    )
    assert dummy_trainer.explore_epsilon == 0.0


# Tests for compute_loss


def test_compute_loss_without_kl(dummy_trainer):
    """Test compute_loss without KL loss when kl_loss_coef=0."""

    dummy_trainer.config.kl_loss_coef = 0.0
    tensor = torch.full((2, 3), 0.2, dtype=torch.float32)
    td = TransitionData(
        states=tensor,
        actions=torch.ones((2, 3), dtype=torch.long) * 2,
        loss_mask=torch.ones((2, 3), dtype=torch.bool),
        pi_logprobs=tensor,
        ref_logprobs=tensor,
        advantages=torch.ones((2, 3), dtype=torch.float32),
    )
    pi_logits = torch.full((2, 3), 0.2, dtype=torch.float32)
    loss, metrics = dummy_trainer.compute_loss(pi_logits, td)
    expected_keys = {
        'train/pg_loss',
        'train/entropy_loss',
        'policy/entropy',
        'policy/approxkl',
        'policy/clipfrac',
    }
    assert expected_keys.issubset(metrics.keys())
    assert isinstance(loss, torch.Tensor)


def test_compute_loss_with_kl(dummy_trainer):
    """Test compute_loss with KL loss when kl_loss_coef > 0."""

    dummy_trainer.config.kl_loss_coef = 0.5
    tensor = torch.full((2, 3), 0.2, dtype=torch.float32)
    td = TransitionData(
        states=tensor,
        actions=torch.ones((2, 3), dtype=torch.long) * 2,
        loss_mask=torch.ones((2, 3), dtype=torch.bool),
        pi_logprobs=tensor,
        ref_logprobs=tensor,
        advantages=torch.ones((2, 3), dtype=torch.float32),
    )
    pi_logits = torch.full((2, 3), 0.2, dtype=torch.float32)
    loss, metrics = dummy_trainer.compute_loss(pi_logits, td)
    expected_keys = {
        'train/pg_loss',
        'train/entropy_loss',
        'policy/entropy',
        'policy/approxkl',
        'policy/clipfrac',
        'train/kl_loss',
        'objective/kl',
    }
    assert expected_keys.issubset(metrics.keys())


# Parameterized tests for _normalize_group_rewards


@pytest.mark.parametrize(
    'rewards_list, zero_mean_only, expected',
    [
        ([1.0, 2.0, 3.0, 4.0], True, [-1.5, -0.5, 0.5, 1.5]),
        (
            [1.0, 2.0, 3.0, 4.0],
            False,
            [
                -1.3416407864998738,
                -0.4472135954999579,
                0.4472135954999579,
                1.3416407864998738,
            ],
        ),
    ],
)
def test_normalize_group_rewards(
    dummy_trainer, rewards_list, zero_mean_only, expected
):
    """Test _normalize_group_rewards returns correctly normalized rewards."""
    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    normalized = dummy_trainer._normalize_group_rewards(
        rewards, zero_mean_only=zero_mean_only
    )
    for a, b in zip(normalized.tolist(), expected):
        assert pytest.approx(a, rel=1e-5) == b


def test_normalize_group_rewards_error(dummy_trainer):
    """Test _normalize_group_rewards raises ValueError for too few rewards."""
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    with pytest.raises(ValueError):
        dummy_trainer._normalize_group_rewards(rewards)


# Parameterized tests for _get_exploration_epsilon


@pytest.mark.parametrize(
    'global_step, decay_steps, expected',
    [
        (0, 10, 0.9),
        (10, 10, 0.1),
        (5, 10, None),
    ],
)
def test_get_exploration_epsilon(
    dummy_trainer, global_step, decay_steps, expected
):
    """Test _get_exploration_epsilon returns correct value based on global step."""
    dummy_trainer.config.explore_decay_steps = decay_steps
    dummy_trainer.config.explore_min_epsilon = 0.1
    dummy_trainer.config.explore_init_epsilon = 0.9
    dummy_trainer.global_step = global_step
    epsilon = dummy_trainer._get_exploration_epsilon()
    if expected is not None:
        assert pytest.approx(epsilon, rel=1e-5) == expected
    else:
        progress = global_step / decay_steps
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        expected_val = 0.1 + (0.9 - 0.1) * cosine_decay
        assert pytest.approx(epsilon, rel=1e-5) == expected_val


def test_get_exploration_epsilon_zero_decay(dummy_trainer):
    """Test _get_exploration_epsilon returns 0 when decay steps is 0."""
    dummy_trainer.config.explore_decay_steps = 0
    dummy_trainer.global_step = 5
    epsilon = dummy_trainer._get_exploration_epsilon()
    assert epsilon == 0.0


# Test for _train_collate_fn


def test_train_collate_fn(dummy_trainer):
    """Test _train_collate_fn correctly pads TransitionData fields."""

    td1 = TransitionData(
        states=torch.tensor([2, 3, 4]),
        actions=torch.tensor([3, 4, 5]),
        loss_mask=torch.tensor([True, True, True]),
        pi_logprobs=torch.tensor([0.2, 0.2, 0.2]),
        ref_logprobs=torch.tensor([0.2, 0.2, 0.2]),
        advantages=torch.tensor([1.0, 1.0, 1.0]),
    )
    td2 = TransitionData(
        states=torch.tensor([2, 3]),
        actions=torch.tensor([3, 4]),
        loss_mask=torch.tensor([True, True]),
        pi_logprobs=torch.tensor([0.2, 0.2]),
        ref_logprobs=torch.tensor([0.2, 0.2]),
        advantages=torch.tensor([1.0, 1.0]),
    )
    collated = dummy_trainer._train_collate_fn([td1, td2])
    assert collated.states.shape == (2, 3)


# Tests for _convert_group_episodes_to_transitions


def test_convert_group_episodes_empty(dummy_trainer):
    """Test _convert_group_episodes_to_transitions returns empty list for empty episodes."""
    transitions = dummy_trainer._convert_group_episodes_to_transitions([])
    assert transitions == []


def test_convert_group_episodes_too_few(dummy_trainer):
    """Test _convert_group_episodes_to_transitions raises ValueError for fewer than 4 episodes."""
    episode = DummyEpisode(0.5, [2, 3], [4, 5], 2, 2)
    with pytest.raises(ValueError):
        dummy_trainer._convert_group_episodes_to_transitions(
            [episode, episode, episode]
        )


def test_convert_group_episodes_valid(dummy_trainer):
    """Test _convert_group_episodes_to_transitions returns transitions for valid episodes."""
    episode1 = DummyEpisode(0.0, [2, 3, 4], [5, 6, 7], 3, 3)
    episode2 = DummyEpisode(1.0, [8, 9, 10], [11, 12, 13], 3, 3)
    episode3 = DummyEpisode(2.0, [3, 4, 5], [6, 7, 8], 3, 3)
    episode4 = DummyEpisode(3.0, [4, 5, 6], [7, 8, 9], 3, 3)
    dummy_trainer.group_reward_std_threshold = 0.1
    transitions = dummy_trainer._convert_group_episodes_to_transitions(
        [episode1, episode2, episode3, episode4]
    )
    assert len(transitions) == 4
    for ep, trans in zip([episode1, episode2, episode3, episode4], transitions):
        assert trans.loss_mask.sum().item() == ep.completion_length


def test_group_episode_invalid_limit_samples(dummy_trainer):
    """Test _check_group_episodes for valid episodes not enough samples."""
    episode1 = DummyEpisode(0.0, [2, 3, 4], [5, 6, 7], 3, 3)
    episode2 = DummyEpisode(1.0, [8, 9, 10], [11, 12, 13], 3, 3)
    dummy_trainer.group_reward_std_threshold = 0.1

    result = dummy_trainer._check_group_episodes([episode1, episode2])
    assert result is False


def test_group_episode_invalid_low_std(dummy_trainer, dummy_logger):
    """Test _check_group_episodes returns empty list for low reward std."""
    episode1 = DummyEpisode(1.0, [2, 3, 4], [5, 6], 3, 2)
    episode2 = DummyEpisode(1.0, [2, 3, 4], [5, 6], 3, 2)
    episode3 = DummyEpisode(1.0, [2, 3, 4], [5, 6], 3, 2)
    episode4 = DummyEpisode(1.0, [2, 3, 4], [5, 6], 3, 2)
    episode5 = DummyEpisode(0.0, [2, 3, 4], [5, 6], 3, 2)
    episode6 = DummyEpisode(0.0, [2, 3, 4], [5, 6], 3, 2)

    dummy_trainer.group_reward_std_threshold = 0.5
    result = dummy_trainer._check_group_episodes(
        [episode1, episode2, episode3, episode4, episode5, episode6]
    )
    assert result is False


def test_evaluate_step_without_env(dummy_trainer):
    """Test evaluate_step does nothing when eval_env is None."""
    dummy_trainer.eval_env = None
    dummy_trainer.evaluate_step()
    pass
