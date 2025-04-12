# flake8: noqa

import math
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from transformers import PreTrainedTokenizer

# Objects under test
from rl4llm.envs import EpisodeData
from rl4llm.trainers.extended_grpo_trainer import (
    ExtendedGRPOConfig,
    ExtendedGRPOTrainer,
    TransitionData,
)

# --- Fixtures ---


@pytest.fixture
def grpo_config() -> ExtendedGRPOConfig:
    """Provides a default ExtendedGRPOConfig instance for testing."""
    return ExtendedGRPOConfig(
        train_rollout_size=16,
        train_micro_batch_size=4,
        group_size=4,
        kl_loss_coef=0.1,
        entropy_loss_coef=0.01,
        normalize_advantages=True,
        normalize_rewards=True,
        clip_eps=0.2,
        num_updates=1,
        max_completion_tokens=10,
        explore_decay_steps=100,
        explore_init_epsilon=0.5,
        explore_min_epsilon=0.1,
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Provides a mock tokenizer instance."""
    mock = MagicMock(
        spec=PreTrainedTokenizer
    )  # spec helps catch missing attributes/methods
    mock.pad_token_id = 0
    mock.eos_token_id = 0  # Often same as pad for GPT-2 style
    mock.vocab_size = 1000  # Use a smaller, arbitrary vocab size for tests

    # Simple mock encode: returns tensor of token lengths (modulo vocab_size)
    def _mock_encode(text, return_tensors=None):
        # Split text into 'words' and assign a simple ID (length)
        ids = [len(word) % mock.vocab_size for word in text.split()]
        if not ids:  # Handle empty string
            ids = [mock.eos_token_id]
        if return_tensors == 'pt':
            return torch.tensor(ids, dtype=torch.long)
        return ids

    mock.encode = Mock(side_effect=_mock_encode)
    # Make the mock callable like a tokenizer instance if needed elsewhere
    # mock.__call__ = Mock(side_effect=_mock_encode) # Not strictly needed for these tests

    return mock


@pytest.fixture
def mock_policy_engine(
    mock_tokenizer: MagicMock,
) -> MagicMock:  # Depends on mock_tokenizer
    """Provides a mock DeepSpeedEngine for the policy model."""
    engine = MagicMock()
    engine.device = torch.device('cpu')
    vocab_size = mock_tokenizer.vocab_size  # Use vocab_size from mock
    engine.forward.return_value = Mock(
        logits=torch.randn(4, 10, vocab_size)
    )  # batch, seq_len, vocab_size
    engine.backward = Mock()
    engine.step = Mock()
    engine.is_gradient_accumulation_boundary.return_value = True
    engine.get_lr.return_value = [0.0001]
    return engine


@pytest.fixture
def mock_ref_model(
    mock_tokenizer: MagicMock,
) -> MagicMock:  # Depends on mock_tokenizer
    """Provides a mock model for the reference model."""
    model = MagicMock()
    model.device = torch.device('cpu')
    vocab_size = mock_tokenizer.vocab_size  # Use vocab_size from mock
    model.forward.return_value = Mock(
        logits=torch.randn(4, 10, vocab_size)
    )  # batch, seq_len, vocab_size
    return model


@pytest.fixture
def mock_dist_manager() -> MagicMock:
    """Provides a mock DistributedManager."""
    manager = MagicMock()
    manager.world_size = 1
    manager.is_main_process.return_value = True
    return manager


@pytest.fixture
def mock_logger() -> MagicMock:
    """Provides a mock LoggingManager."""
    return MagicMock()


@pytest.fixture
def mock_train_env() -> MagicMock:
    """Provides a mock LocalLLMEnv for training."""
    env = MagicMock()
    env.reward_functions = {'reward1': Mock()}
    return env


@pytest.fixture
def mock_reward_transform_fn() -> MagicMock:
    """Provides a mock reward transformation function."""
    fn = MagicMock()
    # Simple sum aggregation for testing
    fn.side_effect = lambda rewards_dict: sum(rewards_dict.values())
    return fn


@pytest.fixture
def sample_episode_data(
    mock_tokenizer: MagicMock,
) -> EpisodeData:  # Depends on mock_tokenizer
    """Provides a sample EpisodeData instance."""
    prompt = 'Once upon a time'
    completion = ' there was a dragon.'
    # Use the mock tokenizer's encode method
    prompt_tokens = mock_tokenizer.encode(prompt, return_tensors='pt')
    completion_tokens = mock_tokenizer.encode(completion, return_tensors='pt')
    return EpisodeData(
        prompt_text=prompt,
        completion_text=completion,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_length=len(prompt_tokens),
        completion_length=len(completion_tokens),
        reward_dict={'reward1': 1.5},
        meta_data={},
    )


@pytest.fixture
def sample_group_episodes(sample_episode_data) -> List[EpisodeData]:
    """Provides a list of sample EpisodeData instances for a group."""
    # Create slightly different rewards for testing normalization/std check
    ep1 = sample_episode_data.model_copy(deep=True)
    ep1.reward_dict = {'reward1': 1.5}
    ep2 = sample_episode_data.model_copy(deep=True)
    ep2.reward_dict = {'reward1': 2.0}
    ep3 = sample_episode_data.model_copy(deep=True)
    ep3.reward_dict = {'reward1': 1.0}
    ep4 = sample_episode_data.model_copy(deep=True)
    ep4.reward_dict = {'reward1': 2.5}
    return [ep1, ep2, ep3, ep4]


@pytest.fixture
def sample_transition_data(
    mock_tokenizer: MagicMock,
) -> TransitionData:  # Depends on mock_tokenizer
    """Provides a sample TransitionData instance representing ONE sequence."""
    seq_len = 10
    vocab_size = mock_tokenizer.vocab_size  # Use vocab_size from mock
    return TransitionData(
        states=torch.randint(0, vocab_size, (seq_len,)),  # Shape (seq_len,)
        actions=torch.randint(0, vocab_size, (seq_len,)),
        loss_mask=torch.ones(seq_len, dtype=torch.bool),
        pi_logprobs=torch.randn(
            seq_len,
        ),
        ref_logprobs=torch.randn(
            seq_len,
        ),
        advantages=torch.randn(
            seq_len,
        ),
    )


@pytest.fixture
def grpo_trainer(
    grpo_config: ExtendedGRPOConfig,
    mock_tokenizer: MagicMock,
    mock_policy_engine: MagicMock,
    mock_dist_manager: MagicMock,
    mock_logger: MagicMock,
    mock_train_env: MagicMock,
    mock_ref_model: MagicMock,
    mock_reward_transform_fn: MagicMock,
) -> ExtendedGRPOTrainer:
    """Provides a ExtendedGRPOTrainer instance with mocked dependencies."""
    # Mock the reward transform function only if needed (multiple rewards)
    reward_fn = None
    if len(mock_train_env.reward_functions) > 1:
        reward_fn = mock_reward_transform_fn

    trainer = ExtendedGRPOTrainer(
        config=grpo_config,
        tokenizer=mock_tokenizer,  # Pass the mock tokenizer
        policy_engine=mock_policy_engine,
        dist_manager=mock_dist_manager,
        logger=mock_logger,
        artifacts_path='/tmp/test_grpo',
        train_env=mock_train_env,
        ref_model=mock_ref_model,
        reward_transform_fn=reward_fn,
        seed=42,
    )
    # Manually set device and dtype for consistency in tests
    trainer.device = torch.device('cpu')
    trainer.torch_dtype = torch.float32
    trainer.initialize_trainer()  # Initialize trainer specific settings
    return trainer


# --- Test Cases ---


def test_initialize_trainer(grpo_trainer: ExtendedGRPOTrainer):
    """Tests if trainer-specific attributes are initialized correctly."""
    assert grpo_trainer.explore_epsilon == 0.0
    assert isinstance(grpo_trainer.group_reward_std_threshold, torch.Tensor)
    assert (
        grpo_trainer.group_reward_std_threshold > 0
    )  # Should be calculated based on dummy rewards


def test_get_exploration_epsilon_decay(grpo_trainer: ExtendedGRPOTrainer):
    """Tests the cosine decay calculation for exploration epsilon."""
    grpo_trainer.config.explore_init_epsilon = 0.5
    grpo_trainer.config.explore_min_epsilon = 0.1
    grpo_trainer.config.explore_decay_steps = 100

    # Start
    grpo_trainer.global_step = 0
    assert grpo_trainer._get_exploration_epsilon() == pytest.approx(0.5)

    # Mid decay
    grpo_trainer.global_step = 50
    expected_mid = 0.1 + (0.5 - 0.1) * 0.5 * (1 + math.cos(math.pi * 50 / 100))
    assert grpo_trainer._get_exploration_epsilon() == pytest.approx(
        expected_mid
    )  # Should be 0.3

    # End
    grpo_trainer.global_step = 100
    assert grpo_trainer._get_exploration_epsilon() == pytest.approx(0.1)

    # After decay
    grpo_trainer.global_step = 150
    assert grpo_trainer._get_exploration_epsilon() == pytest.approx(0.1)


def test_get_exploration_epsilon_no_decay(grpo_trainer: ExtendedGRPOTrainer):
    """Tests exploration epsilon when decay steps are zero."""
    grpo_trainer.config.explore_decay_steps = 0
    grpo_trainer.config.explore_init_epsilon = 0.5
    grpo_trainer.global_step = 50
    assert grpo_trainer._get_exploration_epsilon() == 0.0


def test_check_group_episodes(
    grpo_trainer: ExtendedGRPOTrainer, sample_group_episodes: List[EpisodeData]
):
    """Tests the validity check for a group of episodes."""
    # Mock the transform function to return rewards with sufficient std dev
    with patch.object(
        grpo_trainer,
        'transform_batch_rewards',
        return_value=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    ):
        assert grpo_trainer._check_group_episodes(sample_group_episodes) is True


@pytest.mark.parametrize(
    'episodes, expected_result',
    [
        ([], False),  # Empty list
        ([Mock()] * 3, False),  # List too short
    ],
)
def test_check_group_episodes_invalid_size(
    grpo_trainer: ExtendedGRPOTrainer, episodes: List, expected_result: bool
):
    """Tests the group check fails for invalid list sizes."""
    assert grpo_trainer._check_group_episodes(episodes) is expected_result


def test_check_group_episodes_low_std(
    grpo_trainer: ExtendedGRPOTrainer, sample_group_episodes: List[EpisodeData]
):
    """Tests the group check fails for rewards with low standard deviation."""
    # Mock transform to return rewards with very low std dev
    low_std_rewards = torch.tensor([1.0, 1.0, 1.0, 1.01])  # Low std
    grpo_trainer.group_reward_std_threshold = (
        low_std_rewards.std(unbiased=False) + 0.01
    )  # Ensure threshold is higher

    with patch.object(
        grpo_trainer, 'transform_batch_rewards', return_value=low_std_rewards
    ):
        assert (
            grpo_trainer._check_group_episodes(sample_group_episodes) is False
        )
        grpo_trainer.logger.debug.assert_called_once()
        grpo_trainer.logger.log_scalar.assert_called_with(
            'other/skipped_sample_count', len(low_std_rewards)
        )
