# flake8: noqa

import math
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from transformers import PreTrainedTokenizer

# Objects under test
from rl4llm.envs import EpisodeData
from rl4llm.trainers.grpo_trainer import (
    GRPOConfig,
    GRPOTrainer,
    RewardTransform,
    TransitionData,
)

# --- Fixtures ---


@pytest.fixture
def grpo_config() -> GRPOConfig:
    """Provides a default GRPOConfig instance for testing."""
    return GRPOConfig(
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
def mock_dist_ops() -> MagicMock:
    """Provides a mock DistributedOps."""
    manager = MagicMock()
    manager.world_size = 1
    manager.is_master.return_value = True
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
    grpo_config: GRPOConfig,
    mock_tokenizer: MagicMock,
    mock_policy_engine: MagicMock,
    mock_dist_ops: MagicMock,
    mock_logger: MagicMock,
    mock_train_env: MagicMock,
    mock_ref_model: MagicMock,
    mock_reward_transform_fn: MagicMock,
    mocker,
) -> GRPOTrainer:
    """Provides a GRPOTrainer instance with mocked dependencies."""
    # Mock the reward transform function only if needed (multiple rewards)
    reward_fn = None
    if len(mock_train_env.reward_functions) > 1:
        reward_fn = mock_reward_transform_fn

    mocker.patch(
        'rl4llm.core.distributed.DistributedOps.get_instance',
        return_value=mock_dist_ops,
        autospec=True,
    )
    # mocker.patch(
    #     "rl4llm.logging.logging_manager.LoggingManager",
    #     return_value=dummy_logger,
    #     autospec=True
    # )
    mocker.patch(
        'rl4llm.core.base_trainer.LoggingManager',
        return_value=mock_logger,
        autospec=True,
    )

    trainer = GRPOTrainer(
        config=grpo_config,
        tokenizer=mock_tokenizer,  # Pass the mock tokenizer
        policy_engine=mock_policy_engine,
        log_config={'output_dir': '/tmp/test_rl_trainer'},
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


@pytest.mark.parametrize(
    'rewards, zero_mean_only, expected_mean_approx, expected_std_approx',
    [
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            False,
            0.0,
            1.11803,
        ),  # unbiased=False std
        (
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
            False,
            0.0,
            0.0,
        ),  # Std is 0, handled by eps
        (
            torch.tensor([5.0, 6.0, 7.0, 8.0]),
            True,
            0.0,
            1.11803,
        ),  # Only mean subtracted, mean becomes 0
    ],
)
def test_normalize_group_rewards(
    grpo_trainer: GRPOTrainer,
    rewards: torch.Tensor,
    zero_mean_only: bool,
    expected_mean_approx: float,
    expected_std_approx: float,
):
    """Tests the normalization of group rewards."""
    normalized_rewards = grpo_trainer._normalize_group_rewards(
        rewards, zero_mean_only=zero_mean_only
    )
    assert normalized_rewards.mean().item() == pytest.approx(
        expected_mean_approx, abs=1e-5
    )
    if zero_mean_only:
        # Check if only mean was subtracted
        assert torch.allclose(normalized_rewards, rewards - rewards.mean())
        # Check std is preserved (approx)
        assert normalized_rewards.std(unbiased=False).item() == pytest.approx(
            expected_std_approx, abs=1e-5
        )
    else:
        # Check std is approx 1 (unless original std was 0)
        if (
            expected_std_approx > 1e-8
        ):  # Avoid checking std=1 for zero-std input
            assert normalized_rewards.std(
                unbiased=False
            ).item() == pytest.approx(1.0, abs=1e-5)
        else:
            assert normalized_rewards.std(
                unbiased=False
            ).item() == pytest.approx(0.0, abs=1e-5)


def test_normalize_group_rewards_raises_error_for_small_group(
    grpo_trainer: GRPOTrainer,
):
    """Tests that normalization fails if the group size is less than 4."""
    rewards = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(
        ValueError, match='Number of group rewards must be greater than 4'
    ):
        grpo_trainer._normalize_group_rewards(rewards)


def test_train_collate_fn(
    grpo_trainer: GRPOTrainer, sample_transition_data: TransitionData
):
    """Tests the collate function for creating training batches."""
    # Create slightly different length samples for padding test
    sample2 = sample_transition_data.model_copy(deep=True)
    sample2.states = sample2.states[:-1]
    sample2.actions = sample2.actions[:-1]
    sample2.loss_mask = sample2.loss_mask[:-1]
    sample2.pi_logprobs = sample2.pi_logprobs[:-1]
    sample2.ref_logprobs = sample2.ref_logprobs[:-1]
    sample2.advantages = sample2.advantages[:-1]

    batch_list = [sample_transition_data] * 2 + [
        sample2
    ] * 2  # Batch of 4, two lengths
    collated_batch = grpo_trainer._train_collate_fn(batch_list)

    assert isinstance(collated_batch, TransitionData)
    expected_batch_size = 4
    expected_max_seq_len = 10  # From sample_transition_data fixture
    expected_shape = (expected_batch_size, expected_max_seq_len)
    pad_token_id = (
        grpo_trainer.tokenizer.pad_token_id
    )  # Use pad_token_id from mock

    assert collated_batch.states.shape == expected_shape
    assert collated_batch.states.dtype == torch.long
    # Check padding was applied correctly to shorter sequences
    assert torch.all(collated_batch.states[2:, -1] == pad_token_id)
    assert torch.all(
        collated_batch.states[:2, -1] != pad_token_id
    )  # Assuming original didn't end with pad

    assert collated_batch.actions.shape == expected_shape
    assert collated_batch.actions.dtype == torch.long
    assert torch.all(collated_batch.actions[2:, -1] == pad_token_id)

    assert collated_batch.loss_mask.shape == expected_shape
    assert collated_batch.loss_mask.dtype == torch.bool
    assert torch.all(
        collated_batch.loss_mask[2:, -1] == False
    )  # Padding mask should be False

    assert collated_batch.advantages.shape == expected_shape
    assert collated_batch.advantages.dtype == grpo_trainer.torch_dtype
    assert torch.all(
        collated_batch.advantages[2:, -1] == 0.0
    )  # Padding advantages should be 0.0

    assert collated_batch.pi_logprobs.shape == expected_shape
    assert collated_batch.pi_logprobs.dtype == grpo_trainer.torch_dtype
    assert torch.all(
        collated_batch.pi_logprobs[2:, -1] == 0.0
    )  # Padding logprobs should be 0.0

    assert collated_batch.ref_logprobs.shape == expected_shape
    assert collated_batch.ref_logprobs.dtype == grpo_trainer.torch_dtype
    assert torch.all(
        collated_batch.ref_logprobs[2:, -1] == 0.0
    )  # Padding logprobs should be 0.0


# --- compute_loss Tests ---


@pytest.fixture
def loss_inputs(
    mock_tokenizer: MagicMock,
) -> Dict[str, torch.Tensor]:  # Depends on mock_tokenizer
    """Provides sample inputs for the compute_loss function."""
    batch_size = 2
    seq_len = 5
    vocab_size = mock_tokenizer.vocab_size  # Use vocab_size from mock

    # Ensure actions are within vocab size
    actions = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.long
    )
    # Make loss mask have some False values
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    loss_mask[0, :2] = False  # Mask first two tokens of first sequence
    loss_mask[1, :1] = False  # Mask first token of second sequence

    # Ensure logprobs correspond to actions
    pi_logits = (
        torch.randn(batch_size, seq_len, vocab_size) * 0.1
    )  # Smaller logits for stability
    pi_logprobs = torch.gather(
        pi_logits.log_softmax(-1), -1, actions.unsqueeze(-1)
    ).squeeze(-1)

    ref_logprobs = torch.randn(batch_size, seq_len) * 0.1
    advantages = torch.randn(batch_size, seq_len)

    # Create a dummy batch object
    experience_batch = TransitionData(
        states=torch.zeros_like(actions),  # Not used directly in compute_loss
        actions=actions,
        loss_mask=loss_mask,
        pi_logprobs=pi_logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
    )

    return {
        'pi_logits': pi_logits,
        'experience_batch': experience_batch,
    }


def test_compute_loss_basic(
    grpo_trainer: GRPOTrainer, loss_inputs: Dict[str, torch.Tensor]
):
    """Tests the basic computation of the GRPO loss."""
    loss = grpo_trainer.compute_loss(**loss_inputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # Scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Check if logs were called - verify keys exist and values are floats
    log_calls = {
        c.args[0]: c.args[1]
        for c in grpo_trainer.logger.log_scalar.call_args_list
    }
    assert 'train/pg_loss' in log_calls
    assert isinstance(log_calls['train/pg_loss'], float)

    assert 'train/entropy_loss' in log_calls
    assert isinstance(log_calls['train/entropy_loss'], float)

    assert 'policy/entropy' in log_calls
    assert isinstance(log_calls['policy/entropy'], float)

    assert 'policy/approxkl' in log_calls
    assert isinstance(log_calls['policy/approxkl'], float)

    assert 'policy/clipfrac' in log_calls
    assert isinstance(log_calls['policy/clipfrac'], float)

    # KL loss is calculated because config has kl_loss_coef > 0
    assert 'train/kl_loss' in log_calls
    assert isinstance(log_calls['train/kl_loss'], float)
    assert 'objective/kl' in log_calls
    assert isinstance(log_calls['objective/kl'], float)

    # Check if the final loss roughly matches the sum of logged components
    expected_loss_approx = (
        log_calls['train/pg_loss']
        + log_calls['train/kl_loss']
        + log_calls['train/entropy_loss']
    )
    assert loss.item() == pytest.approx(expected_loss_approx)


def test_compute_loss_no_kl(
    grpo_trainer: GRPOTrainer, loss_inputs: Dict[str, torch.Tensor]
):
    """Tests loss computation when KL coefficient is zero."""
    grpo_trainer.config.kl_loss_coef = 0.0
    loss = grpo_trainer.compute_loss(**loss_inputs)
    assert isinstance(loss, torch.Tensor)

    # Check that kl_loss related logs were not called
    log_calls = {
        c.args[0]: c.args[1]
        for c in grpo_trainer.logger.log_scalar.call_args_list
    }
    assert 'train/kl_loss' not in log_calls
    assert 'objective/kl' not in log_calls

    # Check if the final loss roughly matches the sum of remaining logged components
    expected_loss_approx = (
        log_calls['train/pg_loss'] + log_calls['train/entropy_loss']
    )
    assert loss.item() == pytest.approx(expected_loss_approx)


def test_compute_loss_no_advantage_norm(
    grpo_trainer: GRPOTrainer, loss_inputs: Dict[str, torch.Tensor]
):
    """Tests loss computation without advantage normalization."""
    grpo_trainer.config.normalize_advantages = False
    # Mock masked_whiten to check it's not called
    with patch.object(
        grpo_trainer, 'masked_whiten', wraps=grpo_trainer.masked_whiten
    ) as mock_whiten:
        loss = grpo_trainer.compute_loss(**loss_inputs)
        assert isinstance(loss, torch.Tensor)
        mock_whiten.assert_not_called()


def test_compute_loss_with_advantage_norm(
    grpo_trainer: GRPOTrainer, loss_inputs: Dict[str, torch.Tensor]
):
    """Tests loss computation with advantage normalization."""
    grpo_trainer.config.normalize_advantages = True
    # Mock masked_whiten to check it's called
    with patch.object(
        grpo_trainer, 'masked_whiten', wraps=grpo_trainer.masked_whiten
    ) as mock_whiten:
        loss = grpo_trainer.compute_loss(**loss_inputs)
        assert isinstance(loss, torch.Tensor)
        mock_whiten.assert_called_once()


# --- Integration-like Tests (Simplified) ---


@patch('rl4llm.trainers.grpo_trainer.DataLoader')  # Mock DataLoader
def test_build_train_loader(
    mock_dataloader_cls: MagicMock,
    grpo_trainer: GRPOTrainer,
    sample_group_episodes: List[EpisodeData],
    mock_tokenizer: MagicMock,  # Use mock
    mock_policy_engine: MagicMock,
    mock_ref_model: MagicMock,
):
    """Tests the creation of a DataLoader from experience."""
    # Mock the conversion process to return dummy TransitionData
    dummy_transition = TransitionData(
        states=torch.tensor([1, 2]),
        actions=torch.tensor([2, 3]),
        loss_mask=torch.tensor([True, True]),
        pi_logprobs=torch.tensor([-0.1, -0.2]),
        ref_logprobs=torch.tensor([-0.15, -0.25]),
        advantages=torch.tensor([0.5, 0.5]),
    )
    # Mock models returning simple logits for conversion step
    # Determine max length needed for the mock forward pass in conversion
    max_len = max(
        ep.prompt_length + ep.completion_length for ep in sample_group_episodes
    )
    batch_size = len(sample_group_episodes)
    vocab_size = mock_tokenizer.vocab_size  # Use mock vocab size
    mock_policy_engine.forward.return_value = Mock(
        logits=torch.randn(batch_size, max_len - 1, vocab_size)
    )
    mock_ref_model.forward.return_value = Mock(
        logits=torch.randn(batch_size, max_len - 1, vocab_size)
    )

    with patch.object(
        grpo_trainer,
        '_convert_group_episodes_to_transitions',
        return_value=[dummy_transition] * len(sample_group_episodes),
    ) as mock_convert:
        # Provide a list containing one group
        experience = [sample_group_episodes]
        dataloader = grpo_trainer.build_train_loader(experience)

        mock_convert.assert_called_once_with(sample_group_episodes)
        mock_dataloader_cls.assert_called_once()

        # Check args passed to DataLoader
        dataloader_args, dataloader_kwargs = mock_dataloader_cls.call_args
        assert len(dataloader_args[0]) == len(
            sample_group_episodes
        )  # Number of samples
        assert (
            dataloader_kwargs['batch_size']
            == grpo_trainer.config.train_micro_batch_size
        )
        assert dataloader_kwargs['shuffle'] is True
        assert dataloader_kwargs['collate_fn'] == grpo_trainer._train_collate_fn
        assert isinstance(
            dataloader, MagicMock
        )  # It returns the mocked instance


def test_build_train_loader_empty_experience(grpo_trainer: GRPOTrainer):
    """Tests that building a loader from empty experience raises ValueError."""
    # Mock conversion to return empty list
    with patch.object(
        grpo_trainer, '_convert_group_episodes_to_transitions', return_value=[]
    ):
        # Mock _check_group_episodes to allow processing empty groups initially
        with pytest.raises(ValueError, match='No samples for training'):
            grpo_trainer.build_train_loader([[]])  # Empty group list


@patch(
    'rl4llm.trainers.grpo_trainer.pad_sequence',
    side_effect=torch.nn.utils.rnn.pad_sequence,
)  # Use real pad_sequence
def test_convert_group_episodes_to_transitions(
    mock_pad: MagicMock,
    grpo_trainer: GRPOTrainer,
    sample_group_episodes: List[EpisodeData],
    mock_tokenizer: MagicMock,  # Use mock
    mock_policy_engine: MagicMock,
    mock_ref_model: MagicMock,
):
    """Tests the conversion of raw episodes to TransitionData."""
    # Make sequence lengths slightly different for padding test
    # Use mock tokenizer to encode
    sample_group_episodes[1].completion_tokens = mock_tokenizer.encode(
        ' was nice.', return_tensors='pt'
    )
    sample_group_episodes[1].completion_length = len(
        sample_group_episodes[1].completion_tokens
    )

    # Mock model outputs
    max_len = max(
        ep.prompt_length + ep.completion_length for ep in sample_group_episodes
    )
    batch_size = len(sample_group_episodes)
    vocab_size = mock_tokenizer.vocab_size  # Use mock vocab size
    # Logits need shape [batch_size, max_seq_len - 1, vocab_size] because states are seq[:-1]
    mock_policy_engine.forward.return_value = Mock(
        logits=torch.randn(batch_size, max_len - 1, vocab_size)
    )
    mock_ref_model.forward.return_value = Mock(
        logits=torch.randn(batch_size, max_len - 1, vocab_size)
    )

    # Mock reward normalization to avoid dependency on its correctness here
    with patch.object(
        grpo_trainer,
        '_normalize_group_rewards',
        side_effect=lambda x, y, **kwargs: x,
    ):
        transitions = grpo_trainer._convert_group_episodes_to_transitions(
            sample_group_episodes
        )

    assert isinstance(transitions, list)
    assert len(transitions) == len(sample_group_episodes)
    mock_policy_engine.forward.assert_called_once()
    mock_ref_model.forward.assert_called_once()  # Called because kl_loss_coef > 0

    for i, trans in enumerate(transitions):
        ep = sample_group_episodes[i]
        expected_seq_len = (
            ep.prompt_length + ep.completion_length - 1
        )  # states/actions length
        assert isinstance(trans, TransitionData)
        assert trans.states.shape == (expected_seq_len,)
        assert trans.actions.shape == (expected_seq_len,)
        assert trans.loss_mask.shape == (expected_seq_len,)
        assert trans.pi_logprobs.shape == (expected_seq_len,)
        assert trans.ref_logprobs.shape == (expected_seq_len,)
        assert trans.advantages.shape == (
            expected_seq_len,
        )  # Advantages are broadcasted rewards here

        # Check loss mask correctness
        assert torch.all(trans.loss_mask[: ep.prompt_length - 1] == False)
        assert torch.all(trans.loss_mask[ep.prompt_length - 1 :] == True)
        assert trans.loss_mask.sum().item() == ep.completion_length

        # Check advantages are non-zero only where loss_mask is True (using original reward)
        original_reward = ep.reward_dict['reward1']
        assert torch.all(trans.advantages[trans.loss_mask] == original_reward)
        assert torch.all(trans.advantages[~trans.loss_mask] == 0)


def test_convert_group_episodes_to_transitions_small_group(
    grpo_trainer: GRPOTrainer, sample_group_episodes: List[EpisodeData]
):
    """Tests that conversion fails if group size is less than 4."""
    small_group = sample_group_episodes[:3]
    with pytest.raises(
        ValueError, match='Expect group episodes to be greater than 4'
    ):
        grpo_trainer._convert_group_episodes_to_transitions(small_group)
