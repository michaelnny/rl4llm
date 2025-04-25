from copy import deepcopy
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from deepspeed import DeepSpeedEngine

from rl4llm.constants import TRAIN_PHASE
from rl4llm.core.base_env import EpisodeData, EpisodeMetadata
from rl4llm.core.base_trainer import BaseRLConfig, BaseRLTrainer


@pytest.fixture
def mock_config():
    """Provides a minimal BaseRLConfig for testing."""
    config = MagicMock(spec=BaseRLConfig)
    config.train_rollout_size = 8
    config.kl_loss_coef = 0.0
    config.max_steps = 1
    config.eval_interval = 1
    config.eval_rollout_size = 4
    config.checkpoint_interval = 1
    config.sync_reference_interval = 1
    return config


@pytest.fixture
def mock_policy_engine():
    """Mocks DeepSpeed policy engine."""
    engine = MagicMock()
    engine.zero_optimization_stage.return_value = 2
    engine.bfloat16_enabled.return_value = False
    return engine


@pytest.fixture
def mock_dist_ops():
    """Mocks a simple DistributedOps."""
    dist = MagicMock()
    dist.world_size = 1
    dist.is_master = True
    dist.device = 'cpu'
    return dist


@pytest.fixture
def mock_logger():
    """Mocks LoggingManager."""
    logger = MagicMock()
    logger.timer.return_value.__enter__.return_value = None
    logger.timer.return_value.__exit__.return_value = None
    return logger


@pytest.fixture
def sample_episode_data() -> EpisodeData:
    """Provides a sample EpisodeData instance."""
    prompt = 'Once upon a time'
    completion = ' there was a dragon.'

    prompt_tokens = torch.tensor([1, 2, 3])
    completion_tokens = torch.tensor([4, 5, 6])

    full_seq = torch.concat([prompt_tokens, completion_tokens])
    states = full_seq[:-1]
    actions = full_seq[1:]
    loss_mask = torch.zeros_like(actions, dtype=torch.bool)
    loss_mask[len(prompt_tokens) - 1 :] = True

    meta = EpisodeMetadata(
        prompt=prompt,
        completion=completion,
        prompt_length=len(prompt_tokens),
        completion_length=len(completion_tokens),
        reward_dict={'reward1': 1.5},
        ground_truth='123',
    )

    return EpisodeData(
        states=states,
        actions=actions,
        loss_mask=loss_mask,
        terminal_reward=1.5,
        metadata=meta,
    )


@pytest.fixture
def sample_group_episodes(sample_episode_data) -> List[EpisodeData]:
    """Provides a list of sample EpisodeData instances for a group."""
    # Create slightly different rewards for testing normalization/std check
    ep1 = sample_episode_data.model_copy(deep=True)
    ep1.terminal_reward = 1.5
    ep2 = sample_episode_data.model_copy(deep=True)
    ep2.terminal_reward = 2.0
    ep3 = sample_episode_data.model_copy(deep=True)
    ep3.terminal_reward = 1.0
    ep4 = sample_episode_data.model_copy(deep=True)
    ep4.terminal_reward = 2.5
    return [ep1, ep2, ep3, ep4]


@pytest.fixture
def mock_reward_transform_fn() -> MagicMock:
    """Provides a mock reward transformation function."""
    fn = MagicMock()
    # Simple sum aggregation for testing
    fn.side_effect = lambda rewards_dict: sum(rewards_dict.values())
    return fn


@pytest.fixture
def trainer_base(
    mock_config, mock_policy_engine, mock_dist_ops, mock_logger, mocker
):
    """Provides a dummy BaseRLTrainer subclass for testing."""

    class DummyTrainer(BaseRLTrainer):
        def initialize_trainer(self):
            pass

        def generate_experience(self):
            return ['exp']

        def build_train_loader(self, experience):
            return ['batch']

        def evaluate_step(self):
            pass

        def train_step(self, train_dataloader):
            pass

        def can_offload_state(self, model):
            return getattr(model, 'can_offload', False)

        def is_cohost_inference_engine(self):
            return False

        def save_checkpoint(self, tag):
            pass

        def clean_up(self):
            pass

    mocker.patch(
        'rl4llm.core.distributed.DistributedOps.get_instance',
        return_value=mock_dist_ops,
        autospec=True,
    )
    # mocker.patch(
    #     "rl4llm.logging.logging_manager.LoggingManager",
    #     return_value=mock_logger,
    #     autospec=True
    # )
    mocker.patch(
        'rl4llm.core.base_trainer.LoggingManager',
        return_value=mock_logger,
        autospec=True,
    )

    trainer = DummyTrainer(
        config=mock_config,
        tokenizer=MagicMock(),
        policy_engine=mock_policy_engine,
        log_config={'output_dir': '/tmp/test_rl_trainer'},
        train_env=MagicMock(),
        eval_env=MagicMock(),
        inference_client=MagicMock(),
    )

    return trainer


@pytest.fixture
def mock_model():
    """Provides a mock PyTorch model with configurable attributes."""
    model = MagicMock(spec=torch.nn.Module)
    model.to.return_value = model
    model.eval = MagicMock()
    model.train = MagicMock()
    model.can_offload = False
    return model


def test_log_batch_episodes_invalid_phase(trainer_base):
    """Raises error on invalid log phase."""
    with pytest.raises(ValueError):
        trainer_base.log_batch_episodes('invalid_phase', [], 0)


def test_log_batch_episodes_valid(trainer_base):
    """Logs sample and scalar for valid training episode."""

    meta = EpisodeMetadata(
        prompt='a',
        completion='b',
        prompt_length=1,
        completion_length=2,
        reward_dict={'reward1': 1.5},
        ground_truth='123',
    )

    episode = EpisodeData(
        states=torch.tensor([1]),
        actions=torch.tensor([1]),
        loss_mask=torch.tensor([1]),
        terminal_reward=torch.tensor([1]),
        metadata=meta,
    )
    trainer_base.log_batch_episodes(TRAIN_PHASE, [episode], 1)
    trainer_base.logger.log_sample.assert_called()
    trainer_base.logger.log_scalar.assert_called()


def test_configure_model(mock_model, trainer_base):
    """Test _configure_model for device movement, mode setting, and state management."""
    # Test with no model
    trainer_base._configure_model(
        None, 'cpu', state_action='offload', mode='eval'
    )
    mock_model.to.assert_not_called()

    # Test device movement and eval mode
    trainer_base._configure_model(mock_model, 'cuda', mode='eval')
    mock_model.to.assert_called_once_with('cuda')
    mock_model.eval.assert_called_once()
    mock_model.train.assert_not_called()

    # Reset mocks
    mock_model.to.reset_mock()
    mock_model.eval.reset_mock()

    # Test training mode
    trainer_base._configure_model(mock_model, 'cpu', mode='train')
    mock_model.to.assert_called_once_with('cpu')
    mock_model.train.assert_called_once()
    mock_model.eval.assert_not_called()

    # Test state offloading
    mock_model.can_offload = True
    mock_model.offload_states = MagicMock()
    trainer_base._configure_model(mock_model, 'cuda', state_action='offload')
    mock_model.offload_states.assert_called_once()

    # Test state reloading
    mock_model.reload_states = MagicMock()
    trainer_base._configure_model(mock_model, 'cuda', state_action='reload')
    mock_model.reload_states.assert_called_once()


def test_release_inference_memory(trainer_base):
    """Test _release_inference_memory for co-hosting and non-co-hosting scenarios."""
    # Non-co-hosting: no memory release
    trainer_base.is_cohost_inference_engine = MagicMock(return_value=False)
    trainer_base._release_inference_memory()
    trainer_base.inference_client.release_memory.assert_not_called()
    assert not trainer_base.called_release_inference_memory

    # Co-hosting: memory release called
    trainer_base.is_cohost_inference_engine = MagicMock(return_value=True)
    trainer_base._release_inference_memory()
    trainer_base.inference_client.release_memory.assert_called_once()
    assert trainer_base.called_release_inference_memory is True

    # Co-hosting with error
    trainer_base.called_release_inference_memory = None
    trainer_base.inference_client.release_memory.side_effect = Exception(
        'Memory error'
    )
    with pytest.raises(
        RuntimeError,
        match='Failed to release inference engine memory, error: Memory error',
    ):
        trainer_base._release_inference_memory()


def test_prepare_for_generation(trainer_base, mock_model):
    """Test _prepare_for_generation for model configuration and cleanup."""
    trainer_base.reference_model = mock_model
    trainer_base.value_engine = mock_model
    trainer_base.policy_engine = mock_model
    trainer_base.clean_up = MagicMock()
    trainer_base.device = 'cuda'  # Ensure self.device is 'cuda'

    # Non-co-hosting scenario
    trainer_base.is_cohost_inference_engine = MagicMock(return_value=False)
    trainer_base._prepare_for_generation()
    calls = [
        mock_model.to.call_args_list[0][0][0],  # reference_model to cpu
        mock_model.to.call_args_list[1][0][0],  # policy_engine to cuda
        mock_model.to.call_args_list[2][0][0],  # value_engine to cpu
    ]
    assert calls == ['cpu', 'cuda', 'cpu']
    assert mock_model.eval.call_count == 2  # policy_engine and reference_model
    trainer_base.clean_up.assert_called_once()

    # Co-hosting scenario
    mock_model.to.reset_mock()
    mock_model.eval.reset_mock()
    trainer_base.clean_up.reset_mock()
    trainer_base.is_cohost_inference_engine = MagicMock(return_value=True)
    trainer_base._prepare_for_generation()
    calls = [
        mock_model.to.call_args_list[0][0][0],  # reference_model to cpu
        mock_model.to.call_args_list[1][0][0],  # policy_engine to cpu
        mock_model.to.call_args_list[2][0][0],  # value_engine to cpu
    ]
    assert calls == ['cpu', 'cpu', 'cpu']
    assert mock_model.eval.call_count == 1  # only reference_model
    trainer_base.clean_up.assert_called_once()


def test_prepare_for_pre_processing(trainer_base, mock_model):
    """Test _prepare_for_pre_processing for model configuration and memory release."""
    trainer_base.reference_model = mock_model
    trainer_base.value_engine = mock_model
    trainer_base.policy_engine = mock_model
    trainer_base.clean_up = MagicMock()
    trainer_base.is_cohost_inference_engine = MagicMock(return_value=True)
    trainer_base.device = 'cuda'  # Ensure self.device is 'cuda'

    trainer_base._prepare_for_pre_processing()
    calls = [
        mock_model.to.call_args_list[0][0][0],  # policy_engine to cuda
        mock_model.to.call_args_list[1][0][0],  # value_engine to cuda
        mock_model.to.call_args_list[2][0][0],  # reference_model to cuda
    ]
    assert calls == ['cuda', 'cuda', 'cuda']
    assert mock_model.eval.call_count == 3  # all models
    trainer_base.inference_client.release_memory.assert_called_once()
    trainer_base.clean_up.assert_called_once()


def test_prepare_for_training(trainer_base, mock_model):
    """Test _prepare_for_training for model configuration and memory release."""
    trainer_base.reference_model = mock_model
    trainer_base.value_engine = mock_model
    trainer_base.policy_engine = mock_model
    trainer_base.clean_up = MagicMock()
    trainer_base.is_cohost_inference_engine = MagicMock(return_value=True)
    trainer_base.device = 'cuda'  # Ensure self.device is 'cuda'

    trainer_base._prepare_for_training()
    calls = [
        mock_model.to.call_args_list[0][0][0],  # reference_model to cpu
        mock_model.to.call_args_list[1][0][0],  # policy_engine to cuda
        mock_model.to.call_args_list[2][0][0],  # value_engine to cuda
    ]
    assert calls == ['cpu', 'cuda', 'cuda']
    assert mock_model.train.call_count == 2  # policy_engine and value_engine
    assert mock_model.eval.call_count == 0
    trainer_base.inference_client.release_memory.assert_called_once()
    trainer_base.clean_up.assert_called_once()


def test_sync_reference_model_no_ref_model(trainer_base):
    """Tests that sync does nothing if reference_model is None."""
    trainer_base.reference_model = None
    trainer_base.sync_reference_model()

    assert trainer_base.ref_update_count == 0


def test_sync_reference_model_standard_pytorch(trainer_base):
    """Tests syncing to a standard PyTorch reference model."""
    mock_ref_model = MagicMock()
    # Configure .to() to return the mock itself
    mock_ref_model.to.return_value = mock_ref_model

    trainer_base.reference_model = mock_ref_model
    initial_count = trainer_base.ref_update_count

    trainer_base.sync_reference_model()

    # The policy_engine is likely a DeepSpeedEngine in the fixture,
    # so the unwrapped model is policy_engine.module
    expected_state_dict = trainer_base.policy_engine.module.state_dict()

    # Assert call happened on the original mock
    mock_ref_model.load_state_dict.assert_called_once_with(expected_state_dict)
    # Assert .to() was called (at least twice: to device, then to cpu)
    mock_ref_model.to.assert_any_call(trainer_base.device)
    mock_ref_model.to.assert_any_call('cpu')
    assert trainer_base.ref_update_count == initial_count + 1
    trainer_base.dist_ops.barrier.assert_called()


def test_sync_reference_model_deepspeed_no_zero3(trainer_base, mocker):
    """Tests syncing to a DeepSpeedEngine reference model without Zero-3."""
    # Use spec for isinstance check to potentially work without patching isinstance
    mock_ref_model = mocker.MagicMock(spec=DeepSpeedEngine)
    mock_ref_model.module = MagicMock()
    # Configure .to() to return the mock itself
    mock_ref_model.to.return_value = mock_ref_model

    trainer_base.reference_model = mock_ref_model

    # Mock is_zero3_enabled to return False
    mocker.patch.object(trainer_base, 'is_zero3_enabled', return_value=False)

    initial_count = trainer_base.ref_update_count

    trainer_base.sync_reference_model()

    expected_state_dict = trainer_base.policy_engine.module.state_dict()

    # Assert call happened on the original mock's module
    mock_ref_model.module.load_state_dict.assert_called_once_with(
        expected_state_dict
    )
    # Assert .to() was called
    mock_ref_model.to.assert_any_call(trainer_base.device)
    mock_ref_model.to.assert_any_call('cpu')
    assert trainer_base.ref_update_count == initial_count + 1
    trainer_base.dist_ops.barrier.assert_called()


@patch('deepspeed.zero.GatheredParameters', autospec=True)
def test_sync_reference_model_deepspeed_zero3_master(
    mock_gathered_params, trainer_base, mocker
):
    """Tests syncing to a DeepSpeedEngine reference model with Zero-3 on master."""
    mock_ref_model = mocker.MagicMock(spec=DeepSpeedEngine)
    mock_ref_model.module = MagicMock()
    mock_ref_model.parameters = MagicMock(
        return_value=[]
    )  # Needs to return an iterable
    # Configure .to() to return the mock itself
    mock_ref_model.to.return_value = mock_ref_model

    trainer_base.reference_model = mock_ref_model
    trainer_base.dist_ops.is_master = True

    # Mock is_zero3_enabled to return True
    mocker.patch.object(trainer_base, 'is_zero3_enabled', return_value=True)
    # Again, avoid patching isinstance if spec works

    initial_count = trainer_base.ref_update_count

    trainer_base.sync_reference_model()

    expected_state_dict = trainer_base.policy_engine.module.state_dict()

    # Assert GatheredParameters was used
    mock_gathered_params.assert_called_with(mock_ref_model.parameters())

    # Assert call happened on the original mock's module
    mock_ref_model.module.load_state_dict.assert_called_once_with(
        expected_state_dict
    )
    # Assert .to() was called
    mock_ref_model.to.assert_any_call(trainer_base.device)
    mock_ref_model.to.assert_any_call('cpu')
    assert trainer_base.ref_update_count == initial_count + 1
    trainer_base.dist_ops.barrier.assert_called()


def test_sync_policy_model_inference_disabled(trainer_base, mocker):
    """Tests that sync does nothing if inference engine is disabled."""
    mocker.patch(
        'rl4llm.core.base_trainer.BaseRLTrainer.is_inference_engine_enabled',
        return_value=False,
    )
    trainer_base.save_weights_hf_pretrained = MagicMock()

    trainer_base.sync_policy_model()

    trainer_base.save_weights_hf_pretrained.assert_not_called()
    trainer_base.inference_client.update_weights_from_file.assert_not_called()


@patch('tempfile.TemporaryDirectory')
def test_sync_policy_model_success_master(mock_tempdir, trainer_base, mocker):
    """Tests successful policy sync on the master rank."""
    mocker.patch(
        'rl4llm.core.base_trainer.BaseRLTrainer.is_inference_engine_enabled',
        return_value=True,
    )
    mock_tempdir.return_value.__enter__.return_value = (
        '/fake/temp/path'  # Mock the temp path
    )
    trainer_base.dist_ops.is_master = True
    trainer_base.save_weights_hf_pretrained = MagicMock()

    trainer_base.sync_policy_model()

    trainer_base.save_weights_hf_pretrained.assert_called_once_with(
        trainer_base.policy_engine, '/fake/temp/path'
    )
    trainer_base.inference_client.resume_memory.assert_called_once()
    trainer_base.inference_client.update_weights_from_file.assert_called_once_with(
        model_path='/fake/temp/path'
    )
    trainer_base.dist_ops.barrier.assert_called()


@patch('tempfile.TemporaryDirectory')
def test_sync_policy_model_success_non_master(
    mock_tempdir, trainer_base, mocker
):
    """Tests successful policy sync on a non-master rank."""
    mocker.patch(
        'rl4llm.core.base_trainer.BaseRLTrainer.is_inference_engine_enabled',
        return_value=True,
    )
    mock_tempdir.return_value.__enter__.return_value = '/fake/temp/path'
    trainer_base.is_inference_engine_enabled.return_value = True
    trainer_base.dist_ops.is_master = False  # Set to non-master
    trainer_base.save_weights_hf_pretrained = MagicMock()

    trainer_base.sync_policy_model()

    trainer_base.save_weights_hf_pretrained.assert_called_once_with(
        trainer_base.policy_engine, '/fake/temp/path'
    )
    # Inference client methods should not be called on non-master
    trainer_base.inference_client.resume_memory.assert_not_called()
    trainer_base.inference_client.update_weights_from_file.assert_not_called()
    trainer_base.dist_ops.barrier.assert_called()


# def test_transform_batch_rewards_single_reward(
#     trainer_base, sample_group_episodes
# ):
#     """Tests reward transformation when only one reward function is present."""
#     # Ensure only one reward function is mocked
#     trainer_base.train_env.reward_functions = {'reward1': Mock()}
#     trainer_base.reward_transform_fn = None  # Should not be needed

#     rewards = trainer_base.transform_batch_rewards(sample_group_episodes)
#     expected = torch.tensor(
#         [1.5, 2.0, 1.0, 2.5], dtype=trainer_base.torch_dtype
#     )
#     assert torch.equal(rewards, expected)


# def test_transform_batch_rewards_multiple_rewards(
#     trainer_base,
#     sample_group_episodes,
#     mock_reward_transform_fn,
# ):
#     """Tests reward transformation with multiple reward functions using the transform function."""
#     # Add a second reward
#     for ep in sample_group_episodes:
#         ep.reward_dict['reward2'] = 0.5
#     trainer_base.train_env.reward_functions = {
#         'reward1': Mock(),
#         'reward2': Mock(),
#     }
#     trainer_base.reward_transform_fn = mock_reward_transform_fn

#     rewards = trainer_base.transform_batch_rewards(sample_group_episodes)

#     # Check that the transform function was called correctly
#     mock_reward_transform_fn.assert_called_once()
#     call_args = mock_reward_transform_fn.call_args[0][0]
#     assert 'reward1' in call_args
#     assert 'reward2' in call_args
#     assert torch.equal(
#         call_args['reward1'],
#         torch.tensor([1.5, 2.0, 1.0, 2.5], dtype=trainer_base.torch_dtype),
#     )
#     assert torch.equal(
#         call_args['reward2'],
#         torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=trainer_base.torch_dtype),
#     )

#     # Check the output (based on the mock's side effect: sum)
#     expected = torch.tensor(
#         [2.0, 2.5, 1.5, 3.0], dtype=trainer_base.torch_dtype
#     )
#     assert torch.equal(rewards, expected)


# def test_transform_batch_rewards_empty_list(trainer_base):
#     """Tests that transforming rewards on an empty list raises ValueError."""
#     with pytest.raises(ValueError, match='Episodes list cannot be empty'):
#         trainer_base.transform_batch_rewards([])
