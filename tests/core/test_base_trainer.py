from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch
from deepspeed import DeepSpeedEngine

from rl4llm.constants import TRAIN_PHASE
from rl4llm.core.base_trainer import RLConfig, RLTrainer
from rl4llm.envs import EpisodeData


@pytest.fixture
def dummy_config():
    """Provides a minimal RLConfig for testing."""
    config = MagicMock(spec=RLConfig)
    config.train_rollout_size = 8
    config.kl_loss_coef = 0.0
    config.max_steps = 1
    config.eval_interval = 1
    config.eval_rollout_size = 4
    config.checkpoint_interval = 1
    config.sync_reference_interval = 1
    return config


@pytest.fixture
def dummy_policy_engine():
    """Mocks DeepSpeed policy engine."""
    engine = MagicMock()
    engine.zero_optimization_stage.return_value = 2
    engine.bfloat16_enabled.return_value = False
    return engine


@pytest.fixture
def dummy_dist_manager():
    """Mocks a simple DistributedManager."""
    dist = MagicMock()
    dist.world_size = 1
    dist.is_master = True
    dist.device = 'cpu'
    return dist


@pytest.fixture
def dummy_logger():
    """Mocks LoggingManager."""
    logger = MagicMock()
    logger.timer.return_value.__enter__.return_value = None
    logger.timer.return_value.__exit__.return_value = None
    return logger


@pytest.fixture
def trainer_base(
    dummy_config, dummy_policy_engine, dummy_dist_manager, dummy_logger
):
    """Provides a dummy RLTrainer subclass for testing."""

    class DummyTrainer(RLTrainer):
        def initialize_trainer(self):
            pass

        def generate_experience(self):
            return ['exp']

        def compute_loss(self, experience_batch, **kwargs):
            return torch.tensor(0.0), {}

        def build_train_loader(self, experience):
            return ['batch']

        def evaluate_step(self):
            pass

        def train_step(self, train_dataloader):
            pass

    return DummyTrainer(
        config=dummy_config,
        tokenizer=MagicMock(),
        policy_engine=dummy_policy_engine,
        dist_manager=dummy_dist_manager,
        logger=dummy_logger,
        artifacts_path='/tmp/test_rl_trainer',
        train_env=MagicMock(),
        eval_env=MagicMock(),
        inference_client=MagicMock(),
    )


def test_log_batch_episodes_invalid_phase(trainer_base):
    """Raises error on invalid log phase."""
    with pytest.raises(ValueError):
        trainer_base.log_batch_episodes('invalid_phase', [], 0)


def test_log_batch_episodes_valid(trainer_base):
    """Logs sample and scalar for valid training episode."""
    episode = EpisodeData(
        prompt_text='a',
        prompt_tokens=torch.tensor([1]),
        completion_text='b',
        completion_tokens=torch.tensor([2]),
        prompt_length=1,
        completion_length=2,
        reward_dict={'reward': 1.0},
        raw_data={'ground_truth': 'gt'},
    )
    trainer_base.log_batch_episodes(TRAIN_PHASE, [episode], 1)
    trainer_base.logger.log_sample.assert_called()
    trainer_base.logger.log_scalar.assert_called()


def test_prepare_modes(trainer_base):
    """Switches between eval and train modes."""
    trainer_base.policy_engine.eval = MagicMock()
    trainer_base.policy_engine.train = MagicMock()
    trainer_base.reference_model = MagicMock()
    trainer_base.reference_model.to.return_value = trainer_base.reference_model

    trainer_base._prepare_for_generation()
    trainer_base.policy_engine.eval.assert_called()

    trainer_base._prepare_for_training()
    trainer_base.policy_engine.train.assert_called()


def test_checkpoint_saving(trainer_base):
    """Saves policy engine checkpoint at given step."""
    trainer_base.save_checkpoint(1)
    trainer_base.policy_engine.save_checkpoint.assert_called()


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
    trainer_base.dist_manager.barrier.assert_called()


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
    trainer_base.dist_manager.barrier.assert_called()


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
    trainer_base.dist_manager.is_master = True

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
    trainer_base.dist_manager.barrier.assert_called()


def test_sync_policy_model_inference_disabled(trainer_base, mocker):
    """Tests that sync does nothing if inference engine is disabled."""
    mocker.patch(
        'rl4llm.core.base_trainer.RLTrainer.is_inference_engine_enabled',
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
        'rl4llm.core.base_trainer.RLTrainer.is_inference_engine_enabled',
        return_value=True,
    )
    mock_tempdir.return_value.__enter__.return_value = (
        '/fake/temp/path'  # Mock the temp path
    )
    trainer_base.dist_manager.is_master = True
    trainer_base.save_weights_hf_pretrained = MagicMock()

    trainer_base.sync_policy_model()

    trainer_base.save_weights_hf_pretrained.assert_called_once_with(
        trainer_base.policy_engine, '/fake/temp/path'
    )
    trainer_base.inference_client.resume_memory.assert_called_once()
    trainer_base.inference_client.update_weights_from_file.assert_called_once_with(
        model_path='/fake/temp/path'
    )
    trainer_base.dist_manager.barrier.assert_called()


@patch('tempfile.TemporaryDirectory')
def test_sync_policy_model_success_non_master(
    mock_tempdir, trainer_base, mocker
):
    """Tests successful policy sync on a non-master rank."""
    mocker.patch(
        'rl4llm.core.base_trainer.RLTrainer.is_inference_engine_enabled',
        return_value=True,
    )
    mock_tempdir.return_value.__enter__.return_value = '/fake/temp/path'
    trainer_base.is_inference_engine_enabled.return_value = True
    trainer_base.dist_manager.is_master = False  # Set to non-master
    trainer_base.save_weights_hf_pretrained = MagicMock()

    trainer_base.sync_policy_model()

    trainer_base.save_weights_hf_pretrained.assert_called_once_with(
        trainer_base.policy_engine, '/fake/temp/path'
    )
    # Inference client methods should not be called on non-master
    trainer_base.inference_client.resume_memory.assert_not_called()
    trainer_base.inference_client.update_weights_from_file.assert_not_called()
    trainer_base.dist_manager.barrier.assert_called()
