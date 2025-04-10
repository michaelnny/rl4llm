from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch

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

        def sync_policy_model(self):
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


def test_sync_reference_model_with_model(trainer_base):
    """Syncs reference model correctly and increments counter."""
    trainer_base._create_reference_model = MagicMock(return_value=MagicMock())
    trainer_base.reference_model = deepcopy(
        trainer_base._create_reference_model()
    )

    model_mock = MagicMock()
    trainer_base.unwrapped_model_for_generation = MagicMock(
        return_value=MagicMock(
            __enter__=lambda s: model_mock,
            __exit__=lambda s, t, v, tb: None,
        )
    )
    trainer_base._sync_reference_model()
    assert trainer_base.ref_update_count == 1


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
