from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from rl4llm.logging.handlers.metric import MetricHandler


# Helper mock distributed manager
@pytest.fixture
def mock_dist_manager():
    mock = MagicMock()
    mock.is_master = True
    mock.world_size = 1
    mock.barrier.return_value = None
    mock.gather_object.side_effect = lambda obj, dst=0: [obj]
    return mock


@pytest.fixture
def handler(mock_dist_manager):
    return MetricHandler(dist_manager=mock_dist_manager)


def test_log_scalar_float(handler):
    handler.log_scalar('reward', 1.0)
    assert handler._metric_buffer['reward'] == [1.0]


def test_log_scalar_tensor(handler):
    tensor_val = torch.tensor(2.0)
    handler.log_scalar('reward', tensor_val)
    assert handler._metric_buffer['reward'] == [2.0]


def test_log_scalar_invalid(handler, caplog):
    handler.log_scalar('reward', 'invalid')
    assert 'Could not convert metric' in caplog.text
    assert 'reward' not in handler._metric_buffer


def test_log_scalar_non_finite(handler, caplog):
    handler.log_scalar('reward', float('inf'))
    assert 'Non-finite value' in caplog.text
    assert 'reward' not in handler._metric_buffer


def test_get_aggregation_methods_direct_match(handler):
    methods = handler._get_aggregation_methods('reward')
    assert methods == ['mean', 'std', 'p90', 'min', 'max']


def test_get_aggregation_methods_regex_match(handler):
    methods = handler._get_aggregation_methods('train_loss')
    assert 'mean' in methods


def test_aggregate_single_value(handler):
    handler.log_scalar('reward', 1.0)
    result = handler.aggregate()
    assert np.isclose(result['reward_mean'], 1.0)


def test_aggregate_multiple_values(handler):
    for val in [1.0, 2.0, 3.0]:
        handler.log_scalar('reward', val)
    result = handler.aggregate()
    assert np.isclose(result['reward_mean'], 2.0)
    assert np.isclose(result['reward_p90'], 2.8)
    assert np.isclose(result['reward_min'], 1.0)
    assert np.isclose(result['reward_max'], 3.0)


def test_clear_buffer(handler):
    handler.log_scalar('reward', 1.0)
    handler.clear_buffer()
    assert handler._metric_buffer == {}


def test_user_config_override(mock_dist_manager):
    custom_config = {'custom_metric': ['sum', 'count']}
    handler = MetricHandler(
        dist_manager=mock_dist_manager, user_aggregation_config=custom_config
    )
    handler.log_scalar('custom_metric', 10.0)
    handler.log_scalar('custom_metric', 5.0)
    result = handler.aggregate()
    assert result['custom_metric_sum'] == 15.0
    assert result['custom_metric_count'] == 2
