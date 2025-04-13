from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from rl4llm.logging.handlers.metric_handler import MetricHandler


@pytest.fixture
def mock_dist_ops():
    """Provides a mock distributed manager for testing."""
    mock = MagicMock()
    mock.is_master = True
    mock.world_size = 1
    mock.barrier.return_value = None
    mock.gather_object.side_effect = lambda obj, dst=0: [obj]
    return mock


@pytest.fixture
def handler(mock_dist_ops):
    """Returns a MetricHandler instance using a mock distributed manager."""
    return MetricHandler(dist_ops=mock_dist_ops)


@pytest.mark.parametrize(
    'value,expected', [(1.0, [1.0]), (torch.tensor(2.0), [2.0])]
)
def test_log_scalar_valid(handler, value, expected):
    """Tests logging of valid scalar values (float or tensor)."""
    handler.log_scalar('reward', value)
    assert handler._metric_buffer['reward'] == expected


@pytest.mark.parametrize(
    'value,expected_msg',
    [
        ('invalid', 'Could not convert metric'),
        (float('inf'), 'Non-finite value'),
    ],
)
def test_log_scalar_invalid_or_nonfinite(handler, caplog, value, expected_msg):
    """Tests handling of invalid or non-finite scalar logging."""
    handler.log_scalar('reward', value)
    assert expected_msg in caplog.text
    assert 'reward' not in handler._metric_buffer


def test_get_aggregation_methods_regex_match(handler):
    """Tests retrieval of default aggregation methods based on metric name."""
    methods = handler._get_aggregation_methods('train_loss')
    assert 'mean' in methods


def test_aggregate_single_value(handler):
    """Tests aggregation when only a single scalar value is logged."""
    handler.log_scalar('reward', 1.0)
    result = handler.aggregate()
    assert np.isclose(result['reward_mean'], 1.0)


def test_aggregate_multiple_values(handler):
    """Tests aggregation of multiple logged scalar values."""
    for val in [1.0, 2.0, 3.0]:
        handler.log_scalar('reward', val)
    result = handler.aggregate()
    assert np.isclose(result['reward_mean'], 2.0)
    assert np.isclose(result['reward_p90'], 2.8)


def test_clear_buffer(handler):
    """Tests clearing of the metric buffer."""
    handler.log_scalar('reward', 1.0)
    handler.clear_buffer()
    assert handler._metric_buffer == {}


def test_user_config_override(mock_dist_ops):
    """Tests aggregation using a user-provided aggregation config."""
    custom_config = {'custom_metric': ['sum', 'count']}
    handler = MetricHandler(
        dist_ops=mock_dist_ops, user_aggregation_config=custom_config
    )
    handler.log_scalar('custom_metric', 10.0)
    handler.log_scalar('custom_metric', 5.0)
    result = handler.aggregate()
    assert result['custom_metric_sum'] == 15.0
    assert result['custom_metric_count'] == 2
