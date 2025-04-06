import logging
import os
import sys
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

# Import the class to be tested
from rl4llm.logging.handlers.backend_handler import BackendHandler

# --- Fixtures ---


@pytest.fixture
def mock_logger():
    """Provides a mock logger instance."""
    # Create a real spec to avoid issues with missing methods like isEnabledFor
    logger = logging.getLogger('test_backend_handler')
    logger.setLevel(
        logging.DEBUG
    )  # Ensure debug messages are processed by mocks
    mock = MagicMock(spec=logger)
    # Make isEnabledFor return True for DEBUG level to capture debug logs
    mock.isEnabledFor.return_value = True
    return mock


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Provides a temporary directory path for logs."""
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def mock_wandb(mocker):
    """Provides a mock wandb module, patching the import lookup."""
    mock = MagicMock()
    # Mock the structure expected by the handler
    mock.Settings = MagicMock()
    mock.util = MagicMock()
    mock.util.generate_id.return_value = 'test_run_id'
    mock_run = MagicMock()
    mock_run.get_url.return_value = 'http://mock_wandb_url'
    # Mock config directly on the main mock object as well, as the code uses it there
    mock.config = MagicMock()
    mock.config.update = MagicMock()
    mock_run.config = (
        mock.config
    )  # Can share the same mock if needed, or make distinct
    mock.run = mock_run
    mock.init.return_value = mock_run
    mock.Html = MagicMock(side_effect=lambda x: f"HTML:{x}")
    mock.log = MagicMock()
    mock.finish = MagicMock()
    # Patch the import location using sys.modules
    mocker.patch.dict(sys.modules, {'wandb': mock})
    return mock


@pytest.fixture
def mock_summary_writer(mocker):
    """Provides a mock SummaryWriter class, patching the import lookup."""
    mock_writer_instance = MagicMock(spec=['add_scalar', 'add_text', 'close'])
    mock_writer_class = MagicMock(return_value=mock_writer_instance)
    # Create a mock torch module structure if it doesn't exist
    mock_torch = MagicMock()
    mock_torch.utils = MagicMock()
    mock_torch.utils.tensorboard = MagicMock(SummaryWriter=mock_writer_class)
    # Patch the import location using sys.modules
    mocker.patch.dict(
        sys.modules,
        {
            'torch': mock_torch,
            'torch.utils': mock_torch.utils,
            'torch.utils.tensorboard': mock_torch.utils.tensorboard,
        },
    )
    return mock_writer_class, mock_writer_instance


@pytest.fixture
def mock_os_makedirs(mocker):
    """Mocks os.makedirs."""
    # Use autospec=True if possible, otherwise just patch
    return mocker.patch('os.makedirs', autospec=True)


@pytest.fixture
def mock_time_strftime(mocker):
    """Mocks time.strftime to return a fixed timestamp."""
    # Use autospec=True if possible, otherwise just patch
    return mocker.patch(
        'time.strftime', return_value='20230101_120000', autospec=True
    )


# --- Test Initialization (Focusing on the failed test) ---


def test_init_non_master(mock_logger, tmp_log_dir):
    """Tests that handler initialization skips writer setup on non-master nodes."""
    # The BaseHandler init logs 'Initialized BackendHandler' at DEBUG level
    # The BackendHandler init logs the specific message also at DEBUG level
    handler = BackendHandler(
        log_dir=tmp_log_dir,
        enable_wandb=True,
        enable_tensorboard=True,
        is_master=False,
        logger=mock_logger,
    )
    assert handler._writer is None
    assert not handler._can_log_flag
    # Check that the specific debug message was logged among potentially others
    mock_logger.debug.assert_any_call(
        'Not master rank, skipping backend writer setup.'
    )
    # Verify the base handler init log was also called
    mock_logger.debug.assert_any_call('Initialized BackendHandler')
    # Ensure no warning/error logs related to backend setup were called
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()


# --- Test Logging Methods (Focusing on the failed test) ---


@pytest.fixture
@pytest.mark.usefixtures('mock_os_makedirs', 'mock_time_strftime')
def wandb_handler(mock_logger, tmp_log_dir, mock_wandb):
    """Provides a handler configured for WandB."""
    handler = BackendHandler(
        log_dir=tmp_log_dir,
        enable_wandb=True,
        enable_tensorboard=False,
        is_master=True,
        logger=mock_logger,
    )
    mock_logger.reset_mock()
    # Reset only the methods used in logging, not the whole mock structure
    mock_wandb.log.reset_mock()
    mock_wandb.config.update.reset_mock()  # Reset the correct update mock
    mock_wandb.Html.reset_mock()
    return handler, mock_wandb


@pytest.fixture
@pytest.mark.usefixtures('mock_os_makedirs')
def tb_handler(mock_logger, tmp_log_dir, mock_summary_writer):
    """Provides a handler configured for TensorBoard."""
    _, mock_sw_instance = mock_summary_writer
    handler = BackendHandler(
        log_dir=tmp_log_dir,
        enable_wandb=False,
        enable_tensorboard=True,
        is_master=True,
        logger=mock_logger,
    )
    mock_logger.reset_mock()
    mock_sw_instance.reset_mock()
    return handler, mock_sw_instance


@pytest.mark.parametrize(
    'metrics, expected_logged_metrics_wandb, expected_calls_tb',
    [
        (
            {'metric1': 1.0, 'metric2': 5},
            {'metric1': 1.0, 'metric2': 5},
            [('metric1', 1.0, 10), ('metric2', 5, 10)],
        ),
        (
            {'metric1': 1.0, 'invalid': 'string'},
            {'metric1': 1.0},
            [('metric1', 1.0, 10)],
        ),
        ({'nan_metric': np.nan, 'inf_metric': np.inf}, {}, []),
        ({}, {}, []),
    ],
)
def test_log_metrics(
    wandb_handler,
    tb_handler,
    metrics,
    expected_logged_metrics_wandb,
    expected_calls_tb,
):
    """Tests logging metrics to WandB and TensorBoard."""
    step = 10
    handler_w, mock_w = wandb_handler
    handler_t, mock_t = tb_handler

    # Test WandB
    handler_w.log_metrics(metrics, step)
    if expected_logged_metrics_wandb:
        mock_w.log.assert_called_once_with(
            expected_logged_metrics_wandb, step=step
        )
    else:
        mock_w.log.assert_not_called()

    # Test TensorBoard
    handler_t.log_metrics(metrics, step)
    assert mock_t.add_scalar.call_count == len(expected_calls_tb)
    for call_args in expected_calls_tb:
        mock_t.add_scalar.assert_any_call(*call_args)


def test_log_sample_text(wandb_handler, tb_handler):
    """Tests logging sample text to WandB and TensorBoard."""
    tag = 'test_sample'
    text = 'Line 1\nLine 2'
    step = 20
    handler_w, mock_w = wandb_handler
    handler_t, mock_t = tb_handler

    # Test WandB
    expected_html = 'HTML:Line 1<br>Line 2'
    handler_w.log_sample_text(tag, text, step)
    mock_w.log.assert_called_once_with(
        {f"samples/{tag}": expected_html}, step=step
    )
    mock_w.Html.assert_called_once_with('Line 1<br>Line 2')

    # Test TensorBoard
    handler_t.log_sample_text(tag, text, step)
    mock_t.add_text.assert_called_once_with(f"samples/{tag}", text, step)


def test_log_hyperparams(wandb_handler, tb_handler):
    """Tests logging hyperparameters to WandB and TensorBoard."""
    params = {'lr': 0.001, 'batch_size': 32}
    handler_w, mock_w = wandb_handler
    handler_t, mock_t = tb_handler

    # Test WandB
    handler_w.log_hyperparams(params)
    # The code calls self._writer.config.update, and self._writer is mock_w itself
    mock_w.config.update.assert_called_once_with(params, allow_val_change=True)
    # Ensure the run's config wasn't called directly if it's different
    # (In our fixture they point to the same mock, so this check might be redundant
    # but good practice if they were distinct mocks)
    if (
        hasattr(mock_w, 'run')
        and hasattr(mock_w.run, 'config')
        and mock_w.run.config is not mock_w.config
    ):
        mock_w.run.config.update.assert_not_called()

    # Test TensorBoard
    expected_yaml_str = 'lr: 0.001\nbatch_size: 32\n'
    expected_text = f"```yaml\n{expected_yaml_str}\n```"
    handler_t.log_hyperparams(params)
    mock_t.add_text.assert_called_once_with(
        'configuration/hyperparameters', expected_text, 0
    )


def test_logging_methods_non_master_or_no_backend(
    mock_logger, tmp_log_dir, mock_wandb, mock_summary_writer
):
    """Tests that logging methods do nothing on non-master or when no backend is active."""
    _, mock_sw_instance = mock_summary_writer
    non_master_handler = BackendHandler(
        tmp_log_dir, True, True, False, mock_logger
    )
    no_backend_handler = BackendHandler(
        tmp_log_dir, False, False, True, mock_logger
    )

    for handler in [non_master_handler, no_backend_handler]:
        handler.log_metrics({'m': 1}, 1)
        handler.log_sample_text('t', 'txt', 1)
        handler.log_hyperparams({'p': 1})

    mock_wandb.log.assert_not_called()
    mock_wandb.run.config.update.assert_not_called()
    mock_sw_instance.add_scalar.assert_not_called()
    mock_sw_instance.add_text.assert_not_called()


# --- Test Close Method ---


def test_close(wandb_handler, tb_handler, mock_logger):
    """Tests closing the WandB and TensorBoard writers."""
    handler_w, mock_w = wandb_handler
    handler_t, mock_t = tb_handler
    mock_logger.reset_mock()  # Reset logger before close calls

    # Test WandB close
    assert handler_w._can_log_flag
    handler_w.close()
    mock_w.finish.assert_called_once()
    assert handler_w._writer is None
    assert not handler_w._can_log_flag
    mock_logger.info.assert_any_call('Closed WandB writer.')

    # Test TensorBoard close
    assert handler_t._can_log_flag
    handler_t.close()
    mock_t.close.assert_called_once()
    assert handler_t._writer is None
    assert not handler_t._can_log_flag
    mock_logger.info.assert_any_call('Closed TensorBoard writer.')


def test_close_non_master_or_no_backend(
    mock_logger, tmp_log_dir, mock_wandb, mock_summary_writer
):
    """Tests that close does nothing on non-master or when no backend is active."""
    _, mock_sw_instance = mock_summary_writer
    non_master_handler = BackendHandler(
        tmp_log_dir, True, True, False, mock_logger
    )
    no_backend_handler = BackendHandler(
        tmp_log_dir, False, False, True, mock_logger
    )
    mock_logger.reset_mock()

    non_master_handler.close()
    no_backend_handler.close()

    mock_wandb.finish.assert_not_called()
    mock_sw_instance.close.assert_not_called()
    mock_logger.info.assert_not_called()  # No closure messages expected
