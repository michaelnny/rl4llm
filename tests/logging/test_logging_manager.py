import time

import pytest

from rl4llm.logging.logging_manager import LoggingManager


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(('info', msg))

    def debug(self, msg):
        self.messages.append(('debug', msg))

    def warning(self, msg):
        self.messages.append(('warning', msg))

    def error(self, msg):
        self.messages.append(('error', msg))


def fake_setup_logger(rank, log_level):
    return FakeLogger()


class FakeDistManager:
    def __init__(self, is_master=True, world_size=1, global_rank=0):
        self.is_master = is_master
        self.world_size = world_size
        self.global_rank = global_rank

    def barrier(self):
        pass


@pytest.fixture
def logging_manager(tmp_path, monkeypatch):
    """Fixture to initialize a LoggingManager with fakes and temporary paths."""
    import rl4llm.logging.logging_manager as lm

    monkeypatch.setattr(lm, 'setup_logger', fake_setup_logger)
    lm.LoggingManager.TRAIN = 'train'
    lm.LoggingManager.EVAL = 'eval'

    fake_dist = FakeDistManager(is_master=True, world_size=1, global_rank=0)
    return lm.LoggingManager(
        dist_manager=fake_dist,
        output_dir=str(tmp_path / 'logs'),
        metrics_aggregation_config=None,
        enable_wandb=False,
        enable_tensorboard=True,
        sample_buffer_size=2,
        sample_file_format='jsonl',
        log_level='DEBUG',
    )


def test_logging_manager_initialization(logging_manager):
    """Test LoggingManager initializes all handlers and uses fake logger."""
    assert hasattr(logging_manager, 'metric_handler')
    assert hasattr(logging_manager, 'sample_handler')
    assert hasattr(logging_manager, 'backend_handler')
    assert hasattr(logging_manager, 'resource_handler')
    assert len(logging_manager._handlers) == 4
    assert hasattr(logging_manager.console_logger, 'messages')


@pytest.mark.parametrize(
    'phase, metric_key, metric_value',
    [
        ('train', 'train/some_metric', 1.0),
        ('eval', 'eval/m1', 0.1),
        ('eval', 'eval/m2', 2.0),
    ],
)
def test_logging_metrics(logging_manager, phase, metric_key, metric_value):
    """Test logging scalar metrics and metrics dictionary."""
    if 'some_metric' in metric_key:
        logging_manager.log_scalar(metric_key, metric_value)
    else:
        logging_manager.log_metrics_dict({metric_key: metric_value})

    buf = logging_manager.metric_handler._metric_buffer
    assert metric_key in buf
    assert buf[metric_key] == [metric_value]


def test_log_sample(logging_manager):
    """Test that log_sample adds data to sample buffer."""
    logging_manager.log_sample(
        phase='train',
        sample_data={'data': 'value', 'step': 5},
        step=5,
    )
    buffer = logging_manager.sample_handler._file_loggers['train']._buffer
    assert len(buffer) == 1


def test_log_hyperparams(logging_manager):
    """Test that hyperparams are delegated to backend handler."""
    called = False

    def fake_log_hyperparams(params):
        nonlocal called
        called = True

    logging_manager.backend_handler.log_hyperparams = fake_log_hyperparams
    logging_manager.log_hyperparams({'lr': 0.001, 'batch_size': 32})
    assert called is True


def test_timer(logging_manager):
    """Test that elapsed time is logged using the timer context."""
    with logging_manager.timer('test_timer'):
        time.sleep(0.05)
    buf = logging_manager.metric_handler._metric_buffer
    assert 'time/test_timer' in buf
    assert buf['time/test_timer'][0] > 0


def test_aggregate_and_log(logging_manager, monkeypatch):
    """Test aggregation and logging of metrics and samples."""
    fake_metrics = {'train/loss_mean': 0.5}
    monkeypatch.setattr(
        logging_manager.metric_handler, 'aggregate', lambda: fake_metrics
    )

    logged_metrics = []
    monkeypatch.setattr(
        logging_manager.backend_handler,
        'log_metrics',
        lambda m, s: logged_metrics.append((m, s)),
    )

    initial_logs = len(logging_manager.console_logger.messages)
    logging_manager.aggregate_and_log(step=20)

    assert logged_metrics[0][0] == fake_metrics
    assert logging_manager.metric_handler._metric_buffer == {}
    assert len(logging_manager.console_logger.messages) > initial_logs


def test_flush(logging_manager, monkeypatch):
    """Test flush triggers sample handler flush."""
    flush_called = {'flag': False}

    def fake_flush():
        flush_called['flag'] = True

    monkeypatch.setattr(logging_manager.sample_handler, 'flush', fake_flush)
    logging_manager.flush()
    assert flush_called['flag'] is True


def test_close(logging_manager, monkeypatch):
    """Test that close calls all handler closures and sync barrier."""
    closed = []

    for handler in logging_manager._handlers:

        def make_closer(h):
            return lambda: closed.append(type(h).__name__)

        monkeypatch.setattr(handler, 'close', make_closer(handler))

    logging_manager.dist_manager.world_size = 2
    logging_manager.world_size = 2

    barrier_called = {'flag': False}

    def fake_barrier():
        barrier_called['flag'] = True

    monkeypatch.setattr(logging_manager.dist_manager, 'barrier', fake_barrier)

    logging_manager.close()
    expected = [type(h).__name__ for h in reversed(logging_manager._handlers)]
    assert closed == expected
    assert barrier_called['flag'] is True
