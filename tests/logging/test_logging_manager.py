import html
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pytest
import yaml

from rl4llm.logging.manager import LoggingManager


# ------------------------------------------------
# Fake Logger and setup_logger override
# ------------------------------------------------
class FakeLogger:
    def __init__(self):
        self.messages = []  # Store tuples: (level, message)

    def info(self, msg):
        self.messages.append(('info', msg))

    def debug(self, msg):
        self.messages.append(('debug', msg))

    def warning(self, msg):
        self.messages.append(('warning', msg))

    def error(self, msg):
        self.messages.append(('error', msg))


def fake_setup_logger(rank, log_level):
    # Return a FakeLogger regardless of rank or level.
    return FakeLogger()


# ------------------------------------------------
# Fake Distributed Manager
# ------------------------------------------------
class FakeDistManager:
    def __init__(self, is_master=True, world_size=1, global_rank=0):
        self.is_master = is_master
        self.world_size = world_size
        self.global_rank = global_rank

    def barrier(self):
        pass


# ------------------------------------------------
# Fake Configuration Object
# ------------------------------------------------
class FakeConfig:
    def __init__(self):
        self.project_name = 'fake_project'
        self.run_name = 'fake_run'
        self.run_id = 'fake_run_id'


# ------------------------------------------------
# Pytest Fixture for LoggingManager Instance
# ------------------------------------------------
@pytest.fixture
def logging_manager(tmp_path, monkeypatch):
    # Monkey-patch the setup_logger function in the module where LoggingManager is defined.
    # (Assuming LoggingManager is defined in rl4llm.logging.manager.)
    import rl4llm.logging.manager as lm

    monkeypatch.setattr(lm, 'setup_logger', fake_setup_logger)
    # Also ensure that the TRAIN and EVAL constants are set.
    lm.LoggingManager.TRAIN = 'train'
    lm.LoggingManager.EVAL = 'eval'

    fake_config = FakeConfig()
    fake_dist = FakeDistManager(is_master=True, world_size=1, global_rank=0)
    log_dir = str(tmp_path / 'logs')
    # Create an instance with a small sample buffer and sample interval to ease testing.
    return lm.LoggingManager(
        config=fake_config,
        dist_manager=fake_dist,
        log_dir=log_dir,
        metrics_aggregation_config=None,
        enable_wandb=False,
        enable_tensorboard=True,
        log_sample_interval=1,
        max_backend_samples=2,
        sample_buffer_size=2,
        sample_file_format='jsonl',
        log_level='DEBUG',
    )


# ------------------------------------------------
# Tests for LoggingManager
# ------------------------------------------------
def test_logging_manager_initialization(logging_manager):
    # Verify that all underlying handlers are created.
    assert hasattr(logging_manager, 'metric_handler')
    assert hasattr(logging_manager, 'sample_handler')
    assert hasattr(logging_manager, 'backend_handler')
    # _handlers list should contain three handlers.
    assert len(logging_manager._handlers) == 3
    # The console logger should be our FakeLogger.
    assert hasattr(logging_manager.console_logger, 'messages')


def test_phase_management(logging_manager):
    # Initially, no phase is active.
    assert logging_manager._get_current_phase() is None
    # Use the train_scope context manager.
    with logging_manager.train_scope():
        assert logging_manager._get_current_phase() == 'train'
        # Log a scalar value. The metric key should be "train/some_metric".
        logging_manager.log_scalar('some_metric', 1.0)
        buf = logging_manager.metric_handler._metric_buffer
        assert 'train/some_metric' in buf
        # Check that the value was stored.
        assert buf['train/some_metric'] == [1.0]
    # After scope exit, phase reverts.
    assert logging_manager._get_current_phase() is None


def test_log_metrics_dict(logging_manager):
    # Use eval_scope.
    with logging_manager.eval_scope():
        metrics = {'m1': 0.1, 'm2': 2}
        logging_manager.log_metrics_dict(metrics)
        buf = logging_manager.metric_handler._metric_buffer
        # Keys should be prefixed with "eval/".
        assert 'eval/m1' in buf
        assert 'eval/m2' in buf
        assert buf['eval/m1'] == [0.1]
        assert buf['eval/m2'] == [2]


def test_log_sample(logging_manager):
    # Set a phase and log a sample.
    with logging_manager.train_scope():
        logging_manager.log_sample('tag1', {'data': 'value', 'step': 5}, step=5)
        # Check that the SampleHandler's local log count increased.
        count = logging_manager.sample_handler._local_sample_log_counts.get(
            'train', 0
        )
        assert count == 1


def test_log_hyperparams(logging_manager):
    # To test delegation, override backend_handler.log_hyperparams.
    called = False

    def fake_log_hyperparams(params):
        nonlocal called
        called = True

    logging_manager.backend_handler.log_hyperparams = fake_log_hyperparams
    hyperparams = {'lr': 0.001, 'batch_size': 32}
    logging_manager.log_hyperparams(hyperparams)
    assert called is True
    # Also, on master, hyperparameters are logged to the console.
    # Check that the FakeLogger recorded an info message containing "Hyperparameters:"
    messages = [
        msg
        for level, msg in logging_manager.console_logger.messages
        if 'Hyperparameters:' in msg
    ]
    assert messages  # Should not be empty


def test_timer(logging_manager):
    # Use the timer context manager to log elapsed time.
    with logging_manager.timer('test_timer'):
        time.sleep(0.1)
    buf = logging_manager.metric_handler._metric_buffer
    # A time metric with key "time/test_timer_sec" should be present.
    assert 'time/test_timer_sec' in buf
    # Verify that the logged time is a positive number.
    elapsed = buf['time/test_timer_sec'][0]
    assert elapsed > 0


def test_aggregate_and_log(logging_manager, monkeypatch):
    # Prepare fake aggregated metrics and sample data.
    fake_metrics = {'train/loss_mean': 0.5}
    fake_samples = [('train/tag1', {'step': 10, 'data': 'val'})]
    monkeypatch.setattr(
        logging_manager.metric_handler, 'aggregate', lambda: fake_metrics
    )
    monkeypatch.setattr(
        logging_manager.sample_handler,
        'collect_backend_samples',
        lambda: fake_samples,
    )
    # Record calls to backend_handler logging methods.
    logged_metrics = []
    logged_samples = []

    def fake_log_metrics(metrics, step):
        logged_metrics.append((metrics, step))

    def fake_log_sample_text(tag, formatted_text, step):
        logged_samples.append((tag, formatted_text, step))

    monkeypatch.setattr(
        logging_manager.backend_handler, 'log_metrics', fake_log_metrics
    )
    monkeypatch.setattr(
        logging_manager.backend_handler, 'log_sample_text', fake_log_sample_text
    )
    # Also capture console log messages.
    initial_info_count = len(logging_manager.console_logger.messages)
    logging_manager.aggregate_and_log(step=20)
    # Verify backend logging: our fake metrics and sample should have been processed.
    assert len(logged_metrics) == 1
    assert logged_metrics[0][0] == fake_metrics
    # Since sample data is non-empty, log_sample_text should have been called.
    assert len(logged_samples) == 1
    # Check that both metric and sample buffers have been cleared.
    assert logging_manager.metric_handler._metric_buffer == {}
    assert logging_manager.sample_handler._samples_for_backend_buffer == []
    # Also, console logger should have received additional info messages.
    assert len(logging_manager.console_logger.messages) > initial_info_count


def test_flush(logging_manager, monkeypatch):
    # Override sample_handler.flush to record that it was called.
    flush_called = False

    def fake_flush():
        nonlocal flush_called
        flush_called = True

    monkeypatch.setattr(logging_manager.sample_handler, 'flush', fake_flush)
    logging_manager.flush()
    assert flush_called is True


def test_close(logging_manager, monkeypatch):
    # Patch each handler's close to record a call.
    calls = []
    for handler in logging_manager._handlers:
        monkeypatch.setattr(
            handler, 'close', lambda h=handler: calls.append(type(h).__name__)
        )
    # Simulate a distributed setup with world_size > 1.
    logging_manager.dist_manager.world_size = 2
    logging_manager.world_size = 2
    barrier_called = False

    def fake_barrier():
        nonlocal barrier_called
        barrier_called = True

    monkeypatch.setattr(logging_manager.dist_manager, 'barrier', fake_barrier)
    logging_manager.close()
    # The handlers should be closed in reverse order.
    expected = [type(h).__name__ for h in reversed(logging_manager._handlers)]
    assert calls == expected
    # Verify that the distributed barrier was called.
    assert barrier_called is True
