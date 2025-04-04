import logging
import os
import time

import numpy as np
import pytest
import yaml

from rl4llm.logging.handlers.backend_handler import BackendHandler


class FakeWandB:
    def __init__(self):
        self.logged_metrics = []
        self.finished = False

    def log(self, metrics, step):
        self.logged_metrics.append((metrics, step))

    def finish(self):
        self.finished = True

    def get_url(self):
        return 'http://fake-wandb-url'


class FakeWandBModule:
    run = None

    @staticmethod
    def init(**kwargs):
        instance = FakeWandB()
        FakeWandBModule.run = instance
        return instance

    @staticmethod
    def log(metrics, step):
        if FakeWandBModule.run:
            FakeWandBModule.run.logged_metrics.append((metrics, step))

    @staticmethod
    def finish():
        if FakeWandBModule.run:
            FakeWandBModule.run.finished = True

    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Fake util attribute with generate_id method
    util = type('util', (), {'generate_id': lambda: 'fake_id'})


class FakeSummaryWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.logged_scalars = []
        self.logged_texts = []
        self.closed = False

    def add_scalar(self, tag, scalar_value, global_step):
        self.logged_scalars.append((tag, scalar_value, global_step))

    def add_text(self, tag, text, global_step):
        self.logged_texts.append((tag, text, global_step))

    def close(self):
        self.closed = True


# Test non-master instance: writer should not be initialized.
def test_backend_handler_non_master(tmp_path, caplog):
    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=True,
        enable_tensorboard=True,
        is_master=False,
        logger=logging.getLogger('test_non_master'),
    )
    # On non-master, writer remains None.
    assert handler._writer is None
    assert not handler._can_log_flag

    # Calling log methods should not raise exceptions.
    handler.log_metrics({'accuracy': 0.95}, step=1)
    handler.log_sample_text('sample', 'Test sample text', step=1)
    handler.log_hyperparams({'lr': 0.001})
    handler.close()


# Test with WandB enabled by injecting our fake wandb module.
def test_backend_handler_wandb(tmp_path, monkeypatch, caplog):
    # Inject fake wandb module into sys.modules.
    monkeypatch.setitem(__import__('sys').modules, 'wandb', FakeWandBModule)

    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=True,
        enable_tensorboard=False,
        is_master=True,
        logger=logging.getLogger('test_wandb'),
    )
    # Verify writer is set (i.e. FakeWandBModule was used).
    assert handler._writer is not None

    # Log metrics. Only valid, finite numeric values should be sent.
    metrics = {
        'loss': 0.123,
        'accuracy': 0.95,
        'non_numeric': 'skip',
        'nan': float('nan'),
    }
    handler.log_metrics(metrics, step=10)
    writer = FakeWandBModule.run
    # Expect only loss and accuracy to be logged.
    assert writer.logged_metrics[0][0] == {'loss': 0.123, 'accuracy': 0.95}
    assert writer.logged_metrics[0][1] == 10

    # Test log_sample_text (this path uses wandb.Html internally).
    # We won’t inspect the actual HTML conversion but ensure no exception is raised.
    handler.log_sample_text('sample', 'Line1\nLine2', step=5)

    # Test log_hyperparams: it should update the config.
    try:
        handler.log_hyperparams({'batch_size': 32})
    except Exception:
        pytest.fail('log_hyperparams raised an exception with WandB writer.')

    # Test close: should call finish() on the writer.
    handler.close()
    assert writer.finished is True


# Test with TensorBoard enabled when wandb is not available.
def test_backend_handler_tensorboard(tmp_path, monkeypatch, caplog):
    # Simulate ImportError for wandb by overriding __import__
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == 'wandb':
            raise ImportError("No module named 'wandb'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', fake_import)

    # Inject fake tensorboard module with our FakeSummaryWriter.
    fake_tb_module = type('fake_tb', (), {'SummaryWriter': FakeSummaryWriter})
    monkeypatch.setitem(
        __import__('sys').modules, 'torch.utils.tensorboard', fake_tb_module
    )

    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=False,
        enable_tensorboard=True,
        is_master=True,
        logger=logging.getLogger('test_tb'),
    )
    # Verify writer is set and resembles a TensorBoard writer.
    assert handler._writer is not None
    assert hasattr(handler._writer, 'add_scalar')

    # Log metrics: only valid values should be logged via add_scalar.
    metrics = {
        'loss': 0.456,
        'accuracy': 0.87,
        'non_numeric': 'skip',
        'inf': float('inf'),
    }
    handler.log_metrics(metrics, step=20)
    writer = handler._writer
    # Expect two valid entries (loss and accuracy).
    assert len(writer.logged_scalars) == 2

    # Test log_sample_text: should use add_text.
    handler.log_sample_text('sample', 'Sample text', step=15)
    assert len(writer.logged_texts) == 1

    # Test log_hyperparams: should add text representing hyperparameters.
    handler.log_hyperparams({'dropout': 0.5})
    # Close writer.
    handler.close()
    assert writer.closed is True


# Test that log_metrics filters out invalid or non-finite metrics.
def test_log_metrics_no_valid(tmp_path, monkeypatch, caplog):
    # Create a fake writer supporting add_scalar.
    class FakeWriter:
        def __init__(self):
            self.logged_scalars = []

        def add_scalar(self, tag, value, step):
            self.logged_scalars.append((tag, value, step))

    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=False,
        enable_tensorboard=True,
        is_master=True,
        logger=logging.getLogger('test_no_valid'),
    )
    handler._writer = FakeWriter()
    handler._can_log_flag = True

    invalid_metrics = {'a': 'str', 'b': None, 'c': float('nan')}
    handler.log_metrics(invalid_metrics, step=30)
    # No valid metrics should have been logged.
    assert len(handler._writer.logged_scalars) == 0


# Test the close behavior for both TensorBoard and WandB style writers.
def test_close_backend_writer(tmp_path, monkeypatch, caplog):
    log_dir = str(tmp_path / 'logs')

    # Test for TensorBoard-style writer.
    class FakeWriterTB:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    handler_tb = BackendHandler(
        log_dir=log_dir,
        enable_wandb=False,
        enable_tensorboard=True,
        is_master=True,
        logger=logging.getLogger('test_close_tb'),
    )
    handler_tb._writer = FakeWriterTB()
    handler_tb._can_log_flag = True
    handler_tb.close()
    assert handler_tb._writer is None

    # Test for WandB-style writer.
    class FakeWriterWB:
        def __init__(self):
            self.finished = False

        def finish(self):
            self.finished = True

    handler_wb = BackendHandler(
        log_dir=log_dir,
        enable_wandb=True,
        enable_tensorboard=False,
        is_master=True,
        logger=logging.getLogger('test_close_wb'),
    )
    handler_wb._writer = FakeWriterWB()
    handler_wb._can_log_flag = True
    handler_wb.close()
    assert handler_wb._writer is None
