import logging
import os
import time

import numpy as np
import pytest
import yaml

from rl4llm.logging.handlers.backend_handler import BackendHandler


@pytest.fixture
def fake_wandb_module(monkeypatch):
    """Provides a fake WandB module for testing."""

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

    class FakeConfig:
        def __init__(self):
            self.data = {}

        def update(self, params, allow_val_change=True):
            self.data.update(params)

    class FakeWandBUtil:
        @staticmethod
        def generate_id():
            return 'fake_id'

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWandBModule:
        run = None
        config = FakeConfig()
        util = FakeWandBUtil()
        Settings = FakeSettings

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

        @staticmethod
        def Html(html_text):
            return html_text

    monkeypatch.setitem(__import__('sys').modules, 'wandb', FakeWandBModule)
    return FakeWandBModule


@pytest.fixture
def fake_tensorboard_writer(monkeypatch):
    """Provides a fake TensorBoard SummaryWriter for testing."""

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

    fake_tb_module = type('fake_tb', (), {'SummaryWriter': FakeSummaryWriter})
    monkeypatch.setitem(
        __import__('sys').modules, 'torch.utils.tensorboard', fake_tb_module
    )
    return FakeSummaryWriter


def test_backend_handler_non_master(tmp_path, caplog):
    """Tests BackendHandler behavior on a non-master instance."""
    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=True,
        enable_tensorboard=True,
        is_master=False,
        logger=logging.getLogger('test_non_master'),
    )
    assert handler._writer is None
    assert not handler._can_log_flag
    handler.log_metrics({'accuracy': 0.95}, step=1)
    handler.log_sample_text('sample', 'Test sample text', step=1)
    handler.log_hyperparams({'lr': 0.001})
    handler.close()


def test_backend_handler_wandb(tmp_path, fake_wandb_module, caplog):
    """Tests BackendHandler with WandB enabled."""
    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=True,
        enable_tensorboard=False,
        is_master=True,
        logger=logging.getLogger('test_wandb'),
    )
    assert handler._writer is not None
    metrics = {
        'loss': 0.123,
        'accuracy': 0.95,
        'non_numeric': 'skip',
        'nan': float('nan'),
    }
    handler.log_metrics(metrics, step=10)
    writer = fake_wandb_module.run
    assert writer.logged_metrics[0][0] == {'loss': 0.123, 'accuracy': 0.95}
    assert writer.logged_metrics[0][1] == 10
    handler.log_sample_text('sample', 'Line1\nLine2', step=5)
    handler.log_hyperparams({'batch_size': 32})
    handler.close()
    assert writer.finished is True


def test_backend_handler_tensorboard(
    tmp_path, fake_tensorboard_writer, monkeypatch, caplog
):
    """Tests BackendHandler with TensorBoard enabled when WandB is unavailable."""

    def fake_import(name, *args, **kwargs):
        if name == 'wandb':
            raise ImportError("No module named 'wandb'")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', fake_import)
    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=False,
        enable_tensorboard=True,
        is_master=True,
        logger=logging.getLogger('test_tb'),
    )
    assert handler._writer is not None
    assert hasattr(handler._writer, 'add_scalar')
    metrics = {
        'loss': 0.456,
        'accuracy': 0.87,
        'non_numeric': 'skip',
        'inf': float('inf'),
    }
    handler.log_metrics(metrics, step=20)
    writer = handler._writer
    assert len(writer.logged_scalars) == 2
    handler.log_sample_text('sample', 'Sample text', step=15)
    assert len(writer.logged_texts) == 1
    handler.log_hyperparams({'dropout': 0.5})
    handler.close()
    assert writer.closed is True


@pytest.mark.parametrize(
    'metrics, expected_count',
    [
        ({'a': 'str', 'b': None, 'c': float('nan')}, 0),
        ({'loss': 0.456, 'accuracy': 0.87, 'inf': float('inf')}, 2),
        ({'valid': 1.0, 'nan': float('nan')}, 1),
    ],
)
def test_log_metrics_filtering(tmp_path, caplog, metrics, expected_count):
    """Tests that log_metrics filters out invalid or non-finite metrics."""

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
        logger=logging.getLogger('test_filtering'),
    )
    handler._writer = FakeWriter()
    handler._can_log_flag = True
    handler.log_metrics(metrics, step=30)
    assert len(handler._writer.logged_scalars) == expected_count


@pytest.mark.parametrize(
    'writer_type, close_method',
    [
        ('tensorboard', 'close'),
        ('wandb', 'finish'),
    ],
)
def test_close_backend_writer(tmp_path, caplog, writer_type, close_method):
    """Tests the close behavior for different backend writers."""

    class FakeWriter:
        def __init__(self):
            self.closed = False
            self.finished = False

        def close(self):
            self.closed = True

        def finish(self):
            self.finished = True

    log_dir = str(tmp_path / 'logs')
    handler = BackendHandler(
        log_dir=log_dir,
        enable_wandb=writer_type == 'wandb',
        enable_tensorboard=writer_type == 'tensorboard',
        is_master=True,
        logger=logging.getLogger(f'test_close_{writer_type}'),
    )
    writer = FakeWriter()
    handler._writer = writer
    handler._can_log_flag = True
    handler.close()
    assert handler._writer is None
    assert getattr(writer, close_method) is True
