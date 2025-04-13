import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import psutil

    _PSUTIL_AVAILABLE_FOR_TEST = True
except ImportError:
    _PSUTIL_AVAILABLE_FOR_TEST = False

from rl4llm.logging.handlers.resource_handler import (
    _PSUTIL_AVAILABLE,
    ResourceHandler,
)

BYTES_TO_GB = 1024.0**3
DEFAULT_INTERVAL = 0.2

# --- Fixtures ---


@pytest.fixture
def mock_dist_ops():
    """Returns a mock DistributedOps."""
    return MagicMock()


@pytest.fixture
def logger():
    """Returns a basic test logger."""
    return logging.getLogger('test_resource_handler')


@pytest.fixture
def resource_handler(mock_dist_ops, logger):
    """Yields a ResourceHandler with monitoring dependencies enabled."""
    handler = ResourceHandler(
        dist_ops=mock_dist_ops,
        logger=logger,
        sampling_interval_seconds=DEFAULT_INTERVAL,
    )
    yield handler
    handler.close()


@pytest.fixture
def resource_handler_no_deps(mock_dist_ops, logger):
    """Yields a ResourceHandler with psutil and CUDA mocked as unavailable."""
    with (
        patch(
            'rl4llm.logging.handlers.resource_handler._PSUTIL_AVAILABLE', False
        ),
        patch('torch.cuda.is_available', return_value=False),
    ):
        handler = ResourceHandler(
            dist_ops=mock_dist_ops,
            logger=logger,
            sampling_interval_seconds=DEFAULT_INTERVAL,
        )
        yield handler
        handler.close()


# --- Initialization Tests ---


def test_initialization_no_deps(resource_handler_no_deps):
    """Tests handler initialization when psutil and torch.cuda are unavailable."""
    handler = resource_handler_no_deps
    assert not handler._psutil_initialized
    assert not handler._torch_gpu_available
    assert handler._monitor_thread is None


@pytest.mark.skipif(
    not _PSUTIL_AVAILABLE_FOR_TEST, reason='psutil not installed'
)
def test_initialization_with_psutil(mock_dist_ops, logger):
    """Tests psutil-based initialization of the handler."""
    with patch('torch.cuda.is_available', return_value=False):
        handler = ResourceHandler(
            dist_ops=mock_dist_ops,
            logger=logger,
            sampling_interval_seconds=DEFAULT_INTERVAL,
        )
        assert handler._psutil_initialized
        assert not handler._torch_gpu_available
        assert handler._monitor_thread is not None
        assert handler._monitor_thread.is_alive()
        handler.close()
        assert handler._monitor_thread is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='torch.cuda not available'
)
def test_initialization_with_cuda(mock_dist_ops, logger):
    """Tests torch.cuda-based initialization of the handler."""
    with patch(
        'rl4llm.logging.handlers.resource_handler._PSUTIL_AVAILABLE', False
    ):
        with patch(
            'psutil.Process',
            side_effect=ImportError('Simulating psutil not available'),
        ):
            handler = ResourceHandler(
                dist_ops=mock_dist_ops,
                logger=logger,
                sampling_interval_seconds=DEFAULT_INTERVAL,
            )
    if torch.cuda.is_available():
        assert not handler._psutil_initialized
        assert handler._torch_gpu_available
        assert handler._device_id == torch.cuda.current_device()
        assert handler._monitor_thread is not None
    else:
        pytest.skip('Redundant skip, CUDA became unavailable')
    handler.close()
    assert handler._monitor_thread is None


# --- Runtime Behavior Tests ---


def test_close_stops_thread(resource_handler):
    """Ensures that the monitoring thread is properly stopped by close()."""
    if (
        resource_handler._monitor_thread
        and resource_handler._monitor_thread.is_alive()
    ):
        thread_id = resource_handler._monitor_thread.ident
        resource_handler.close()
        time.sleep(DEFAULT_INTERVAL * 1.5)
        assert not any(t.ident == thread_id for t in threading.enumerate())
        assert resource_handler._monitor_thread is None
    else:
        resource_handler.close()
        assert resource_handler._monitor_thread is None


# --- Metric Collection Tests ---


def test_collect_metrics_empty_initially(resource_handler):
    """Checks that metrics are initially empty after creation."""
    metrics = resource_handler.collect_metrics()
    assert isinstance(metrics, dict)
    assert not metrics


def test_collect_metrics_when_no_thread_running(resource_handler_no_deps):
    """Collecting metrics should return empty dict when thread isn't running."""
    metrics = resource_handler_no_deps.collect_metrics()
    assert isinstance(metrics, dict)
    assert not metrics


def test_collect_metrics_after_wait(resource_handler):
    """Tests metrics are collected after waiting beyond sampling interval."""
    if not (
        resource_handler._psutil_initialized
        or resource_handler._torch_gpu_available
    ):
        pytest.skip('No monitoring backend available.')

    time.sleep(DEFAULT_INTERVAL * 2)
    metrics = resource_handler.collect_metrics()
    assert isinstance(metrics, dict)
    assert metrics

    if resource_handler._psutil_initialized:
        assert 'resource/process_cpu_percent' in metrics
        assert isinstance(metrics['resource/process_cpu_percent'], list)
        assert metrics['resource/process_cpu_percent']

    if resource_handler._torch_gpu_available:
        dev_id = resource_handler._device_id
        key = f"resource/gpu_{dev_id}_mem_allocated_gb"
        assert key in metrics
        assert isinstance(metrics[key], list)
        assert metrics[key]


def test_collect_metrics_clears_buffer(resource_handler):
    """Ensures collect_metrics clears internal buffers after collection."""
    if not (
        resource_handler._psutil_initialized
        or resource_handler._torch_gpu_available
    ):
        pytest.skip('No monitoring backend available.')

    time.sleep(DEFAULT_INTERVAL * 2)
    assert resource_handler.collect_metrics()
    assert not resource_handler.collect_metrics()


# --- GPU Specific Test ---


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='torch.cuda not available'
)
def test_gpu_memory_reporting(resource_handler):
    """Tests GPU memory metrics reflect actual GPU allocations."""
    if not resource_handler._torch_gpu_available:
        pytest.skip('GPU monitoring did not initialize correctly.')

    dev_id = resource_handler._device_id
    key_alloc = f"resource/gpu_{dev_id}_mem_allocated_gb"
    key_reserv = f"resource/gpu_{dev_id}_mem_reserved_gb"
    key_used = f"resource/gpu_{dev_id}_mem_used_gb"

    tensor_size = 1024 * 1024 * 100
    try:
        with torch.cuda.device(dev_id):
            initial_alloc = torch.cuda.memory_allocated(dev_id)
            tensor = torch.randn(
                tensor_size, device=f'cuda:{dev_id}', dtype=torch.float32
            )
            alloc_after = torch.cuda.memory_allocated(dev_id)
            assert alloc_after > initial_alloc

        time.sleep(DEFAULT_INTERVAL * 2)
        metrics = resource_handler.collect_metrics()

        for key in [key_alloc, key_reserv, key_used]:
            assert key in metrics
            assert metrics[key]

        last_alloc_gb = metrics[key_alloc][-1]
        expected_alloc_gb = alloc_after / BYTES_TO_GB
        assert last_alloc_gb == pytest.approx(expected_alloc_gb, rel=0.1)

        del tensor
        torch.cuda.empty_cache()
        time.sleep(DEFAULT_INTERVAL * 2)

        metrics_after = resource_handler.collect_metrics()
        last_after = metrics_after[key_alloc][-1]
        expected_after = torch.cuda.memory_allocated(dev_id) / BYTES_TO_GB
        assert last_after == pytest.approx(expected_after, rel=0.1)

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            pytest.skip(
                'CUDA out of memory, cannot run GPU memory test reliably.'
            )
        raise
