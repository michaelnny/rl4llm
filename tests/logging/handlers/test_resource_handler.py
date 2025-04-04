import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

# Conditional import for psutil testing
try:
    import psutil

    _PSUTIL_AVAILABLE_FOR_TEST = True
except ImportError:
    _PSUTIL_AVAILABLE_FOR_TEST = False

# Import the class to be tested
from rl4llm.logging.handlers.resource_handler import (
    _PSUTIL_AVAILABLE,
    ResourceHandler,
)

# Constants
BYTES_TO_GB = 1024.0**3
DEFAULT_INTERVAL = 0.2  # Use a short interval for testing

# --- Fixtures ---


@pytest.fixture
def mock_dist_manager():
    """Provides a mock DistributedManager."""
    mock = MagicMock()
    # Mock any methods if needed, e.g., mock.is_main_process = True
    return mock


@pytest.fixture
def logger():
    """Provides a basic logger."""
    return logging.getLogger('test_resource_handler')


@pytest.fixture
def resource_handler(mock_dist_manager, logger):
    """Fixture to create and tear down ResourceHandler."""
    handler = ResourceHandler(
        dist_manager=mock_dist_manager,
        logger=logger,
        sampling_interval_seconds=DEFAULT_INTERVAL,
    )
    yield handler
    # Teardown: ensure thread is stopped
    handler.close()


@pytest.fixture
def resource_handler_no_deps(mock_dist_manager, logger):
    """Fixture for ResourceHandler when dependencies are mocked as unavailable."""
    with (
        patch(
            'rl4llm.logging.handlers.resource_handler._PSUTIL_AVAILABLE', False
        ),
        patch('torch.cuda.is_available', return_value=False),
    ):
        handler = ResourceHandler(
            dist_manager=mock_dist_manager,
            logger=logger,
            sampling_interval_seconds=DEFAULT_INTERVAL,
        )
        yield handler
        handler.close()


# --- Test Cases ---


def test_initialization_no_deps(resource_handler_no_deps):
    handler = resource_handler_no_deps
    assert not handler._psutil_initialized
    assert not handler._torch_gpu_available
    assert handler._monitor_thread is None


@pytest.mark.skipif(
    not _PSUTIL_AVAILABLE_FOR_TEST, reason='psutil not installed'
)
def test_initialization_with_psutil(mock_dist_manager, logger):
    # Ensure torch.cuda is seen as unavailable for this test
    with patch('torch.cuda.is_available', return_value=False):
        handler = ResourceHandler(
            dist_manager=mock_dist_manager,
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
def test_initialization_with_cuda(mock_dist_manager, logger):
    # Ensure psutil is seen as unavailable for this test
    with patch(
        'rl4llm.logging.handlers.resource_handler._PSUTIL_AVAILABLE', False
    ):
        # Mock psutil Process init to avoid actual process creation if needed
        with patch(
            'psutil.Process',
            side_effect=ImportError('Simulating psutil not available'),
        ):
            handler = ResourceHandler(
                dist_manager=mock_dist_manager,
                logger=logger,
                sampling_interval_seconds=DEFAULT_INTERVAL,
            )
        # Check actual torch availability
        if torch.cuda.is_available():
            assert not handler._psutil_initialized
            assert handler._torch_gpu_available
            assert handler._device_id == torch.cuda.current_device()
            assert handler._monitor_thread is not None
        else:  # Should not happen due to skipif, but good practice
            pytest.skip('Redundant skip, CUDA became unavailable')

        handler.close()
        assert handler._monitor_thread is None


def test_close_stops_thread(resource_handler):
    # Only run if a thread was actually started
    if (
        resource_handler._monitor_thread
        and resource_handler._monitor_thread.is_alive()
    ):
        initial_thread_id = resource_handler._monitor_thread.ident
        resource_handler.close()
        # Allow some time for the thread to join
        time.sleep(DEFAULT_INTERVAL * 1.5)
        # Check if the specific thread instance is no longer alive
        found = False
        for t in threading.enumerate():
            if t.ident == initial_thread_id:
                found = True
                break
        assert not found
        assert (
            resource_handler._monitor_thread is None
        )  # Should be set to None by close()
    else:
        # If no thread started, close should just do nothing gracefully
        resource_handler.close()
        assert resource_handler._monitor_thread is None


def test_collect_metrics_empty_initially(resource_handler):
    metrics = resource_handler.collect_metrics()
    assert isinstance(metrics, dict)
    assert not metrics  # Should be empty right after init


def test_collect_metrics_after_wait(resource_handler):
    # Only run if monitoring is active
    if (
        not resource_handler._psutil_initialized
        and not resource_handler._torch_gpu_available
    ):
        pytest.skip('Neither psutil nor torch.cuda monitoring is active.')

    # Wait longer than the sampling interval
    time.sleep(DEFAULT_INTERVAL * 2)

    metrics = resource_handler.collect_metrics()
    assert isinstance(metrics, dict)
    assert metrics  # Should have collected *something*

    # Check for expected keys based on initialized components
    if resource_handler._psutil_initialized:
        assert 'resource/process_cpu_percent' in metrics
        assert 'resource/process_ram_rss_gb' in metrics
        assert isinstance(metrics['resource/process_cpu_percent'], list)
        assert len(metrics['resource/process_cpu_percent']) > 0

    if resource_handler._torch_gpu_available:
        dev_id = resource_handler._device_id
        assert f"resource/gpu_{dev_id}_mem_allocated_gb" in metrics
        assert f"resource/gpu_{dev_id}_mem_reserved_gb" in metrics
        assert f"resource/gpu_{dev_id}_mem_used_gb" in metrics
        assert isinstance(
            metrics[f"resource/gpu_{dev_id}_mem_allocated_gb"], list
        )
        assert len(metrics[f"resource/gpu_{dev_id}_mem_allocated_gb"]) > 0


def test_collect_metrics_clears_buffer(resource_handler):
    if (
        not resource_handler._psutil_initialized
        and not resource_handler._torch_gpu_available
    ):
        pytest.skip('Neither psutil nor torch.cuda monitoring is active.')

    time.sleep(DEFAULT_INTERVAL * 2)
    metrics1 = resource_handler.collect_metrics()
    assert metrics1  # Should have collected something

    metrics2 = resource_handler.collect_metrics()
    assert not metrics2  # Should be empty now


def test_collect_metrics_when_no_thread_running(resource_handler_no_deps):
    handler = resource_handler_no_deps
    assert handler._monitor_thread is None
    metrics = handler.collect_metrics()
    assert isinstance(metrics, dict)
    assert not metrics


# Example of how you might test GPU memory reporting more specifically
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='torch.cuda not available'
)
def test_gpu_memory_reporting(resource_handler):
    if not resource_handler._torch_gpu_available:
        pytest.skip('GPU monitoring did not initialize correctly.')

    dev_id = resource_handler._device_id
    key_alloc = f"resource/gpu_{dev_id}_mem_allocated_gb"
    key_reserv = f"resource/gpu_{dev_id}_mem_reserved_gb"
    key_used = f"resource/gpu_{dev_id}_mem_used_gb"

    # Allocate some memory
    tensor_size = 1024 * 1024 * 100  # ~100MB
    try:
        # Use context manager for device selection if needed, though handler uses current_device
        with torch.cuda.device(dev_id):
            initial_alloc = torch.cuda.memory_allocated(dev_id)
            tensor = torch.randn(
                tensor_size, device=f'cuda:{dev_id}', dtype=torch.float32
            )
            alloc_after = torch.cuda.memory_allocated(dev_id)
            assert alloc_after > initial_alloc

        time.sleep(DEFAULT_INTERVAL * 2)  # Wait for collection
        metrics = resource_handler.collect_metrics()

        assert key_alloc in metrics
        assert key_reserv in metrics
        assert key_used in metrics
        assert len(metrics[key_alloc]) > 0

        # Check if the reported allocated memory reflects the allocation (approx)
        # Get the last sample
        last_alloc_gb = metrics[key_alloc][-1]
        expected_alloc_gb = alloc_after / BYTES_TO_GB
        # Use approx due to timing and potential background allocations
        assert last_alloc_gb == pytest.approx(expected_alloc_gb, rel=0.1)

        del tensor  # Free memory
        torch.cuda.empty_cache()  # Clear cache

        time.sleep(DEFAULT_INTERVAL * 2)  # Wait for collection after free
        metrics_after_free = resource_handler.collect_metrics()

        assert key_alloc in metrics_after_free
        last_alloc_gb_after_free = metrics_after_free[key_alloc][-1]
        expected_alloc_gb_after_free = (
            torch.cuda.memory_allocated(dev_id) / BYTES_TO_GB
        )
        # Should be close to the initial allocation before creating the tensor
        assert last_alloc_gb_after_free == pytest.approx(
            expected_alloc_gb_after_free, rel=0.1
        )

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            pytest.skip(
                'CUDA out of memory, cannot run GPU memory test reliably.'
            )
        else:
            raise e
