import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Final, List, Optional, Union

import torch

from rl4llm.core.distributed import DistributedOps  # Assuming this exists
from rl4llm.logging.handlers.base_handler import BaseHandler

# Conditional import for psutil
try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

# Constant for conversion
BYTES_TO_GB: Final[float] = 1024.0**3


class ResourceHandler(BaseHandler):
    """
    Collects essential system (CPU, RAM) and GPU (PyTorch CUDA) resource usage
    metrics in a background thread. Reports memory in Gigabytes (GB).

    Collected Metrics:
    - resource/process_cpu_percent: CPU utilization percentage for the current process.
    - resource/process_ram_rss_gb: Resident Set Size (physical memory) used by the process (GB).
    - resource/gpu_X_mem_allocated_gb: Memory allocated for tensors on GPU X by PyTorch (GB).
    - resource/gpu_X_mem_reserved_gb: Total memory managed by PyTorch's caching allocator on GPU X (GB).
    - resource/gpu_X_mem_used_gb: Total memory used on GPU X by all processes (GB).
    (where X is the GPU device ID)
    """

    def __init__(
        self,
        dist_ops: DistributedOps,
        logger: Optional[logging.Logger] = None,
        sampling_interval_seconds: float = 10.0,
    ):
        super().__init__(logger)
        self.dist_ops = dist_ops
        self._process: Optional[psutil.Process] = None
        self._psutil_initialized: bool = False
        self._torch_gpu_available: bool = False
        self._device_id: Optional[int] = None

        self._sampling_interval: float = sampling_interval_seconds
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()

        # Store collected raw samples between calls to collect_metrics
        self._collected_samples: Dict[str, List[Union[float, int]]] = (
            defaultdict(list)
        )

        self._initialize_psutil()
        self._initialize_torch_gpu()  # Renamed for clarity

        self._logger.info(
            f"ResourceHandler: psutil available: {_PSUTIL_AVAILABLE}, "
            f"initialized: {self._psutil_initialized}"
        )
        self._logger.info(
            f"ResourceHandler: torch.cuda available: {torch.cuda.is_available()}, "
            f"monitoring enabled: {self._torch_gpu_available}"
            f"{f', device_id={self._device_id}' if self._torch_gpu_available else ''}"
        )

        # Start thread if *either* psutil or torch.cuda monitoring is active
        if self._psutil_initialized or self._torch_gpu_available:
            self._start_monitoring_thread()
        else:
            self._logger.warning(
                'ResourceHandler: Neither psutil nor torch.cuda monitoring is active. '
                'Background thread not started.'
            )

    def _initialize_psutil(self):
        """Initialize CPU and RAM monitoring tools using psutil."""
        if not _PSUTIL_AVAILABLE:
            self._logger.warning(
                'psutil library not found. CPU/RAM monitoring disabled.'
            )
            return
        try:
            self._process = psutil.Process(os.getpid())
            # Call cpu_percent once to initialize measurement interval correctly for subsequent calls
            self._process.cpu_percent(interval=None)
            self._psutil_initialized = True
            self._logger.info(
                'psutil initialized successfully for CPU/RAM monitoring.'
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
            self._logger.warning(
                f"Failed to initialize psutil (pid: {os.getpid()}): {e}. "
                'CPU/RAM monitoring disabled.',
                exc_info=True,
            )
            self._process = None
            self._psutil_initialized = False
        except Exception as e:
            self._logger.error(
                f"Unexpected error initializing psutil: {e}", exc_info=True
            )
            self._process = None
            self._psutil_initialized = False

    def _initialize_torch_gpu(self):
        """Check for torch CUDA availability and get current device for GPU monitoring."""
        if torch.cuda.is_available():
            try:
                # Assume the main script/framework (like Accelerate/DDP)
                # has set the correct device for this process/rank.
                self._device_id = torch.cuda.current_device()
                # Perform a small CUDA operation to ensure context is initialized and device is valid
                _ = torch.cuda.get_device_name(self._device_id)
                # Check memory info works
                _ = torch.cuda.mem_get_info(self._device_id)
                self._torch_gpu_available = True
                self._logger.info(
                    f"torch.cuda initialized successfully for GPU monitoring on device {self._device_id}."
                )
            except RuntimeError as e:
                self._logger.warning(
                    f"torch.cuda is available but failed to initialize or get device info "
                    f"(device {self._device_id}): {e}. GPU monitoring disabled.",
                    exc_info=True,
                )
                self._torch_gpu_available = False
                self._device_id = None
            except Exception as e:
                self._logger.error(
                    f"Unexpected error initializing torch.cuda monitoring: {e}",
                    exc_info=True,
                )
                self._torch_gpu_available = False
                self._device_id = None
        else:
            self._logger.info(
                'torch.cuda not available. GPU monitoring disabled.'
            )
            self._torch_gpu_available = False
            self._device_id = None

    def _start_monitoring_thread(self):
        """Starts the background monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._logger.warning('Monitoring thread already running.')
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name='ResourceMonitorThread'
        )
        self._monitor_thread.start()
        self._logger.info(
            f"Resource monitoring thread started with interval {self._sampling_interval:.1f}s."
        )

    def _monitor_loop(self):
        """Background loop collecting resource samples at defined intervals."""
        self._logger.info('Resource monitoring loop started.')
        last_sample_time = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            wait_time = self._sampling_interval - (now - last_sample_time)
            if wait_time > 0:
                # Wait accurately until the next sampling time
                self._stop_event.wait(wait_time)
                if self._stop_event.is_set():  # Check again after wait
                    break
            last_sample_time = (
                time.monotonic()
            )  # Record actual sample time start

            instant_metrics = {}
            collection_successful = False

            # --- Collect CPU/RAM Metrics (psutil) ---
            if self._psutil_initialized and self._process:
                try:
                    # Ensure process still exists and we have permissions
                    if not self._process.is_running():
                        self._logger.warning(
                            'Monitored process is no longer running. Stopping psutil monitoring.'
                        )
                        self._psutil_initialized = False
                    else:
                        # CPU Usage (percent since last call or averaged over interval if first call)
                        cpu_percent = self._process.cpu_percent(interval=None)
                        instant_metrics['resource/process_cpu_percent'] = (
                            cpu_percent
                        )

                        # RAM Usage (Resident Set Size in GB)
                        mem_info = self._process.memory_info()
                        ram_rss_gb = mem_info.rss / BYTES_TO_GB
                        instant_metrics['resource/process_ram_rss_gb'] = (
                            ram_rss_gb
                        )
                        collection_successful = True

                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self._logger.warning(
                        f"psutil error accessing process info: {e}. Disabling psutil monitoring."
                    )
                    self._psutil_initialized = (
                        False  # Stop trying if process died or permissions lost
                    )
                except Exception as e:
                    self._logger.error(
                        f"Unexpected psutil error in monitor loop: {e}",
                        exc_info=True,
                    )
                    # Optionally disable psutil here too, or let it retry next cycle
                    # self._psutil_initialized = False

            # --- Collect GPU Metrics (torch.cuda) ---
            if self._torch_gpu_available and self._device_id is not None:
                try:
                    # Quick check if device is still valid (optional, mem_get_info usually catches issues)
                    # torch.cuda.get_device_name(self._device_id)

                    # PyTorch Tensor Memory (Allocated)
                    mem_allocated_bytes = torch.cuda.memory_allocated(
                        self._device_id
                    )
                    instant_metrics[
                        f"resource/gpu_{self._device_id}_mem_allocated_gb"
                    ] = (mem_allocated_bytes / BYTES_TO_GB)

                    # PyTorch Allocator Memory (Reserved)
                    mem_reserved_bytes = torch.cuda.memory_reserved(
                        self._device_id
                    )
                    instant_metrics[
                        f"resource/gpu_{self._device_id}_mem_reserved_gb"
                    ] = (mem_reserved_bytes / BYTES_TO_GB)

                    # Device-wide Memory Usage (Used/Total)
                    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(
                        self._device_id
                    )
                    used_mem_bytes = total_mem_bytes - free_mem_bytes

                    instant_metrics[
                        f"resource/gpu_{self._device_id}_mem_used_gb"
                    ] = (used_mem_bytes / BYTES_TO_GB)

                    collection_successful = True

                except RuntimeError as e:
                    # Catch CUDA errors (e.g., context lost, OOM during check, invalid device)
                    self._logger.warning(
                        f"torch.cuda runtime error for device {self._device_id}: {e}. "
                        'Disabling GPU monitoring.',
                        exc_info=False,  # Keep log concise
                    )
                    self._torch_gpu_available = (
                        False  # Stop trying if CUDA fails persistently
                    )
                except Exception as e:
                    self._logger.error(
                        f"Unexpected torch.cuda error for device {self._device_id} in monitor loop: {e}",
                        exc_info=True,
                    )
                    # Optionally disable GPU monitoring here too
                    # self._torch_gpu_available = False

            # --- Store Collected Metrics ---
            if instant_metrics:
                with self._lock:
                    for name, value in instant_metrics.items():
                        self._collected_samples[name].append(value)

            # --- Check if monitoring is still possible ---
            if not self._psutil_initialized and not self._torch_gpu_available:
                self._logger.warning(
                    'Both psutil and torch.cuda monitoring are disabled or failed. '
                    'Stopping resource monitor thread.'
                )
                break  # Exit the loop

            # --- Wait handled at the start of the loop ---

        self._logger.info('Resource monitoring loop finished.')

    def collect_metrics(self) -> Dict[str, List[Union[float, int]]]:
        """
        Returns all raw samples collected by the background thread
        since the last call and clears the internal sample buffer.

        Returns:
            A dictionary where keys are metric names (e.g., 'resource/process_cpu_percent')
            and values are lists of raw float/int samples collected. Returns an empty
            dict if the monitoring thread is not running or no samples were collected.
        """
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            # Check if thread should be running but isn't
            if not self._stop_event.is_set() and (
                self._psutil_initialized or self._torch_gpu_available
            ):
                self._logger.warning(
                    'collect_metrics called but monitoring thread is not running unexpectedly.'
                )
            return {}  # Return empty dict if thread not running

        samples_to_return: Dict[str, List[Union[float, int]]] = {}
        with self._lock:
            # Swap buffers efficiently only if there's something to swap
            if self._collected_samples:
                samples_to_return = self._collected_samples
                self._collected_samples = defaultdict(list)

        # Optional: Log number of samples collected for debugging
        # total_samples = sum(len(v) for v in samples_to_return.values())
        # if total_samples > 0:
        #    self._logger.debug(f"Collected {total_samples} resource metric samples.")

        return (
            samples_to_return  # Return standard dict (or empty if no samples)
        )

    def close(self) -> None:
        """Signals the monitoring thread to stop and waits for it to exit."""
        self._logger.debug('Closing ResourceHandler.')
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._logger.info('Stopping resource monitoring thread...')
            self._stop_event.set()
            # Wait for the thread to finish, give it a bit longer than the sampling interval
            join_timeout = self._sampling_interval + 5.0
            self._monitor_thread.join(timeout=join_timeout)
            if self._monitor_thread.is_alive():
                self._logger.warning(
                    f"Resource monitoring thread did not stop gracefully within {join_timeout}s."
                )
            else:
                self._logger.info('Resource monitoring thread stopped.')
        else:
            self._logger.info(
                'Resource monitoring thread was not running or already stopped.'
            )

        self._monitor_thread = None
        # Reset state flags for clarity, although object is likely being destroyed
        self._psutil_initialized = False
        self._torch_gpu_available = False
        self._process = None
        self._device_id = None
        self._logger.info('ResourceHandler closed.')
