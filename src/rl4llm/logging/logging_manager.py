"""Centralized logging manager for log training metrics, samples"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml

from rl4llm.constants import (
    EVAL_PHASE,
    LOGGER_NAME,
    LOGGING_PHASES,
    TRAIN_PHASE,
)
from rl4llm.core.distributed import DistributedOps
from rl4llm.logging.handlers import (
    BackendHandler,
    BaseHandler,
    MetricHandler,
    ResourceHandler,
    SampleHandler,
)


def setup_logger(
    rank: int = 0, log_level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    # Prevent messages from bubbling to the root logger
    logger.propagate = False

    # Clear any pre-existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"[rank {rank}] - %(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(log_level)
    return logger


class LoggingManager:
    """
    Coordinates logging across different handlers (Console, Metrics, Samples, Backend).
    Provides a unified API using delegation.
    """

    _log_phases = LOGGING_PHASES
    _train_phase = TRAIN_PHASE
    _eval_phase = EVAL_PHASE

    def __init__(
        self,
        dist_ops: DistributedOps,
        output_dir: str,
        metrics_aggregation_config: Optional[Dict[str, List[str]]] = None,
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        sample_buffer_size: int = 100,
        sample_file_format: str = 'parquet',
        log_level: Optional[str] = None,
    ):
        self.dist_ops = dist_ops
        self.output_dir = output_dir
        self.rank = dist_ops.global_rank
        self.is_master = dist_ops.is_master
        self.world_size = dist_ops.world_size

        # Setup Console Logger
        log_level_str = (
            log_level
            or os.environ.get(
                'LOG_LEVEL', 'INFO' if self.is_master else 'WARNING'
            ).upper()
        )
        self.console_logger = setup_logger(self.rank, log_level_str)
        self.console_logger.info(
            f"LoggingManager initializing on Rank {self.rank}..."
        )

        # Initialize Handlers
        self.metric_handler = MetricHandler(
            dist_ops=self.dist_ops,
            user_aggregation_config=metrics_aggregation_config,
            logger=self.console_logger,
        )
        self.sample_handler = SampleHandler(
            dist_ops=self.dist_ops,
            log_dir=self.output_dir,
            sample_file_format=sample_file_format,
            sample_buffer_size=sample_buffer_size,
            logger=self.console_logger,
        )
        self.backend_handler = BackendHandler(
            log_dir=self.output_dir,
            enable_wandb=enable_wandb,
            enable_tensorboard=enable_tensorboard,
            is_master=self.is_master,
            logger=self.console_logger,
        )
        self.resource_handler = ResourceHandler(
            dist_ops=self.dist_ops,
            logger=self.console_logger,
            sampling_interval_seconds=10.0,
        )

        self._handlers: List[BaseHandler] = [
            self.metric_handler,
            self.sample_handler,
            self.backend_handler,
            self.resource_handler,
        ]

    def log_scalar(
        self, name: str, value: Union[float, int, torch.Tensor]
    ) -> None:
        """Logs a single scalar metric. Aggregation happens later."""
        self.metric_handler.log_scalar(name, value)

    def log_metrics_dict(
        self, metrics: Dict[str, Union[float, int, torch.Tensor]]
    ) -> None:
        """Logs multiple scalar metrics from a dictionary."""
        for name, value in metrics.items():
            self.metric_handler.log_scalar(name, value)

    def log_sample(
        self, phase: str, sample_data: Dict[str, Any], step: int
    ) -> None:
        """
        Logs sample data associated with a specific phase (e.g., 'train', 'eval').
        The phase is used to determine the output file.
        """
        # Phase validation remains crucial here for file routing
        if phase not in self._log_phases:
            # Use warning instead of assert to be less disruptive? Or keep assert?
            # Let's keep assert for now to enforce valid phases for samples.
            raise ValueError(
                f"Invalid phase '{phase}' for log_sample. Must be one of {self._log_phases}"
            )
        self.sample_handler.log_sample(phase, sample_data, step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters to the backend and local file (master only)."""
        if self.is_master:
            try:
                self.backend_handler.log_hyperparams(params)
                config_file_path = os.path.join(self.log_dir, 'job_params.yaml')
                with open(config_file_path, 'w') as f:
                    yaml.dump(params, f, sort_keys=False, indent=2, width=120)
            except Exception:
                self.info(f"Hyperparameters: {params}")

    @contextmanager
    def timer(self, name: str) -> None:
        """Context manager to time a block of code and log the duration."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            time_metric_name = f"time/{name}"
            self.metric_handler.log_scalar(time_metric_name, elapsed)
            self.debug(f"Timed [{name}]: {elapsed:.4f}s")

    def aggregate_and_log(self, step: int) -> None:
        """
        Aggregates metrics across all ranks, logs them to the backend (master only),
        logs them to the console (master only), flushes samples, and clears metric buffers.
        """
        self.debug(f"Starting aggregation and logging for step {step}...")

        try:
            # Collect raw resource metric samples (returns Dict[str, List[values]])
            resource_metric_samples = self.resource_handler.collect_metrics()

            # Log each individual sample using the MetricHandler
            if resource_metric_samples:
                self.debug(
                    f"Logging {sum(len(v) for v in resource_metric_samples.values())} individual resource samples to MetricHandler..."
                )
                for name, samples_list in resource_metric_samples.items():
                    for sample_value in samples_list:
                        self.metric_handler.log_scalar(name, sample_value)
            else:
                self.debug('No resource metric samples collected this step.')

        except Exception as e:
            # Catch errors during collection/logging to prevent crashing the whole step
            self.error(
                f"Error during resource collection/logging for step {step}: {e}",
                exc_info=True,
            )

        aggregated_metrics = self.metric_handler.aggregate()

        # Logging to backend and console (master only)
        if self.is_master:
            if aggregated_metrics:
                self.backend_handler.log_metrics(aggregated_metrics, step)

            self._log_aggregated_metrics_to_console(aggregated_metrics, step)

        # Clear buffers on all ranks
        self.metric_handler.clear_buffer()
        self.sample_handler.flush()
        self.debug(f"Finished aggregation and logging for step {step}.")

    def _log_aggregated_metrics_to_console(
        self, aggregated_metrics: Dict[str, float], step: int
    ):
        """
        Formats and logs ALL aggregated metrics to the console on master,
        without phase-based categorization. Metrics are sorted alphabetically.
        """
        if not self.is_master or not aggregated_metrics:
            return

        # Sort metrics alphabetically by name for consistent output
        sorted_items = sorted(aggregated_metrics.items())

        # Format each metric
        items_str = ' | '.join(
            [
                (
                    f"{n}: {v:.4f}"
                    # Check for NaN explicitly before formatting as float
                    if isinstance(v, (float, np.floating)) and not np.isnan(v)
                    else f"{n}: {v}"  # Keep non-float or NaN as is
                )
                for n, v in sorted_items
            ]
        )

        if items_str:
            self.info(f"Step: {step} | Aggregated Metrics | [ {items_str} ]")
        else:
            # This case should be rare if aggregated_metrics is not empty,
            # but good to handle.
            self.info(f"Step: {step} | No metrics aggregated or logged.")

    def flush(self) -> None:
        """Flushes handlers that support it (currently SampleHandler for files)."""
        self.debug('Flushing LoggingManager resources...')
        # Only SampleHandler has an explicit flush for file buffers now
        if hasattr(self.sample_handler, 'flush'):
            self.sample_handler.flush()

    def close(self) -> None:
        """Closes all registered handlers."""
        self.info(f"Closing LoggingManager on Rank {self.rank}...")

        for handler in reversed(self._handlers):
            try:
                handler.close()
            except Exception as e:
                self.error(
                    f"Error closing handler {type(handler).__name__}: {e}"
                )
        self.dist_ops.barrier()
        self.info(f"LoggingManager closed on Rank {self.rank}.")

    def info(self, message: str, **kwargs) -> None:
        if self.is_master:
            self.console_logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.console_logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.console_logger.error(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        self.console_logger.debug(message, **kwargs)
