# rl4llm/logging/manager.py

import html
import json
import logging
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from rl4llm.constants import (
    EVAL_PHASE,
    LOGGER_NAME,
    LOGGING_PHASES,
    TRAIN_PHASE,
)
from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers import (
    BackendHandler,
    BaseHandler,
    MetricHandler,
    SampleHandler,
)


def setup_logger(rank: int = 0, log_level=logging.INFO) -> logging.Logger:
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
        dist_manager: DistributedManager,
        log_dir: str,
        metrics_aggregation_config: Optional[Dict[str, List[str]]] = None,
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        sample_buffer_size: int = 100,
        sample_file_format: str = 'parquet',
        log_level: Optional[str] = None,
    ):
        self.dist_manager = dist_manager
        self.log_dir = log_dir
        self.rank = dist_manager.global_rank
        self.is_master = dist_manager.is_master
        self.world_size = dist_manager.world_size

        # 1. Setup Console Logger
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

        # 2. Initialize Handlers (pass logger)
        self.metric_handler = MetricHandler(
            dist_manager=self.dist_manager,
            user_aggregation_config=metrics_aggregation_config,
            logger=self.console_logger,
        )
        self.sample_handler = SampleHandler(
            dist_manager=self.dist_manager,
            log_dir=self.log_dir,  # Base log dir
            sample_file_format=sample_file_format,
            sample_buffer_size=sample_buffer_size,
            logger=self.console_logger,
        )
        # BackendHandler now creates its own writer internally
        self.backend_handler = BackendHandler(
            log_dir=self.log_dir,
            enable_wandb=enable_wandb,
            enable_tensorboard=enable_tensorboard,
            is_master=self.is_master,
            logger=self.console_logger,
        )

        # Store handlers for easier management (e.g., closing)
        self._handlers: List[BaseHandler] = [
            self.metric_handler,
            self.sample_handler,
            self.backend_handler,
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
        """Logs hyperparameters to the backend and console (master only)."""
        if self.is_master:  # Still log to console on master
            try:
                self.backend_handler.log_hyperparams(params)
                params_str = yaml.dump(
                    params, sort_keys=False, indent=2, width=120
                )
                self.info(f"Hyperparameters:\n---\n{params_str.strip()}\n---")
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
        # self.flush() # Flush might be called within handler's close if needed
        for handler in reversed(
            self._handlers
        ):  # Close in reverse order? (Backend last?)
            try:
                handler.close()
            except Exception as e:
                self.error(
                    f"Error closing handler {type(handler).__name__}: {e}"
                )
        if self.world_size > 1:
            self.dist_manager.barrier()  # Final barrier after all cleanup
        self.info(f"LoggingManager closed on Rank {self.rank}.")

    # --- Console Logging Passthrough (Unchanged) ---
    def info(self, message: str) -> None:
        if self.is_master:
            self.console_logger.info(message)

    def warning(self, message: str) -> None:
        self.console_logger.warning(message)

    def error(self, message: str) -> None:
        self.console_logger.error(message)

    def debug(self, message: str) -> None:
        self.console_logger.debug(message)


# --- Testing Code (Adjust BackendHandler instantiation) ---
if __name__ == '__main__':
    # Keep MockConfig and MockDistributedManager as they were in the previous refactor

    class MockConfig:  # Ensure it has needed attrs for BackendHandler
        project_name = 'test_logging_final'
        learning_rate = 1e-4
        batch_size = 8
        env_type = 'mock'
        seed = 42
        run_name = f"test_run_{time.strftime('%H%M%S')}"  # Example run name
        run_id = None  # Let WandB generate or set manually if needed

        # Make it dict-like
        def __iter__(self):
            yield from vars(self).items()

        def items(self):
            return self.__iter__()

        def keys(self):
            return vars(self).keys()

        def __getitem__(self, key):
            return getattr(self, key)

    # MockDistributedManager as before...
    class MockDistributedManager:  # Simplified mock focusing on core needs
        def __init__(self, rank=0, world_size=1):
            self._rank = rank
            self._world_size = world_size
            self.logger = setup_logger(self._rank, 'DEBUG')
            self.logger = logging.LoggerAdapter(
                self.logger, {'rank': self.global_rank}
            )
            self._device = torch.device('cpu')  # Mock device
            self.logger.info(
                f"[MockDist] Rank={rank}, WorldSize={world_size}, Device={self._device}"
            )

        @property
        def device(self):
            return self._device

        @property
        def global_rank(self):
            return self._rank

        @property
        def world_size(self):
            return self._world_size

        @property
        def is_master(self):
            return self._rank == 0

        def barrier(self):
            time.sleep(0.001)

        def gather_object(self, obj: Any, dst: int = 0) -> Optional[List[Any]]:
            if self.global_rank == dst:
                return [
                    obj for _ in range(self.world_size)
                ]  # Simulate receiving from all
            return None

        def all_gather_object(self, obj: Any) -> List[Any]:
            return [obj for _ in range(self.world_size)]

        def teardown(self):
            self.logger.info('[MockDist] Teardown.')

    print('--- Starting Final Refactored LoggingManager Test ---')

    rank = 0
    world_size = 1
    dist_manager_mock = MockDistributedManager(rank=rank, world_size=world_size)
    config = MockConfig()
    log_dir = './test_runs/single_process'

    if rank == 0 and os.path.exists(log_dir):
        import shutil

        shutil.rmtree(log_dir)
    # No need to create log_dir here, handlers might do it if needed (e.g. BackendHandler)

    logger = LoggingManager(
        config,
        dist_manager_mock,
        log_dir,  # Pass base log dir
        enable_wandb=False,
        enable_tensorboard=True,
        sample_file_format='jsonl',
        log_sample_interval=2,
        max_backend_samples=3,
        log_level='DEBUG',
    )

    # --- Training Loop Simulation (identical to previous refactor test) ---
    num_steps = 5
    global_step_counter = 0
    for step in range(num_steps):
        logger.debug(
            f"--- Start Step {step} (Global: {global_step_counter}) ---"
        )
        with logger.train_scope():
            loss = 1.0 / (global_step_counter + 1) + random.random() * 0.1
            accuracy = (
                0.8 + (global_step_counter * 0.01) + random.random() * 0.01
            )
            lr = config.learning_rate * (0.9**step)
            logger.log_scalar('loss', loss)
            logger.log_metrics_dict({'accuracy': accuracy, 'learning_rate': lr})
            for i in range(3):
                logger.log_sample(
                    'generation',
                    {
                        'prompt': f"TrP{global_step_counter}_{i}",
                        'resp': f"TrR{global_step_counter}_{i}",
                        'score': accuracy * 10 + i,
                    },
                    global_step_counter,
                )
            with logger.timer('train_batch_time'):
                time.sleep(0.01 + random.random() * 0.01)

        if step % 2 == 0:
            logger.info(f"--- Running Eval at Step {step} ---")
            with logger.eval_scope():
                eval_loss = loss * 1.1 + random.random() * 0.05
                eval_perp = np.exp(eval_loss)
                eval_rew = random.uniform(5, 10)
                logger.log_scalar('loss', eval_loss)
                logger.log_metrics_dict(
                    {
                        'perplexity': eval_perp,
                        'reward': eval_rew,
                        f"reward_cls_{random.randint(0, 1)}": eval_rew
                        + random.random(),
                    }
                )
                for i in range(2):
                    logger.log_sample(
                        'validation',
                        {
                            'prompt': f"EvP{global_step_counter}_{i}",
                            'resp': f"EvR{global_step_counter}_{i}",
                        },
                        global_step_counter,
                    )
                with logger.timer('eval_total_time'):
                    time.sleep(0.02 + random.random() * 0.01)

        logger.log_scalar('buffer_size', random.randint(100, 200))
        logger.aggregate_and_log(global_step_counter)
        logger.debug(f"--- End Step {step} (Global: {global_step_counter}) ---")
        global_step_counter += 1

    logger.close()
    print(f"[Rank {rank}] Finished final refactored test.")
    print(f"Logs saved in: {os.path.abspath(log_dir)}")
    print(
        "Check subdirectories: 'wandb' or 'tensorboard', 'train/samples', 'eval/samples', 'general/samples'"
    )

    print('\n--- Final Refactored LoggingManager Test Complete ---')
