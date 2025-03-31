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


from rl4llm.constants import LOGGER_NAME
from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers import (
    BackendHandler,
    BaseHandler,
    MetricHandler,
    SampleHandler,
)


def setup_logger(rank: int = 0, log_level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False  # Prevent messages from bubbling to the root logger

    # Clear any pre-existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"%(asctime)s - %(levelname)s - [rank {rank}] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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

    TRAIN = "train"
    EVAL = "eval"

    def __init__(
        self,
        config: Any,
        dist_manager: DistributedManager,
        log_dir: str,
        metrics_aggregation_config: Optional[Dict[str, List[str]]] = None,
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        log_sample_interval: int = 50,
        max_backend_samples: int = 10,
        sample_buffer_size: int = 100,
        sample_file_format: str = "parquet",
        log_level: Optional[str] = None,
    ):
        self.config = config
        self.dist_manager = dist_manager
        self.log_dir = log_dir
        self.rank = dist_manager.global_rank
        self.is_master = dist_manager.is_master
        self.world_size = dist_manager.world_size

        # 1. Setup Console Logger
        log_level_str = (
            log_level
            or os.environ.get(
                "LOG_LEVEL", "INFO" if self.is_master else "WARNING"
            ).upper()
        )
        self.console_logger = setup_logger(self.rank, log_level_str)
        self.console_logger.info(f"LoggingManager initializing on Rank {self.rank}...")

        # 2. Initialize Handlers (pass logger)
        self.metric_handler = MetricHandler(
            dist_manager=self.dist_manager,
            user_aggregation_config=metrics_aggregation_config,
            logger=self.console_logger,
        )
        self.sample_handler = SampleHandler(
            dist_manager=self.dist_manager,
            log_dir=self.log_dir,  # Base log dir
            phases=[self.TRAIN, self.EVAL],
            sample_file_format=sample_file_format,
            sample_buffer_size=sample_buffer_size,
            log_sample_interval=log_sample_interval,
            max_backend_samples=max_backend_samples,
            logger=self.console_logger,
        )
        # BackendHandler now creates its own writer internally
        self.backend_handler = BackendHandler(
            log_dir=self.log_dir,
            config=self.config,
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

        # Phase Management State
        self._current_phase: Optional[str] = None
        self._phase_stack: List[Optional[str]] = []

        # Log hyperparameters via BackendHandler
        self.log_hyperparams(
            vars(config) if hasattr(config, "__dict__") else dict(config)
        )

    # --- Phase Management (Unchanged) ---
    @contextmanager
    def _phase_scope(self, phase: str):
        if phase not in [self.TRAIN, self.EVAL]:
            raise ValueError(f"Unsupported scope: '{phase}'")
        self._phase_stack.append(self._current_phase)
        self._current_phase = phase
        self.debug(f"Entering logging phase: {phase}")
        try:
            yield
        finally:
            self.debug(f"Exiting logging phase: {self._current_phase}")
            self._current_phase = self._phase_stack.pop()

    def train_scope(self):
        return self._phase_scope(self.TRAIN)

    def eval_scope(self):
        return self._phase_scope(self.EVAL)

    def _get_current_phase(self) -> Optional[str]:
        return self._current_phase

    def log_scalar(self, name: str, value: Union[float, int, torch.Tensor]) -> None:
        phase = self._get_current_phase()
        metric_key = f"{phase}/{name}" if phase else name
        self.metric_handler.log_scalar(metric_key, value)

    def log_metrics_dict(
        self, metrics: Dict[str, Union[float, int, torch.Tensor]]
    ) -> None:
        phase = self._get_current_phase()
        for name, value in metrics.items():
            metric_key = f"{phase}/{name}" if phase else name
            self.metric_handler.log_scalar(metric_key, value)

    def log_sample(self, tag: str, sample_data: Dict[str, Any], step: int) -> None:
        phase = self._get_current_phase()
        self.sample_handler.log_sample(tag, sample_data, step, phase)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        self.backend_handler.log_hyperparams(params)  # Delegate to backend handler
        if self.is_master:  # Still log to console on master
            try:
                params_str = yaml.dump(params, sort_keys=False, indent=2, width=120)
                self.info(f"Hyperparameters:\n---\n{params_str.strip()}\n---")
            except Exception:
                self.info(f"Hyperparameters: {params}")

    @contextmanager
    def timer(self, name: str) -> None:
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            time_metric_name = f"time/{name}_sec"
            self.metric_handler.log_scalar(time_metric_name, elapsed)  # Delegate
            self.debug(f"Timed [{name}]: {elapsed:.4f}s")

    # --- Aggregation and Flushing (Minor adjustments) ---
    def aggregate_and_log(self, step: int) -> None:
        self.debug(f"Starting aggregation and logging for step {step}...")
        aggregated_metrics = self.metric_handler.aggregate()
        samples_for_backend = self.sample_handler.collect_backend_samples()

        # Logging to backend and console (master only)
        if self.is_master:
            if aggregated_metrics:
                self.backend_handler.log_metrics(aggregated_metrics, step)
            if samples_for_backend:
                self.info(
                    f"Logging {len(samples_for_backend)} samples to backend for step {step}"
                )
                for tag, sample_dict in samples_for_backend:
                    sample_step = sample_dict.get("step", step)
                    formatted_text = self._format_sample_for_backend(
                        sample_dict
                    )  # Use helper
                    self.backend_handler.log_sample_text(
                        tag, formatted_text, sample_step
                    )
            self._log_aggregated_metrics_to_console(
                aggregated_metrics, step
            )  # Use helper

        # Clear buffers on all ranks
        self.metric_handler.clear_buffer()
        self.sample_handler.clear_backend_buffer()
        self.debug(f"Finished aggregation and logging for step {step}.")

    # --- Helper methods (_log_aggregated_metrics_to_console, _format_sample_for_backend - Unchanged) ---
    def _log_aggregated_metrics_to_console(
        self, aggregated_metrics: Dict[str, float], step: int
    ):
        """Formats and logs aggregated metrics to the console on master."""
        if not self.is_master or not aggregated_metrics:
            return
        metrics_by_category = defaultdict(dict)
        known_categories = [self.TRAIN, self.EVAL, "time", SampleHandler.GENERAL_PHASE]
        for k, v in aggregated_metrics.items():
            found_category = False
            for cat in known_categories:
                if k.startswith(f"{cat}/"):
                    metric_name = k[len(cat) + 1 :]
                    metrics_by_category[cat][metric_name] = v
                    found_category = True
                    break
            if not found_category:
                metrics_by_category[SampleHandler.GENERAL_PHASE][k] = v
        output_lines = []
        ordered_categories = [
            self.TRAIN,
            self.EVAL,
            "time",
            SampleHandler.GENERAL_PHASE,
        ]
        for category in ordered_categories:
            if category in metrics_by_category:
                cat_metrics = metrics_by_category[category]
                items_str = " | ".join(
                    [
                        f"{n}: {v:.4f}"
                        if isinstance(v, float) and not np.isnan(v)
                        else f"{n}: {v}"
                        for n, v in sorted(cat_metrics.items())
                    ]
                )
                if items_str:
                    output_lines.append(f"{category.upper()}: [ {items_str} ]")
        if output_lines:
            self.info(
                f"Step: {step} | Aggregated Metrics | {' || '.join(output_lines)}"
            )
        else:
            self.info(f"Step: {step} | No metrics aggregated or logged.")

    def _format_sample_for_backend(self, sample_data: Dict[str, Any]) -> str:
        """Helper to format a sample dictionary into a string for backend text logging."""
        parts = [f"**Step:** {sample_data.get('step', 'N/A')}"]
        sorted_keys = sorted(
            sample_data.keys(),
            key=lambda k: 0
            if k == "step"
            else 1
            if isinstance(sample_data[k], (dict, list, str))
            and len(str(sample_data[k])) > 80
            else 0,
        )
        for key in sorted_keys:
            if key == "step":
                continue
            value = sample_data[key]
            value_str = str(value)
            key_title = key.replace("_", " ").title()
            if isinstance(value, (dict, list)):
                try:
                    value_str = f"\n```json\n{json.dumps(value, indent=2, sort_keys=True)}\n```\n"
                except TypeError:
                    value_str = f"\n```\n{str(value)}\n```\n"
            elif isinstance(value, str) and ("\n" in value_str or len(value_str) > 80):
                value_str = f"\n```\n{html.escape(value_str)}\n```\n"
            elif isinstance(value, float):
                value_str = f"{value:.4g}"
            parts.append(f"**{key_title}:** {value_str}")
        return "\n\n".join(parts)

    def flush(self) -> None:
        """Flushes handlers that support it (currently SampleHandler for files)."""
        self.debug("Flushing LoggingManager resources...")
        # Only SampleHandler has an explicit flush for file buffers now
        if hasattr(self.sample_handler, "flush"):
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
                self.error(f"Error closing handler {type(handler).__name__}: {e}")
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
if __name__ == "__main__":
    # Keep MockConfig and MockDistributedManager as they were in the previous refactor

    class MockConfig:  # Ensure it has needed attrs for BackendHandler
        project_name = "test_logging_final"
        learning_rate = 1e-4
        batch_size = 8
        env_type = "mock"
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
            self.logger = setup_logger(self._rank, "DEBUG")
            self.logger = logging.LoggerAdapter(self.logger, {"rank": self.global_rank})
            self._device = torch.device("cpu")  # Mock device
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
            self.logger.info("[MockDist] Teardown.")

    print("--- Starting Final Refactored LoggingManager Test ---")

    rank = 0
    world_size = 1
    dist_manager_mock = MockDistributedManager(rank=rank, world_size=world_size)
    config = MockConfig()
    log_dir = "./test_runs/single_process"

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
        sample_file_format="jsonl",
        log_sample_interval=2,
        max_backend_samples=3,
        log_level="DEBUG",
    )

    # --- Training Loop Simulation (identical to previous refactor test) ---
    num_steps = 5
    global_step_counter = 0
    for step in range(num_steps):
        logger.debug(f"--- Start Step {step} (Global: {global_step_counter}) ---")
        with logger.train_scope():
            loss = 1.0 / (global_step_counter + 1) + random.random() * 0.1
            accuracy = 0.8 + (global_step_counter * 0.01) + random.random() * 0.01
            lr = config.learning_rate * (0.9**step)
            logger.log_scalar("loss", loss)
            logger.log_metrics_dict({"accuracy": accuracy, "learning_rate": lr})
            for i in range(3):
                logger.log_sample(
                    "generation",
                    {
                        "prompt": f"TrP{global_step_counter}_{i}",
                        "resp": f"TrR{global_step_counter}_{i}",
                        "score": accuracy * 10 + i,
                    },
                    global_step_counter,
                )
            with logger.timer("train_batch_time"):
                time.sleep(0.01 + random.random() * 0.01)

        if step % 2 == 0:
            logger.info(f"--- Running Eval at Step {step} ---")
            with logger.eval_scope():
                eval_loss = loss * 1.1 + random.random() * 0.05
                eval_perp = np.exp(eval_loss)
                eval_rew = random.uniform(5, 10)
                logger.log_scalar("loss", eval_loss)
                logger.log_metrics_dict(
                    {
                        "perplexity": eval_perp,
                        "reward": eval_rew,
                        f"reward_cls_{random.randint(0, 1)}": eval_rew
                        + random.random(),
                    }
                )
                for i in range(2):
                    logger.log_sample(
                        "validation",
                        {
                            "prompt": f"EvP{global_step_counter}_{i}",
                            "resp": f"EvR{global_step_counter}_{i}",
                        },
                        global_step_counter,
                    )
                with logger.timer("eval_total_time"):
                    time.sleep(0.02 + random.random() * 0.01)

        logger.log_scalar("buffer_size", random.randint(100, 200))
        logger.aggregate_and_log(global_step_counter)
        logger.debug(f"--- End Step {step} (Global: {global_step_counter}) ---")
        global_step_counter += 1

    logger.close()
    print(f"[Rank {rank}] Finished final refactored test.")
    print(f"Logs saved in: {os.path.abspath(log_dir)}")
    print(
        "Check subdirectories: 'wandb' or 'tensorboard', 'train/samples', 'eval/samples', 'general/samples'"
    )

    print("\n--- Final Refactored LoggingManager Test Complete ---")
