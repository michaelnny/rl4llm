import gzip
import json
import logging
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
import yaml

from rl4llm.core.dist_utils import DistributedManager
from rl4llm.core.helper_utils import setup_logger

# --- Internal Component: Sample File Logger ---


class SampleFileLogger:
    """
    Handles writing structured samples to JSONL or Parquet files, with optional compression.

    This class provides a unified interface to log structured data samples to disk in either
    JSONL (optionally gzipped) or Parquet format. It uses pandas DataFrames for all file formats
    to ensure consistency in data handling.

    Attributes:
        save_dir (str): Directory path where sample files will be saved
        rank (int): Process rank identifier (for distributed environments)
        file_format (str): Output format ('parquet', 'jsonl.gz', or 'jsonl')
        compression (str): Compression algorithm for files
        buffer_size (int): Number of samples to buffer before writing to files
    """

    SUPPORTED_FORMATS = {
        'parquet': {'extension': 'parquet', 'default_compression': 'snappy'},
        'jsonl': {'extension': 'jsonl', 'default_compression': None},
        'jsonl.gz': {'extension': 'jsonl.gz', 'default_compression': 'gzip'},
    }

    def __init__(
        self,
        save_dir: str,
        rank: int,
        file_format: str = 'parquet',
        compression: str = None,
        buffer_size: int = 100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SampleFileLogger.

        Args:
            save_dir (str): Base directory where samples will be saved (a 'samples' subdirectory will be created)
            rank (int): Process rank identifier for distributed logging
            file_format (str): File format to use ('parquet', 'jsonl.gz', or 'jsonl')
            compression (str, optional): Compression algorithm. If None, will use format-specific default
            buffer_size (int): Number of samples to buffer before writing to disk
            logger (Optional[logging.Logger]): Logger instance

        Raises:
            ValueError: If file_format is not one of the supported formats
        """
        file_format = file_format.lower()
        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                f"Must be one of: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        self.save_dir = os.path.join(save_dir, 'samples')
        self.rank = rank
        self.file_format = file_format

        # Use format-specific default compression if not specified
        self.compression = (
            compression
            or self.SUPPORTED_FORMATS[file_format]['default_compression']
        )
        self.buffer_size = buffer_size

        # Define buffers for all tags
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}

        # Initialize logger
        self._logger = (
            logger if logger is not None else logging.getLogger('RL4LLM')
        )

        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except OSError as e:
            self._logger.error(
                f"Failed to create sample directory {self.save_dir}: {e}"
            )
            raise

    def _get_filepath(self, tag: str) -> str:
        """
        Generate the filepath for a given tag.

        Args:
            tag (str): The tag identifying the sample category

        Returns:
            str: The full filepath where samples will be saved
        """
        safe_tag = tag.replace('/', '_')
        extension = self.SUPPORTED_FORMATS[self.file_format]['extension']
        return os.path.join(
            self.save_dir, f"{safe_tag}_rank{self.rank}.{extension}"
        )

    def _get_buffer(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get or create the buffer for a tag.

        Args:
            tag (str): The tag identifying the sample category

        Returns:
            List[Dict[str, Any]]: The buffer for the specified tag
        """
        if tag not in self._buffers:
            self._buffers[tag] = []
        return self._buffers[tag]

    def log(self, tag: str, data: Dict[str, Any], step: int) -> None:
        """
        Log a sample with the given tag and data.

        Args:
            tag (str): The tag identifying the sample category
            data (Dict[str, Any]): The sample data to log
            step (int): The current step or iteration number
        """
        log_entry = {'step': step, **data}

        # Add entry to buffer
        buffer = self._get_buffer(tag)
        buffer.append(log_entry)

        # Flush if buffer reaches the specified size
        if len(buffer) >= self.buffer_size:
            self._flush(tag)

    def _flush(self, tag: str) -> None:
        """
        Flush the buffer for a specific tag to disk.

        Args:
            tag (str): The tag identifying the sample category

        Raises:
            IOError: If unable to write to the file
        """
        buffer = self._buffers.get(tag, [])
        if not buffer:
            return

        filepath = self._get_filepath(tag)
        df = pd.DataFrame(buffer)

        try:
            file_exists = os.path.exists(filepath)

            if self.file_format == 'parquet':
                # Write to Parquet file
                if file_exists:
                    df.to_parquet(
                        filepath,
                        engine='pyarrow',
                        compression=self.compression,
                        index=False,
                        append=True,
                        default_handler=str,
                    )
                else:
                    df.to_parquet(
                        filepath,
                        engine='pyarrow',
                        compression=self.compression,
                        index=False,
                        default_handler=str,
                    )
            else:  # jsonl or jsonl.gz
                # For JSON formats
                mode = 'a' if file_exists else 'w'
                df.to_json(
                    filepath,
                    orient='records',
                    lines=True,
                    compression=self.compression,
                    mode=mode,
                    default_handler=str,
                )

            self._logger.info(
                f"Flushed {len(buffer)} rows to {self.file_format} file: {filepath}"
            )
        except Exception as e:
            self._logger.error(f"Failed to write data for tag '{tag}': {e}")
            raise IOError(f"Failed to write to {self.file_format} file: {e}")

        # Clear the buffer after successful flush
        self._buffers[tag] = []

    def flush(self) -> None:
        """
        Flush all buffers to disk.

        This ensures that any buffered data is written to the corresponding files.
        """
        for tag in list(self._buffers.keys()):
            try:
                self._flush(tag)
            except Exception as e:
                self._logger.warning(
                    f"Failed to flush data for tag '{tag}': {e}"
                )

    def close(self) -> None:
        """
        Flush all buffers and clean up resources.

        This method should be called when the logger is no longer needed to ensure
        all data is properly written and resources are released.
        """
        self.flush()
        self._buffers.clear()
        self._logger.info('Flushed all buffers and closed logger.')


# --- Internal Component: Backend Logger ---


class BackendLogger:
    """
    Handles logging metrics, formatted text/samples, and hyperparameters to various backend systems.

    This class provides a unified interface for logging to different backend systems such as
    Weights & Biases (WandB) and TensorBoard. It supports logging metrics, formatted text samples,
    hyperparameters, and individual scalar values.

    Attributes:
        is_master (bool): Whether this logger instance is the master (primary) logger
    """

    def __init__(
        self,
        writer: Optional[Any],
        is_master: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the BackendLogger.

        Args:
            writer: The backend writer object (TensorBoard SummaryWriter or WandB run)
            is_master (bool): Whether this instance is the master logger that should perform logging
            logger (Optional[logging.Logger]): Logger instance
        """
        self._writer = writer
        self.is_master = is_master
        self._logger = (
            logger if logger is not None else logging.getLogger('RL4LLM')
        )

    def _is_wandb_writer(self) -> bool:
        """
        Check if the writer is a Weights & Biases (WandB) run.

        Returns:
            bool: True if the writer is a WandB run, False otherwise
        """
        return self._writer is not None and hasattr(self._writer, 'log')

    def _is_tensorboard_writer(self) -> bool:
        """
        Check if the writer is a TensorBoard SummaryWriter.

        Returns:
            bool: True if the writer is a TensorBoard SummaryWriter, False otherwise
        """
        return self._writer is not None and hasattr(self._writer, 'add_scalar')

    def _can_log(self) -> bool:
        """
        Check if this logger instance can log to the backend.

        Returns:
            bool: True if logging is possible, False otherwise
        """
        return self.is_master and self._writer is not None

    def log_scalar(
        self, name: str, value: Union[float, int], step: int
    ) -> None:
        """
        Log a single scalar value to the backend.

        Args:
            name (str): The name/tag of the scalar
            value (Union[float, int]): The scalar value to log
            step (int): The current step or iteration number

        Example:
            ```python
            logger.log_scalar("loss/train", 0.123, step=1000)
            ```
        """
        if not self._can_log():
            return

        try:
            if self._is_wandb_writer():  # WandB style
                self._writer.log({name: value}, step=step)
            elif self._is_tensorboard_writer():  # TensorBoard style
                self._writer.add_scalar(name, value, step)
        except Exception as e:
            self._logger.warning(
                f"Failed to log scalar '{name}' to backend at step {step}: {e}"
            )

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: int
    ) -> None:
        """
        Log multiple scalar metrics to the backend.

        Args:
            metrics (Dict[str, Union[float, int]]): Dictionary mapping metric names to values
            step (int): The current step or iteration number

        Example:
            ```python
            logger.log_metrics({
                "loss/train": 0.123,
                "accuracy/train": 0.987,
                "learning_rate": 0.001
            }, step=1000)
            ```
        """
        if not self._can_log() or not metrics:
            return

        try:
            if self._is_wandb_writer():  # WandB style
                self._writer.log(metrics, step=step)
            elif self._is_tensorboard_writer():  # TensorBoard style
                for name, value in metrics.items():
                    self._writer.add_scalar(name, value, step)
        except Exception as e:
            self._logger.warning(
                f"Failed to log metrics to backend at step {step}: {e}"
            )

    def log_sample_text(self, tag: str, formatted_text: str, step: int) -> None:
        """
        Log formatted text (representing a sample) to the backend.

        Args:
            tag (str): The tag identifying the sample category
            formatted_text (str): The formatted text content to log
            step (int): The current step or iteration number

        Example:
            ```python
            logger.log_sample_text(
                "generation/example",
                "Input: What is ML?\nOutput: Machine Learning is...",
                step=1000
            )
            ```
        """
        if not self._can_log():
            return

        try:
            log_tag = f"samples/{tag}"  # Group samples under 'samples/' prefix

            if self._is_wandb_writer():  # WandB style
                # Use HTML for better formatting in WandB text panels
                import wandb

                html_text = formatted_text.replace('\n\n', '<br><br>').replace(
                    '\n', '<br>'
                )
                self._writer.log({log_tag: wandb.Html(html_text)}, step=step)
            elif hasattr(self._writer, 'add_text'):  # TensorBoard style
                # Use Markdown formatting
                self._writer.add_text(log_tag, formatted_text, step)
        except Exception as e:
            self._logger.warning(
                f"Failed to log sample text (tag: {tag}) to backend at step {step}: {e}"
            )

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to the backend.

        Args:
            params (Dict[str, Any]): Dictionary of hyperparameters to log

        Example:
            ```python
            logger.log_hyperparams({
                "learning_rate": 0.001,
                "batch_size": 32,
                "model_type": "transformer"
            })
            ```
        """
        if not self._can_log():
            return

        try:
            if self._is_wandb_writer():  # WandB style
                self._writer.config.update(params, allow_val_change=True)
            elif hasattr(self._writer, 'add_text'):  # TensorBoard style
                params_str = yaml.dump(params, sort_keys=False, indent=2)
                self._writer.add_text(
                    'configuration/hyperparameters',
                    f"```yaml\n{params_str}\n```",
                    0,
                )
        except Exception as e:
            self._logger.warning(
                f"Failed to log hyperparameters to backend: {e}"
            )

    def log_multi_scalars(
        self, tag_prefix: str, values: List[Union[float, int]], step: int
    ) -> None:
        """
        Log multiple related scalar values with a common tag prefix.

        Args:
            tag_prefix (str): Common prefix for all scalar tags
            values (List[Union[float, int]]): List of scalar values to log
            step (int): The current step or iteration number

        Example:
            ```python
            logger.log_multi_scalars("layer_outputs", [0.1, 0.2, 0.3], step=1000)
            # Will log: "layer_outputs/0": 0.1, "layer_outputs/1": 0.2, "layer_outputs/2": 0.3
            ```
        """
        if not self._can_log():
            return

        try:
            metrics = {
                f"{tag_prefix}/{i}": value for i, value in enumerate(values)
            }
            self.log_metrics(metrics, step)
        except Exception as e:
            self._logger.warning(
                f"Failed to log multi scalars '{tag_prefix}' to backend at step {step}: {e}"
            )

    def close(self) -> None:
        """
        Close the backend writer and release resources.

        This method should be called when the logger is no longer needed to ensure
        all data is properly written and resources are released.
        """
        if not self._can_log():
            return

        try:
            if hasattr(self._writer, 'finish'):  # WandB style
                self._writer.finish()
            elif hasattr(self._writer, 'close'):  # TensorBoard style
                self._writer.close()
            self._logger.info('Closed backend writer.')
        except Exception as e:
            self._logger.error(f"Failed to close backend writer: {e}")


# --- Main LoggingManager Coordinator ---


# class DefaultRankFilter(logging.Filter):
#     def __init__(self, default_rank='N/A'):
#         super().__init__()
#         self.default_rank = default_rank

#     def filter(self, record):
#         if not hasattr(record, 'rank'):
#             record.rank = self.default_rank
#         return True


# # Attach the filter to the root logger so that every log record gets a default rank.
# logging.getLogger().addFilter(DefaultRankFilter(default_rank='N/A'))


class LoggingManager:
    """
    Coordinates logging of metrics and samples to files and optional backends.

    Manages aggregation in distributed settings (if world_size > 1) and provides
    context managers (`train_scope`, `eval_scope`) for phase-specific logging.
    Designed to work correctly both in single-process (world_size=1) and
    multi-process distributed environments.

    Attributes:
        log_dir (str): Base directory for logs.
        rank (int): Process rank in distributed environment (0 if world_size=1).
        world_size (int): Total number of processes.
        is_master (bool): True if rank is 0.
        config (Any): Experiment configuration object.
        dist_manager (DistributedManager): Distributed manager utility.
    """

    TRAIN = 'train'
    EVAL = 'eval'

    def __init__(
        self,
        config: Any,
        dist_manager: DistributedManager,
        log_dir: str,
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        log_sample_interval: int = 50,
        sample_buffer_size: int = 100,
        sample_file_format: str = 'parquet',
    ):
        """
        Initializes the LoggingManager.

        Args:
            config: Configuration object (e.g., argparse Namespace or OmegaConf dict).
            dist_manager: DistributedManager instance handling process group info.
            log_dir: Base directory for saving logs (files and backend artifacts).
            enable_wandb: If True, try to initialize Weights & Biases logging (master only).
            enable_tensorboard: If True and WandB is not enabled/fails, try TensorBoard (master only).
            log_sample_interval: Frequency to log samples gathered across ranks to log to backend.
            sample_buffer_size: Number of samples to buffer per tag before flushing to file.
            sample_file_format: Format for saving samples ('jsonl.gz', 'jsonl', 'parquet').
        """
        self.config = config
        self.dist_manager = dist_manager
        self.log_dir = log_dir
        self.rank = self.dist_manager.global_rank
        self.world_size = self.dist_manager.world_size
        self.is_master = self.dist_manager.is_master

        self._log_sample_interval = log_sample_interval
        # Log fewer samples to backend to avoid clutter/cost, file logger saves all
        self._log_samples_counts = {
            phase: 0 for phase in [self.TRAIN, self.EVAL]
        }

        self._setup_console_logger()

        self._backend_writer = self._setup_backend_writer(
            enable_wandb, enable_tensorboard
        )

        # Initialize file loggers for different phases
        self._file_loggers: Dict[str, SampleFileLogger] = {
            phase: SampleFileLogger(
                os.path.join(
                    log_dir, phase
                ),  # Save under phase-specific subdirs
                self.rank,
                buffer_size=sample_buffer_size,
                file_format=sample_file_format,
                logger=self.console_logger,
            )
            for phase in [self.TRAIN, self.EVAL]
        }
        self._backend_logger = BackendLogger(
            self._backend_writer,
            self.is_master,
            logger=self.console_logger,
        )

        # Internal buffers
        self._metric_buffer: Dict[str, List[float]] = defaultdict(list)
        self._samples_for_backend_buffer: List[Tuple[str, Dict[str, Any]]] = []
        self._current_phase: Optional[str] = None
        self._phase_stack: List[Optional[str]] = (
            []
        )  # For nested scopes if ever needed

        # Log hyperparameters once at the beginning
        self.log_hyperparams(
            vars(config) if hasattr(config, '__dict__') else dict(config)
        )

    @contextmanager
    def _phase_scope(self, phase: str):
        """Internal helper context manager to set the current logging phase."""
        if phase not in self._file_loggers:
            raise ValueError(
                f"Unsupported logging phase: '{phase}'. Must be one of {list(self._file_loggers.keys())}"
            )

        self._phase_stack.append(self._current_phase)
        self._current_phase = phase
        self.console_logger.debug(f"Entering logging phase: {phase}")
        try:
            yield
        finally:
            self.console_logger.debug(
                f"Exiting logging phase: {self._current_phase}"
            )
            self._current_phase = self._phase_stack.pop()

    def train_scope(self):
        """Context manager to set the logging phase to TRAINING."""
        return self._phase_scope(self.TRAIN)

    def eval_scope(self):
        """Context manager to set the logging phase to EVALUATION."""
        return self._phase_scope(self.EVAL)

    def _get_current_phase_or_raise(self) -> str:
        """Ensures logging happens within a train_scope or eval_scope."""
        if self._current_phase is None:
            raise RuntimeError(
                'Logging methods (log_scalar, log_sample, etc.) must be called '
                'within a `train_scope()` or `eval_scope()` context.'
            )
        return self._current_phase

    def log_scalar(
        self,
        name: str,
        value: Union[float, torch.Tensor],
        step: Optional[int] = None,
    ) -> None:
        """
        Logs a scalar metric within the current phase (train/eval).

        Buffers the metric locally. Aggregation happens in `aggregate_and_log`.

        Args:
            name: Metric name (e.g., "loss"). Phase prefix is added automatically.
            value: Scalar value (float or 0/1-dim Tensor).
            step: Optional step number (primarily used by backend during aggregation).
        """
        current_phase = self._get_current_phase_or_raise()
        full_metric_name = f"{current_phase}/{name}"

        if isinstance(value, torch.Tensor):
            # Ensure tensor is on CPU and detached before converting
            value = value.detach().cpu().item()

        self._metric_buffer[full_metric_name].append(value)
        self.console_logger.debug(
            f"Buffered scalar [{full_metric_name}]: {value}"
        )

    def log_metrics_dict(
        self,
        metrics: Dict[str, Union[float, torch.Tensor]],
        step: Optional[int] = None,
    ) -> None:
        """
        Logs a dictionary of scalar metrics within the current phase (train/eval).

        Args:
            metrics: Dict of metric names to scalar values. Phase prefix added automatically.
            step: Optional step number.
        """
        # Check context once
        current_phase = self._get_current_phase_or_raise()
        for name, value in metrics.items():
            # Use internal logic directly
            full_metric_name = f"{current_phase}/{name}"
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self._metric_buffer[full_metric_name].append(value)
            self.console_logger.debug(
                f"Buffered scalar [{full_metric_name}]: {value}"
            )

    def log_sample(
        self, tag: str, sample_data: Dict[str, Any], step: int
    ) -> None:
        """
        Logs a structured sample within the current phase (train/eval).

        Saves to the phase-specific file and may buffer for backend logging.

        Args:
            tag: Identifier for the sample type (e.g., "generation"). Phase prefix added automatically for backend/aggregation.
            sample_data: The structured data (JSON serializable).
            step: Current step number.
        """
        current_phase = self._get_current_phase_or_raise()
        full_tag = f"{current_phase}/{tag}"  # Tag used for backend

        # 1. Save to file via phase-specific logger (uses original tag for filename)
        file_logger = self._file_loggers[current_phase]
        file_logger.log(tag, sample_data, step)
        self.console_logger.debug(
            f"Logged sample to file [{current_phase}/{tag}] Step: {step}"
        )

        # 2. Buffer for potential backend logging (uses full tag)
        if (
            self._log_samples_counts[current_phase]
            and self._log_sample_interval > 0
            and self._log_samples_counts[current_phase]
            % self._log_sample_interval
            == 0
        ):
            # Include step for context when gathered
            self._samples_for_backend_buffer.append(
                (full_tag, {'step': step, **sample_data})
            )
            self.console_logger.debug(
                f"Buffered sample for backend [{full_tag}]"
            )

        self._log_samples_counts[current_phase] += 1

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters to the backend (master only) and console."""
        self._backend_logger.log_hyperparams(params)
        if self.is_master:
            # Ensure complex objects are represented reasonably in console log
            try:
                params_str = yaml.dump(
                    params,
                    sort_keys=False,
                    indent=2,
                    default_flow_style=False,
                    width=120,
                )
                self.info(f"Hyperparameters:\n{params_str}")
            except Exception:
                self.info(f"Hyperparameters: {params}")  # Fallback

    @contextmanager
    def timer(self, name: str) -> None:
        """
        Context manager for timing operations. Logs elapsed time as 'time/<name>'.

        Note: Time metrics are logged independently of the train/eval phase context.

        Args:
            name: Name for the timed section (e.g., "batch_processing").
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            # Log time metric directly, bypassing phase context check
            time_metric_name = f"time/{name}"
            self._metric_buffer[time_metric_name].append(elapsed)
            self.console_logger.debug(f"Timed [{name}]: {elapsed:.4f}s")

    def aggregate_and_log(self, step: int) -> None:
        """
        Aggregates metrics across ranks (if world_size > 1), logs to backend (master only),
        gathers samples (if world_size > 1), logs subset to backend (master only),
        and clears buffers. Call periodically (e.g., end of step/epoch).

        Args:
            step: Current step number for backend logging.
        """
        # 1. Aggregate numerical metrics
        aggregated_metrics = self._aggregate_metrics()

        # 2. Log aggregated metrics to backend and console (master only)
        if self.is_master and aggregated_metrics:
            self._backend_logger.log_metrics(aggregated_metrics, step)

            # Log aggregated metrics to console
            log_items = []
            # Group by phase for console clarity
            phases = sorted(
                list(self._file_loggers.keys()) + ['time']
            )  # ['eval', 'time', 'train']
            metrics_by_phase = defaultdict(dict)
            other_metrics = {}

            for k, v in aggregated_metrics.items():
                found_phase = False
                for phase in phases:
                    if k.startswith(f"{phase}/"):
                        metrics_by_phase[phase][k.split('/', 1)[1]] = v
                        found_phase = True
                        break
                if not found_phase:
                    other_metrics[k] = v

            for phase in phases:
                if phase in metrics_by_phase:
                    log_items.append(
                        f"{phase.upper()}: ["
                        + ' | '.join(
                            [
                                f"{name}: {val:.4f}"
                                for name, val in sorted(
                                    metrics_by_phase[phase].items()
                                )
                            ]
                        )
                        + ']'
                    )
            if other_metrics:
                log_items.append(
                    'OTHER: ['
                    + ' | '.join(
                        [
                            f"{k}: {v:.4f}"
                            for k, v in sorted(other_metrics.items())
                        ]
                    )
                    + ']'
                )

            if log_items:
                self.info(
                    f"Step: {step} | Aggregated Metrics | {' || '.join(log_items)}"
                )
            else:
                self.info(f"Step: {step} | No metrics aggregated.")

        # 3. Handle buffered samples for backend logging
        self._gather_and_log_samples(step)

        # 4. Clear buffers on all ranks
        self._metric_buffer.clear()
        self._samples_for_backend_buffer.clear()

        # 5. Optional barrier for synchronization if needed after logging/before next step
        if self.world_size > 1:
            self.dist_manager.barrier()
            self.console_logger.debug('Passed aggregation barrier.')

    def _aggregate_metrics(self) -> Dict[str, float]:
        """Aggregates metrics from the local buffer across all processes."""
        aggregated_metrics = {}
        if not self._metric_buffer:
            # If buffer is empty locally, still need to participate if world_size > 1
            # Collect all potential metric names across ranks first if distributed
            if self.world_size > 1:
                all_keys_list = [None] * self.world_size
                local_keys = list(self._metric_buffer.keys())
                # Ensure participation even with empty data
                dist.all_gather_object(all_keys_list, local_keys)
                # Master gets all keys
                all_metric_names = (
                    sorted(
                        list(
                            set(
                                key
                                for sublist in all_keys_list
                                for key in sublist
                            )
                        )
                    )
                    if self.is_master
                    else []
                )
                # Broadcast keys back so all ranks process the same set (or derive from all_gather_object result)
                # simpler: just process local keys and rely on all_reduce matching tags
                if not self.is_master:
                    all_metric_names = local_keys  # Non-masters only process their own keys for reduction
            else:  # world_size == 1
                all_metric_names = sorted(list(self._metric_buffer.keys()))

        else:  # Local buffer is not empty
            all_metric_names = sorted(list(self._metric_buffer.keys()))
            if self.world_size > 1:
                # Optional: Ensure all ranks have the same superset of keys for robustness, see above
                pass  # Sticking to simpler approach: reduce keys present locally

        if not all_metric_names and self.world_size == 1:
            return {}  # No metrics logged locally in single process case

        # --- Perform Reduction ---
        for name in all_metric_names:
            values = self._metric_buffer.get(
                name, []
            )  # Use .get for safety if key list was aggregated

            if not values and self.world_size == 1:
                continue  # Skip empty metrics in single process case

            # Prepare tensor for reduction (even if empty on some ranks in distributed case)
            local_tensor = torch.tensor(
                values, dtype=torch.float32, device=self.dist_manager.device
            )

            if self.world_size > 1:
                # Determine reduction op (SUM for counts/totals, AVG otherwise)
                is_count_or_total = (
                    'count' in name or 'episodes' in name or 'total' in name
                )
                op = (
                    dist.ReduceOp.SUM
                    if is_count_or_total
                    else dist.ReduceOp.AVG
                )

                # All ranks must call all_reduce, even with empty tensors if the metric exists elsewhere
                # We need total count for averaging correctly if op is AVG
                local_count = torch.tensor(
                    len(values),
                    dtype=torch.int64,
                    device=self.dist_manager.device,
                )
                global_sum_tensor = self.dist_manager.all_reduce_tensor(
                    local_tensor.sum().unsqueeze(0), op=dist.ReduceOp.SUM
                )
                global_count_tensor = self.dist_manager.all_reduce_tensor(
                    local_count.unsqueeze(0), op=dist.ReduceOp.SUM
                )

                if self.is_master:
                    global_sum = global_sum_tensor.item()
                    global_count = global_count_tensor.item()

                    if global_count > 0:
                        if op == dist.ReduceOp.SUM:
                            aggregated_metrics[name] = global_sum
                        else:  # AVG
                            aggregated_metrics[name] = global_sum / global_count
                    # else: metric existed but was empty everywhere, skip or log NaN/0? Skip.

            else:  # world_size == 1
                if values:  # Only process if list is not empty
                    is_count_or_total = (
                        'count' in name or 'episodes' in name or 'total' in name
                    )
                    if is_count_or_total:
                        aggregated_metrics[name] = local_tensor.sum().item()
                    else:  # Average
                        aggregated_metrics[name] = local_tensor.mean().item()
                # else: skip if values list is empty

        return aggregated_metrics

    def _gather_and_log_samples(self, step: int) -> None:
        """Gathers samples across ranks (if world_size > 1) and logs a subset to the backend."""
        gathered_samples: List[Tuple[str, Dict[str, Any]]] = []

        if self.world_size > 1:
            # All ranks prepare their local buffer (even if empty)
            local_samples = self._samples_for_backend_buffer

            # Gather lists of samples from all ranks onto the master rank
            world_samples_list = [None] * self.world_size
            # gather_object requires pickleable objects. Our list of tuples should be fine.
            dist.gather_object(
                local_samples,
                (
                    world_samples_list if self.is_master else None
                ),  # Receive buffer only on master
                dst=0,  # Destination rank is master
            )

            if self.is_master:
                # Flatten the list of lists received from all ranks
                gathered_samples = [
                    item
                    for sublist in world_samples_list
                    if sublist is not None
                    for item in sublist
                ]
                self.console_logger.debug(
                    f"Gathered {len(gathered_samples)} samples from {self.world_size} ranks."
                )

        else:  # world_size == 1
            # No gathering needed, just use the local buffer
            gathered_samples = self._samples_for_backend_buffer
            self.console_logger.debug(
                f"Using {len(gathered_samples)} local samples (world_size=1)."
            )

        # Master logs a random subset of gathered samples
        if self.is_master and gathered_samples:
            random.shuffle(gathered_samples)
            samples_to_log = gathered_samples[: self._log_sample_interval]
            self.info(
                f"Logging {len(samples_to_log)} samples to backend (out of {len(gathered_samples)} gathered)."
            )

            for tag, sample_dict in samples_to_log:
                # Use step from sample dict if available, fallback to aggregation step
                sample_step = sample_dict.get('step', step)
                formatted_text = self._format_sample_for_backend(sample_dict)
                self._backend_logger.log_sample_text(
                    tag, formatted_text, sample_step
                )

    def _format_sample_for_backend(self, sample_data: Dict[str, Any]) -> str:
        """Formats a sample dictionary into a readable Markdown string for backend UIs."""
        parts = []
        # Ensure step is always present and formatted nicely
        step = sample_data.get('step', 'N/A')
        parts.append(f"**Step:** {step}")

        for key, value in sample_data.items():
            if key == 'step':
                continue  # Already handled

            value_str = str(value)
            key_title = key.replace('_', ' ').title()

            # Basic formatting: Use code blocks for structure/long text
            if isinstance(value, (dict, list)):
                import json

                try:
                    value_str = (
                        f"\n```json\n{json.dumps(value, indent=2)}\n```\n"
                    )
                except TypeError:
                    value_str = f"\n```\n{str(value)}\n```\n"  # Fallback
            elif isinstance(value, str) and (
                '\n' in value_str or len(value_str) > 80
            ):
                import html

                escaped_value = html.escape(value_str)
                value_str = f"\n```\n{escaped_value}\n```\n"
            elif isinstance(value, float):
                value_str = f"{value:.4f}"

            parts.append(f"**{key_title}:** {value_str}")

        return '\n\n'.join(parts)

    def flush(self) -> None:
        """Flushes all underlying file loggers."""
        self.console_logger.debug('Flushing file loggers...')
        for logger in self._file_loggers.values():
            logger.flush()

    def close(self) -> None:
        """Flushes and closes all loggers and the backend writer."""
        self.info('Closing LoggingManager...')
        self.flush()  # Ensure all file buffers are written
        for logger in self._file_loggers.values():
            logger.close()
        self._backend_logger.close()  # Handles master check internally

        # Final barrier to ensure all ranks clean up before exiting
        if self.world_size > 1:
            self.dist_manager.barrier()
        self.info('LoggingManager closed.')

    # --- Console Logging Setup ---
    def _setup_console_logger(self) -> None:
        """Sets up the console logger for this process."""

        log_level_str = os.environ.get(
            'LOG_LEVEL', 'INFO' if self.is_master else 'WARNING'
        ).upper()
        self.console_logger = setup_logger(self.rank, log_level_str)
        self.console_logger.debug('Console logger initialized.')

    def _setup_backend_writer(
        self, enable_wandb: bool, enable_tensorboard: bool
    ) -> Optional[Any]:
        """Initializes WandB or TensorBoard writer on the master process."""
        if not self.is_master:
            self.console_logger.debug(
                'Not master rank, skipping backend writer setup.'
            )
            return None

        backend_writer = None
        backend_name = 'None'

        # Try WandB first if enabled
        if enable_wandb:
            try:
                global wandb  # Make wandb available if imported
                import wandb

                wandb_dir = os.path.join(
                    self.log_dir, 'wandb_files'
                )  # Store wandb internal files here
                os.makedirs(wandb_dir, exist_ok=True)

                # Generate a run name (can be customized via config)
                run_name = f"run_{os.path.basename(self.log_dir)}_{time.strftime('%Y%m%d_%H%M%S')}"
                run_id = (
                    wandb.util.generate_id()
                )  # Generate a unique ID for potential resuming

                wandb.init(
                    project=getattr(
                        self.config, 'project_name', 'rl4llm_project'
                    ),
                    config=(
                        vars(self.config)
                        if hasattr(self.config, '__dict__')
                        else dict(self.config)
                    ),
                    dir=self.log_dir,  # Main log directory
                    # save_code=True, # Optional: save main script to wandb
                    name=run_name,
                    id=run_id,
                    resume='allow',  # Allow resuming if run with the same ID exists
                    settings=wandb.Settings(
                        _stats_sample_rate_seconds=300,  # Reduce frequency of sys stats logging
                        _stats_disk_paths=[
                            self.log_dir
                        ],  # Monitor disk usage of log dir
                        # log_internal=os.path.join(wandb_dir, "wandb_internal.log") # Redirect internal logs
                    ),
                )
                backend_writer = wandb
                backend_name = 'WandB'
                self.info(
                    f"Initialized WandB. Run name: {run_name}, Run ID: {run_id}"
                )
                self.info(
                    f"WandB dashboard: {wandb.run.get_url() if wandb.run else 'N/A'}"
                )

            except ImportError:
                self.warning(
                    'WandB requested but `wandb` package not installed. Skipping.'
                )
                enable_wandb = False
            except Exception as e:
                self.error(
                    f"Failed to initialize WandB: {e}. Disabling WandB logging."
                )
                enable_wandb = False

        # Fallback/alternative: TensorBoard if enabled and WandB isn't active
        if enable_tensorboard and not backend_writer:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_log_dir = os.path.join(self.log_dir, 'tensorboard_logs')
                os.makedirs(tb_log_dir, exist_ok=True)
                backend_writer = SummaryWriter(
                    log_dir=tb_log_dir,
                    comment='_' + os.path.basename(self.log_dir),
                )
                backend_name = 'TensorBoard'
                self.info(f"Initialized TensorBoard. Logs: {tb_log_dir}")
                # Print command hint for TensorBoard
                try:
                    # Get absolute path for clarity
                    abs_log_dir = os.path.abspath(self.log_dir)
                    self.info(
                        f"To view TensorBoard, run: tensorboard --logdir {abs_log_dir}"
                    )
                except Exception:
                    pass  # Ignore errors getting path etc.

            except ImportError:
                self.warning(
                    'TensorBoard requested but `tensorboard` package not installed. Skipping.'
                )
            except Exception as e:
                self.error(
                    f"Failed to initialize TensorBoard: {e}. Disabling TensorBoard logging."
                )

        if not backend_writer:
            self.warning(
                'No backend logger (WandB/TensorBoard) was initialized.'
            )
        else:
            self.info(f"Using {backend_name} for backend logging.")

        return backend_writer

    # --- Convenience wrappers for console logger ---
    def info(self, message: str) -> None:
        if (
            self.is_master
        ):  # Only master logs info by default, others log warnings/errors
            self.console_logger.info(message)

    def warning(self, message: str) -> None:
        # All ranks log warnings
        self.console_logger.warning(message)

    def error(self, message: str) -> None:
        # All ranks log errors
        self.console_logger.error(message)

    def debug(self, message: str) -> None:
        # Useful for detailed tracing, enable via LOG_LEVEL=DEBUG env var
        self.console_logger.debug(message)


# --- Testing Code ---
if __name__ == '__main__':
    # Mock config and dist manager for testing
    class MockConfig:
        def __init__(self):
            self.project_name = 'test_logging'
            self.learning_rate = 1e-4
            self.batch_size = 8
            self.env_type = 'mock'
            self.seed = 42

        # Make it behave like a dict too for vars() or dict()
        def __iter__(self):
            yield 'project_name', self.project_name
            yield 'learning_rate', self.learning_rate
            yield 'batch_size', self.batch_size
            yield 'env_type', self.env_type
            yield 'seed', self.seed

        def items(self):
            return self.__iter__()

        def keys(self):
            return [k for k, v in self]

        def __getitem__(self, key):
            return getattr(self, key)

    # --- Mock DistributedManager ---
    # Allows testing without initializing torch.distributed
    class MockDistributedManager(DistributedManager):
        def __init__(self, rank=0, world_size=1):
            self._rank = rank
            self._world_size = world_size
            # Simulate device based on availability or mock
            self._device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            # print(f"[MockDistManager] Initialized: Rank={self.global_rank}, WorldSize={self.world_size}, Device={self.device}")

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
            # if self._world_size > 1:
            #     print(f"[MockDistManager Rank {self.global_rank}] Sync Barrier (Simulation)")
            # In real scenario: dist.barrier()
            time.sleep(0.01)  # Small delay to simulate sync

        def all_reduce_tensor(
            self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.AVG
        ) -> torch.Tensor:
            if self._world_size == 1:
                return tensor  # No reduction needed
            else:
                # Simulate reduction: master gets average/sum, others might get zeros or original
                # For testing, let's return the original tensor but print simulation info
                # print(f"[MockDistManager Rank {self.global_rank}] AllReduce Simulation (Op: {op}) Tensor: {tensor.shape}")
                # A real simulation would require communication or assumptions.
                # For testing the logger logic, assuming master gets a valid result is enough.
                # If master, simulate averaging/summing (e.g., divide by world_size for AVG)
                if self.is_master:
                    if op == dist.ReduceOp.AVG and self.world_size > 0:
                        return (
                            tensor / self.world_size
                        )  # Simplistic avg simulation
                    elif op == dist.ReduceOp.SUM:
                        return (
                            tensor * self.world_size
                        )  # Simplistic sum simulation
                    else:
                        return tensor
                else:
                    # Non-masters don't strictly need the correct reduced value usually
                    return torch.zeros_like(
                        tensor
                    )  # Return zeros for non-masters

        def setup(self):  # Add setup method if base class requires it
            pass

        def cleanup(self):  # Add cleanup method if base class requires it
            pass

    # --- Test Execution ---
    print('--- Starting LoggingManager Test ---')
    # Set environment variables to control behavior if needed
    # os.environ['LOG_LEVEL'] = 'DEBUG' # uncomment for verbose logging

    # Use the mock distributed manager
    # Test with world_size = 1
    print('\n--- Testing with world_size = 1 ---')
    rank_1 = 0
    world_size_1 = 1
    dist_manager_1 = MockDistributedManager(
        rank=rank_1, world_size=world_size_1
    )

    config_1 = MockConfig()
    log_dir_1 = './test_runs/single_process'
    # Clean previous run
    if rank_1 == 0 and os.path.exists(log_dir_1):
        import shutil

        shutil.rmtree(log_dir_1)
    os.makedirs(log_dir_1, exist_ok=True)

    # Initialize LoggingManager (disable wandb for local tests unless configured)
    logger_1 = LoggingManager(
        config_1,
        dist_manager_1,
        log_dir_1,
        enable_wandb=False,  # Set True if you have wandb configured and want to test it
        enable_tensorboard=True,
        sample_file_format='jsonl',  # Use plain jsonl for easier inspection
    )

    # --- Training Loop Simulation ---
    num_steps = 5
    for step in range(num_steps):
        logger_1.debug(f"--- Start Step {step} ---")
        # Simulate work and metrics
        time.sleep(0.05)
        loss = 1.0 / (step + 1) + random.random() * 0.1
        accuracy = 0.8 + (step * 0.02) + random.random() * 0.01
        sample_prompt = f"Train Prompt {step} Rank {rank_1}"
        sample_response = f"Train Response {step} - Detail ..."
        sample_score = accuracy * 10

        # Log within train scope
        with logger_1.train_scope():
            logger_1.log_scalar('loss', loss, step)
            logger_1.log_metrics_dict(
                {'accuracy': accuracy, 'learning_rate': config_1.learning_rate},
                step,
            )
            if step % 2 == 0:
                logger_1.log_sample(
                    'generation',
                    {
                        'prompt': sample_prompt,
                        'response': sample_response,
                        'score': sample_score,
                        'nested': {'a': 1, 'b': [1, 2]},
                    },
                    step,
                )
            # Test timer
            with logger_1.timer('data_loading'):
                time.sleep(0.01)
            with logger_1.timer('model_update'):
                time.sleep(0.02)

        # Simulate evaluation periodically
        if step % 3 == 0:
            time.sleep(0.02)
            eval_loss = loss * 1.2 + random.random() * 0.05
            eval_perplexity = 2**eval_loss
            eval_prompt = f"Eval Prompt {step} Rank {rank_1}"
            eval_response = f"Eval Response {step} - Validation Detail ..."

            with logger_1.eval_scope():
                logger_1.log_scalar('loss', eval_loss, step)  # "eval/loss"
                logger_1.log_scalar('perplexity', eval_perplexity, step)
                logger_1.log_sample(
                    'validation',
                    {'prompt': eval_prompt, 'response': eval_response},
                    step,
                )
                with logger_1.timer('eval_step_time'):
                    time.sleep(0.01)

        # Aggregate and log (should work fine for world_size=1)
        logger_1.aggregate_and_log(step)
        logger_1.debug(f"--- End Step {step} ---")

    # --- Cleanup ---
    logger_1.close()
    print(f"[Rank {rank_1}] Finished single process test.")
    print(f"Logs saved in: {os.path.abspath(log_dir_1)}")

    # Optional: Add a test for world_size > 1 if you want to simulate it without MPI/torchrun
    # This would involve running this script multiple times with different rank arguments
    # or creating multiple MockDistributedManager instances and manually coordinating.
    # For simplicity, the world_size=1 test covers the core logic changes.

    print('\n--- LoggingManager Test Complete ---')
