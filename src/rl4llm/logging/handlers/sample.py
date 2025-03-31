import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers.base import BaseHandler


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
        "parquet": {"extension": "parquet", "default_compression": "snappy"},
        "jsonl": {"extension": "jsonl", "default_compression": None},
        "jsonl.gz": {"extension": "jsonl.gz", "default_compression": "gzip"},
    }

    def __init__(
        self,
        save_dir: str,
        rank: int,
        file_format: str = "parquet",
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

        self.save_dir = os.path.join(save_dir, "samples")
        self.rank = rank
        self.file_format = file_format

        # Use format-specific default compression if not specified
        self.compression = (
            compression or self.SUPPORTED_FORMATS[file_format]["default_compression"]
        )
        self.buffer_size = buffer_size

        # Define buffers for all tags
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}

        # Initialize logger
        self._logger = logger if logger is not None else logging.getLogger("RL4LLM")

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
        safe_tag = tag.replace("/", "_")
        extension = self.SUPPORTED_FORMATS[self.file_format]["extension"]
        return os.path.join(self.save_dir, f"{safe_tag}_rank{self.rank}.{extension}")

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
        log_entry = {"step": step, **data}

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

            if self.file_format == "parquet":
                # Write to Parquet file
                if file_exists:
                    df.to_parquet(
                        filepath,
                        engine="pyarrow",
                        compression=self.compression,
                        index=False,
                        append=True,
                        default_handler=str,
                    )
                else:
                    df.to_parquet(
                        filepath,
                        engine="pyarrow",
                        compression=self.compression,
                        index=False,
                        default_handler=str,
                    )
            else:  # jsonl or jsonl.gz
                # For JSON formats
                mode = "a" if file_exists else "w"
                df.to_json(
                    filepath,
                    orient="records",
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
                self._logger.warning(f"Failed to flush data for tag '{tag}': {e}")

    def close(self) -> None:
        """
        Flush all buffers and clean up resources.

        This method should be called when the logger is no longer needed to ensure
        all data is properly written and resources are released.
        """
        self.flush()
        self._buffers.clear()
        self._logger.info("Flushed all buffers and closed logger.")


class SampleHandler(BaseHandler):
    """Handles sample file logging and gathering samples for backend logging."""

    GENERAL_PHASE = "general"

    def __init__(
        self,
        dist_manager: DistributedManager,
        log_dir: str,
        phases: List[str],
        sample_file_format: str = "parquet",
        sample_buffer_size: int = 100,
        log_sample_interval: int = 50,
        max_backend_samples: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)  # Initialize base class
        self.dist_manager = dist_manager
        self.rank = dist_manager.global_rank
        self.world_size = dist_manager.world_size
        self.is_master = dist_manager.is_master

        self._log_phases = phases + [self.GENERAL_PHASE]
        self._log_sample_interval = log_sample_interval
        self._max_backend_samples = max_backend_samples

        self._file_loggers: Dict[str, SampleFileLogger] = {}
        # Ensure base log_dir exists before creating phase subdirs
        os.makedirs(log_dir, exist_ok=True)
        for phase in self._log_phases:
            phase_log_dir = os.path.join(
                log_dir, phase, "samples"
            )  # samples subdir within phase dir
            try:
                # Pass the specific phase_log_dir base to SampleFileLogger
                # SampleFileLogger creates its own "samples" subdir, so adjust path
                # Let's modify SampleFileLogger to take the exact dir maybe?
                # OR: Create the phase dir here, SampleFileLogger uses it.
                # Let's stick to the previous: pass base dir, it creates 'samples' subdir.
                # So, we need os.path.join(log_dir, phase) as the base for SampleFileLogger
                base_phase_dir = os.path.join(log_dir, phase)
                # SampleFileLogger will create base_phase_dir/samples if needed
                self._file_loggers[phase] = SampleFileLogger(
                    save_dir=base_phase_dir,  # Pass the phase base dir
                    rank=self.rank,
                    buffer_size=sample_buffer_size,
                    file_format=sample_file_format,
                    logger=self._logger,
                )
                self._logger.debug(
                    f"Initialized SampleFileLogger for phase '{phase}' in {os.path.join(base_phase_dir, 'samples')}"
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize SampleFileLogger for phase '{phase}': {e}"
                )
                raise RuntimeError(f"Failed setup for phase {phase}") from e

        self._samples_for_backend_buffer: List[Tuple[str, Dict[str, Any]]] = []
        self._local_sample_log_counts: Dict[str, int] = defaultdict(int)

    # ... (Keep log_sample, collect_backend_samples, clear_backend_buffer methods) ...
    def log_sample(
        self, tag: str, sample_data: Dict[str, Any], step: int, phase: Optional[str]
    ):
        """Logs a sample to file and potentially buffers it for backend."""
        current_phase = phase or self.GENERAL_PHASE
        if current_phase not in self._file_loggers:
            self._logger.warning(
                f"No logger for phase '{current_phase}'. Skipping file log for '{tag}'."
            )
        else:
            try:
                self._file_loggers[current_phase].log(tag, sample_data, step)
                self._logger.debug(
                    f"Logged sample to file [{current_phase}/{tag}] Step: {step}"
                )
            except Exception as e:
                self._logger.error(
                    f"Failed log sample file tag='{tag}' phase='{current_phase}': {e}"
                )

        if self._log_sample_interval > 0:
            self._local_sample_log_counts[current_phase] += 1
            if (
                self._local_sample_log_counts[current_phase] % self._log_sample_interval
                == 0
            ):
                backend_tag = f"{current_phase}/{tag}"
                self._samples_for_backend_buffer.append(
                    (backend_tag, {"step": step, **sample_data})
                )
                self._logger.debug(f"Buffered sample for backend [{backend_tag}]")

    def collect_backend_samples(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Gathers samples buffered for backend logging from all ranks to rank 0."""
        gathered_samples: List[Tuple[str, Dict[str, Any]]] = []
        local_samples_buffer = self._samples_for_backend_buffer

        if self.world_size > 1:
            self.dist_manager.barrier()
            world_samples_list = self.dist_manager.gather_object(
                local_samples_buffer, dst=0
            )
            if self.is_master and world_samples_list:
                gathered_samples = [
                    item
                    for sublist in world_samples_list
                    if sublist
                    for item in sublist
                ]
                self._logger.debug(
                    f"Master gathered {len(gathered_samples)} candidate samples."
                )
        elif self.is_master:
            gathered_samples = local_samples_buffer
            self._logger.debug(
                f"Master collected {len(gathered_samples)} local samples."
            )

        selected_samples = []
        if self.is_master and gathered_samples:
            if len(gathered_samples) > self._max_backend_samples:
                random.shuffle(gathered_samples)
                selected_samples = gathered_samples[: self._max_backend_samples]
                self._logger.info(
                    f"Selected {len(selected_samples)} samples for backend (of {len(gathered_samples)})."
                )
            else:
                selected_samples = gathered_samples
                self._logger.info(
                    f"Using all {len(selected_samples)} gathered samples for backend."
                )

        if self.world_size > 1:
            self.dist_manager.barrier()
        # Only master returns selected samples, others return empty list
        return selected_samples if self.is_master else []

    def clear_backend_buffer(self) -> None:
        """Clears the buffer of samples intended for the backend."""
        self._samples_for_backend_buffer.clear()
        self._logger.debug("Cleared backend sample buffer.")

    def flush(self) -> None:  # Renamed from flush_files for consistency potential
        """Flushes all underlying file loggers."""
        self._logger.debug("Flushing sample file loggers...")
        for phase, file_logger in self._file_loggers.items():
            try:
                file_logger.flush()
            except Exception as e:
                self._logger.warning(f"Failed flush phase '{phase}': {e}")

    # Implement the abstract 'close' method
    def close(self) -> None:
        """Flushes and closes all underlying file loggers."""
        self._logger.info("Closing sample file loggers...")
        self.flush()  # Ensure data is written before closing
        for phase, file_logger in self._file_loggers.items():
            try:
                file_logger.close()
            except Exception as e:
                self._logger.warning(f"Error closing phase '{phase}': {e}")
        self._file_loggers.clear()
        self._logger.debug("SampleHandler closed.")
