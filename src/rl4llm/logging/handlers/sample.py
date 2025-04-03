import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rl4llm.constants import EVAL_PHASE, TRAIN_PHASE
from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers.base import BaseHandler


class SampleFileLogger:
    """
    Handles writing structured samples to a single file (JSONL or Parquet) per instance.

    This class manages buffering and writing data samples for a specific phase and rank
    to a dedicated file on disk. It supports JSONL (optionally gzipped) or Parquet format.
    The distinction between different types of samples (previously 'tags') should be
    handled by including a 'tag' field within the logged data itself.

    Attributes:
        save_path (str): Full path to the output file.
        file_format (str): Output format ('parquet', 'jsonl.gz', or 'jsonl').
        compression (str): Compression algorithm for files.
        buffer_size (int): Number of samples to buffer before writing.
    """

    SUPPORTED_FORMATS = {
        'parquet': {'extension': 'parquet', 'default_compression': 'snappy'},
        'jsonl': {'extension': 'jsonl', 'default_compression': None},
        'jsonl.gz': {'extension': 'jsonl.gz', 'default_compression': 'gzip'},
    }

    def __init__(
        self,
        phase_dir: str,
        rank: int,
        file_format: str = 'parquet',
        compression: str = None,
        buffer_size: int = 100,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SampleFileLogger for a specific phase and rank.

        Args:
            phase_dir (str): Directory specific to the phase where samples will be saved
                             (a 'samples' subdirectory will be created within this).
            rank (int): Process rank identifier for distributed logging.
            file_format (str): File format ('parquet', 'jsonl.gz', or 'jsonl').
            compression (str, optional): Compression algorithm. Uses format default if None.
            buffer_size (int): Number of samples to buffer before writing.
            logger (Optional[logging.Logger]): Logger instance.

        Raises:
            ValueError: If file_format is not supported.
            OSError: If the directory cannot be created.
        """
        file_format = file_format.lower()
        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_format}. "
                f"Must be one of: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        # Directory where the actual sample file will live
        self.samples_dir = os.path.join(phase_dir, 'samples')
        self.rank = rank
        self.file_format = file_format
        self.compression = (
            compression
            or self.SUPPORTED_FORMATS[file_format]['default_compression']
        )
        self.buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []  # Single buffer per instance

        self._logger = (
            logger if logger is not None else logging.getLogger('RL4LLM')
        )

        try:
            os.makedirs(self.samples_dir, exist_ok=True)
        except OSError as e:
            self._logger.error(
                f"Failed to create sample directory {self.samples_dir}: {e}"
            )
            raise

        # Determine the single file path this instance manages
        self.save_path = self._get_filepath()
        self._logger.info(
            f"Initialized SampleFileLogger for rank {rank}. Output file: {self.save_path}"
        )

    def _get_filepath(self) -> str:
        """
        Generate the single filepath managed by this logger instance.

        Returns:
            str: The full filepath where samples for this phase/rank will be saved.
        """
        # No tag needed here, filename is determined by rank only within the phase dir
        extension = self.SUPPORTED_FORMATS[self.file_format]['extension']
        # Filename could be simpler, e.g., "samples_rankX.ext" or just "rankX.ext"
        # Let's use "rankX.ext" for simplicity within the specific samples dir.
        return os.path.join(self.samples_dir, f"rank{self.rank}.{extension}")

    def log(self, data: Dict[str, Any], step: int) -> None:
        """
        Log a sample. The sample is added to the buffer and flushed if full.

        Args:
            data (Dict[str, Any]): The sample data to log. Should include any
                                   distinguishing 'tag' if needed.
            step (int): The current step or iteration number.
        """
        log_entry = {'step': step, **data}

        self._buffer.append(log_entry)

        if len(self._buffer) >= self.buffer_size:
            self._flush()

    def _flush(self) -> None:
        """
        Flush the buffer to the designated file on disk. Handles appending correctly
        for Parquet by reading existing data, concatenating, and overwriting.

        Raises:
            IOError: If unable to write to the file.
        """
        if not self._buffer:
            self._logger.debug(
                f"Rank {self.rank}: Flush called with empty buffer for {self.save_path}."
            )
            return

        self._logger.debug(
            f"Rank {self.rank}: Flushing {len(self._buffer)} records to {self.save_path} (format: {self.file_format})."
        )
        new_df = pd.DataFrame(self._buffer)
        new_table = pa.Table.from_pandas(new_df, preserve_index=False)

        try:
            file_exists = os.path.exists(self.save_path)

            if self.file_format == 'parquet':
                if file_exists:
                    # Read the existing Parquet file into an Arrow Table
                    try:
                        existing_table = pq.read_table(self.save_path)
                        # Concatenate the existing table and the new table
                        combined_table = pa.concat_tables(
                            [existing_table, new_table]
                        )
                        # Write the combined table, overwriting the old file
                        pq.write_table(
                            combined_table,
                            self.save_path,
                            compression=self.compression,
                        )
                        self._logger.debug(
                            f"Rank {self.rank}: Appended {len(new_table)} rows to existing Parquet: {self.save_path}"
                        )
                    except Exception as read_concat_error:
                        self._logger.error(
                            f"Rank {self.rank}: Failed to read/concatenate Parquet {self.save_path} for append: {read_concat_error}",
                            exc_info=True,
                        )
                        # Decide how to handle: maybe try writing to a new file? For now, re-raise.
                        raise IOError(
                            f"Failed during Parquet append operation: {read_concat_error}"
                        ) from read_concat_error

                else:
                    # File doesn't exist, write the new table directly
                    pq.write_table(
                        new_table, self.save_path, compression=self.compression
                    )
                    self._logger.debug(
                        f"Rank {self.rank}: Created new Parquet file: {self.save_path}"
                    )

            else:  # jsonl or jsonl.gz (This logic should be correct)
                mode = 'a' if file_exists else 'w'
                # Use the DataFrame directly for to_json
                new_df.to_json(
                    self.save_path,
                    orient='records',
                    lines=True,
                    compression=self.compression,
                    mode=mode,
                )
                self._logger.debug(
                    f"Rank {self.rank}: {'Appended' if mode == 'a' else 'Wrote'} {len(new_df)} rows to JSONL: {self.save_path}"
                )

            self._logger.debug(
                f"Rank {self.rank}: Flushed {len(self._buffer)} rows to {self.file_format} file: {self.save_path}"
            )
            # Clear buffer only on success
            self._buffer = []

        except Exception as e:
            # Catch any exception during write/append and log before raising
            # Avoid logging the specific read_concat_error again if it was already caught above
            if 'read_concat_error' not in locals():
                self._logger.error(
                    f"Rank {self.rank}: Failed to write data to {self.save_path}: {e}",
                    exc_info=True,
                )
            # Do not clear buffer on failure
            raise IOError(
                f"Failed to write to {self.file_format} file: {e}"
            ) from e

    def flush(self) -> None:
        """
        Flush any buffered data to disk.
        """
        try:
            self._flush()
        except Exception as e:
            # Log warning but don't crash the whole flush process if one fails
            self._logger.warning(
                f"Failed to flush buffer to {self.save_path}: {e}"
            )

    def close(self) -> None:
        """
        Flush buffer and clean up resources.
        """
        self.flush()
        # No other resources to explicitly close here (file handles managed by pandas)
        self._logger.info(f"SampleFileLogger for {self.save_path} closed.")


class SampleHandler(BaseHandler):
    """
    Handles sample logging by dispatching to phase-specific file loggers.

    Creates a SampleFileLogger for each specified phase, directing samples
    to the appropriate logger based on the phase. The 'tag' associated with
    a sample is included as part of the data written to the file.
    """

    def __init__(
        self,
        dist_manager: DistributedManager,
        log_dir: str,
        sample_file_format: str = 'parquet',
        sample_buffer_size: int = 100,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)
        self.dist_manager = dist_manager
        self.rank = dist_manager.global_rank
        self.world_size = dist_manager.world_size
        self.is_master = dist_manager.is_master

        # Include 'general' phase for samples logged without a specific phase
        self._log_phases = set([TRAIN_PHASE, EVAL_PHASE])

        self._file_loggers: Dict[str, SampleFileLogger] = {}
        os.makedirs(log_dir, exist_ok=True)  # Ensure base log directory exists

        for phase in self._log_phases:
            # Define the directory for this specific phase
            phase_log_dir = os.path.join(log_dir, phase)
            # SampleFileLogger will create a 'samples' subdir inside phase_log_dir
            try:
                self._file_loggers[phase] = SampleFileLogger(
                    phase_dir=phase_log_dir,  # Pass the phase-specific directory
                    rank=self.rank,
                    buffer_size=sample_buffer_size,
                    file_format=sample_file_format,
                    logger=self._logger,
                )
                self._logger.debug(
                    f"Initialized SampleFileLogger for phase '{phase}' in {os.path.join(phase_log_dir, 'samples')}"
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize SampleFileLogger for phase '{phase}': {e}"
                )
                # Decide if this is fatal or if we can continue without this phase logger
                raise RuntimeError(f"Failed setup for phase {phase}") from e

        # Buffer for backend logging (if used, seems unrelated to file logging change)
        self._samples_for_backend_buffer: List[Tuple[str, Dict[str, Any]]] = []

    def log_sample(
        self,
        phase: str,
        sample_data: Dict[str, Any],
        step: int,
    ):
        """
        Logs a sample to the file corresponding to its phase.

        The 'tag' is added as a field within the sample data dictionary before logging.

        Args:
            phase (Optional[str]): The phase ('train', 'eval', etc.)..
            sample_data (Dict[str, Any]): The core data of the sample.
            step (int): The training step or iteration number.

        """

        if phase not in self._file_loggers:
            # This might happen if phase wasn't in the initial list
            self._logger.warning(f"No logger configured for phase '{phase}'. ")
            return

        try:
            # Get the logger for the current phase
            file_logger = self._file_loggers[phase]
            # Call the simplified log method (no tag argument needed here)
            file_logger.log(sample_data, step)
            self._logger.debug(
                f"Logged sample to file [Phase: {phase}, Rank: {self.rank}] Step: {step}"
            )
        except Exception as e:
            self._logger.error(
                f"Failed to log sample file [Phase: {phase}, Rank: {self.rank}"
            )

    def flush(self) -> None:
        """Flushes all underlying phase-specific file loggers."""
        self._logger.debug(
            f"Flushing sample file loggers for rank {self.rank}..."
        )
        for phase, file_logger in self._file_loggers.items():
            try:
                # self._logger.debug(f"Flushing logger for phase '{phase}'...") # Optional: more verbose logging
                file_logger.flush()
            except Exception as e:
                # Log error but continue flushing other loggers
                self._logger.warning(
                    f"Failed to flush sample logger for phase '{phase}' on rank {self.rank}: {e}"
                )

    def close(self) -> None:
        """Flushes and closes all underlying phase-specific file loggers."""
        self._logger.info(
            f"Closing sample file loggers for rank {self.rank}..."
        )
        self.flush()
        for phase, file_logger in self._file_loggers.items():
            try:
                file_logger.close()
            except Exception as e:
                self._logger.warning(
                    f"Error closing sample logger for phase '{phase}' on rank {self.rank}: {e}"
                )
        self._file_loggers.clear()
        self._logger.debug(f"SampleHandler closed for rank {self.rank}.")
