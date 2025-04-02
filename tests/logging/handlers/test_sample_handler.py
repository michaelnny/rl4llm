import gzip
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple  # Added Tuple
from unittest.mock import MagicMock  # For mocking DistributedManager

import pandas as pd
import pyarrow as pa  # Required for parquet
import pyarrow.parquet as pq
import pytest

from rl4llm.constants import LOGGING_PHASES
from rl4llm.logging.handlers.sample import SampleFileLogger, SampleHandler


# Mock DistributedManager
class MockDistributedManager:
    def __init__(self, rank=0, world_size=1):
        self.global_rank = rank
        self.world_size = world_size
        self.is_master = rank == 0


# Define logging phases (replace with actual phases if available)
# LOGGING_PHASES = ["train", "eval", "predict"]

# --- Helper Functions ---


def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Reads a JSONL file (gzipped or plain)."""
    data = []
    open_func = gzip.open if filepath.endswith('.gz') else open
    try:
        with open_func(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        return []  # Return empty list if file doesn't exist
    return data


def read_parquet(filepath: str) -> pd.DataFrame:
    """Reads a Parquet file."""
    try:
        return pd.read_parquet(filepath)
    except (
        FileNotFoundError,
        pa.lib.ArrowIOError,
    ):  # ArrowIOError for empty/non-existent
        return (
            pd.DataFrame()
        )  # Return empty DataFrame if file doesn't exist or is invalid


# --- Pytest Fixtures ---


@pytest.fixture
def mock_logger():
    """Fixture for a mock logger."""
    # Using a real logger captured by caplog is often better for functional tests
    # return MagicMock(spec=logging.Logger)
    return logging.getLogger('TestLogger')


@pytest.fixture(params=[0, 1])  # Test with rank 0 and rank 1
def rank(request):
    """Fixture for rank."""
    return request.param


@pytest.fixture
def mock_dist_manager(rank):
    """Fixture for a mock DistributedManager."""
    return MockDistributedManager(rank=rank, world_size=2)


@pytest.fixture(params=['parquet', 'jsonl', 'jsonl.gz'])
def file_format(request):
    """Fixture for different file formats."""
    return request.param


@pytest.fixture
def phase_dir(tmp_path):
    """Fixture for a temporary phase directory."""
    # tmp_path is a pytest fixture providing a temporary directory unique to the test
    p_dir = tmp_path / 'test_phase'
    # No need to create it here, SampleFileLogger should do it
    # p_dir.mkdir()
    return str(p_dir)


@pytest.fixture
def log_dir(tmp_path):
    """Fixture for a temporary base log directory for SampleHandler."""
    l_dir = tmp_path / 'logs'
    l_dir.mkdir()
    return str(l_dir)


# --- Test Class for SampleFileLogger ---


class TestSampleFileLogger:
    def test_init_creates_directory_and_sets_path(
        self, phase_dir, rank, file_format, mock_logger
    ):
        """Verify directory creation and correct file path."""
        logger = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format=file_format,
            logger=mock_logger,
        )
        expected_samples_dir = os.path.join(phase_dir, 'samples')
        assert os.path.isdir(expected_samples_dir)

        expected_extension = SampleFileLogger.SUPPORTED_FORMATS[file_format][
            'extension'
        ]
        expected_filename = f"rank{rank}.{expected_extension}"
        expected_path = os.path.join(expected_samples_dir, expected_filename)
        assert logger.save_path == expected_path
        assert logger.rank == rank
        assert logger.file_format == file_format
        assert logger._buffer == []
        logger.close()  # Cleanup

    def test_init_invalid_format_raises_value_error(
        self, phase_dir, rank, mock_logger
    ):
        """Verify ValueError for unsupported file format."""
        with pytest.raises(ValueError, match='Unsupported file format: csv'):
            SampleFileLogger(
                phase_dir=phase_dir,
                rank=rank,
                file_format='csv',
                logger=mock_logger,
            )

    # Note: Testing OSError on directory creation failure is tricky without
    # more complex mocking (e.g., patching os.makedirs) or filesystem manipulation.
    # We'll assume os.makedirs works as expected for typical functional tests.

    def test_log_adds_to_buffer(
        self, phase_dir, rank, file_format, mock_logger
    ):
        """Verify logging adds data with step to the buffer."""
        logger = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format=file_format,
            buffer_size=10,
            logger=mock_logger,
        )
        sample_data = {'col1': 'value1', 'col2': 10}
        step = 5
        logger.log(sample_data, step)

        assert len(logger._buffer) == 1
        expected_entry = {'step': step, **sample_data}
        assert logger._buffer[0] == expected_entry
        assert not os.path.exists(logger.save_path)  # Buffer not full yet
        logger.close()

    def test_log_triggers_flush_at_buffer_size(
        self, phase_dir, rank, file_format, mock_logger
    ):
        """Verify flush is triggered automatically when buffer is full."""
        buffer_size = 3
        logger = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format=file_format,
            buffer_size=buffer_size,
            logger=mock_logger,
        )

        for i in range(buffer_size - 1):
            logger.log({'id': i, 'data': f"sample_{i}"}, step=i * 2)

        assert len(logger._buffer) == buffer_size - 1
        assert not os.path.exists(logger.save_path)  # File shouldn't exist yet

        # This log call should fill the buffer and trigger flush
        logger.log(
            {'id': buffer_size - 1, 'data': f"sample_{buffer_size - 1}"},
            step=(buffer_size - 1) * 2,
        )

        assert len(logger._buffer) == 0  # Buffer should be empty after flush
        assert os.path.exists(logger.save_path)  # File should now exist

        # Verify content
        if file_format == 'parquet':
            df = read_parquet(logger.save_path)
            assert len(df) == buffer_size
            assert df['id'].tolist() == list(range(buffer_size))
            assert df['step'].tolist() == [i * 2 for i in range(buffer_size)]
        else:  # jsonl or jsonl.gz
            data = read_jsonl(logger.save_path)
            assert len(data) == buffer_size
            assert [d['id'] for d in data] == list(range(buffer_size))
            assert [d['step'] for d in data] == [
                i * 2 for i in range(buffer_size)
            ]

        logger.close()

    def test_explicit_flush_writes_buffer(
        self, phase_dir, rank, file_format, mock_logger
    ):
        """Verify explicit flush writes remaining buffer contents."""
        logger = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format=file_format,
            buffer_size=10,  # Larger than needed
            logger=mock_logger,
        )
        samples_to_log = 2
        for i in range(samples_to_log):
            logger.log({'id': i, 'val': f"data_{i}"}, step=i + 1)

        assert len(logger._buffer) == samples_to_log
        assert not os.path.exists(logger.save_path)

        logger.flush()  # Explicitly flush

        assert len(logger._buffer) == 0  # Buffer should be empty
        assert os.path.exists(logger.save_path)

        # Verify content
        if file_format == 'parquet':
            df = read_parquet(logger.save_path)
            assert len(df) == samples_to_log
            assert df['id'].tolist() == list(range(samples_to_log))
            assert df['step'].tolist() == [i + 1 for i in range(samples_to_log)]
        else:  # jsonl or jsonl.gz
            data = read_jsonl(logger.save_path)
            assert len(data) == samples_to_log
            assert [d['id'] for d in data] == list(range(samples_to_log))
            assert [d['step'] for d in data] == [
                i + 1 for i in range(samples_to_log)
            ]

        logger.close()

    def test_append_to_existing_file(
        self, phase_dir, rank, file_format, mock_logger
    ):
        """Verify data is appended correctly on subsequent flushes."""
        buffer_size = 2
        logger = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format=file_format,
            buffer_size=buffer_size,
            logger=mock_logger,
        )

        # First batch (triggers flush)
        for i in range(buffer_size):
            logger.log({'id': i, 'batch': 1}, step=i)

        assert os.path.exists(logger.save_path)
        assert len(logger._buffer) == 0

        # Second batch (triggers another flush)
        for i in range(buffer_size):
            logger.log(
                {'id': i + buffer_size, 'batch': 2}, step=i + buffer_size
            )

        assert os.path.exists(logger.save_path)
        assert len(logger._buffer) == 0

        # Verify combined content
        total_samples = buffer_size * 2
        if file_format == 'parquet':
            df = read_parquet(logger.save_path)
            assert len(df) == total_samples
            assert df['id'].tolist() == list(range(total_samples))
            assert df['step'].tolist() == list(range(total_samples))
            assert df['batch'].tolist() == [1] * buffer_size + [2] * buffer_size
        else:  # jsonl or jsonl.gz
            data = read_jsonl(logger.save_path)
            assert len(data) == total_samples
            assert [d['id'] for d in data] == list(range(total_samples))
            assert [d['step'] for d in data] == list(range(total_samples))
            assert [d['batch'] for d in data] == [1] * buffer_size + [
                2
            ] * buffer_size

        logger.close()

    def test_close_flushes_remaining_data(
        self, phase_dir, rank, file_format, mock_logger
    ):
        """Verify close() flushes any data left in the buffer."""
        logger = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format=file_format,
            buffer_size=5,
            logger=mock_logger,
        )
        logger.log({'id': 0, 'final': True}, step=100)
        assert len(logger._buffer) == 1
        assert not os.path.exists(
            logger.save_path
        )  # Assuming file wasn't created before

        logger.close()  # Should trigger flush

        assert len(logger._buffer) == 0  # Buffer cleared by internal flush
        assert os.path.exists(logger.save_path)

        # Verify content
        if file_format == 'parquet':
            df = read_parquet(logger.save_path)
            assert len(df) == 1
            assert df.iloc[0]['id'] == 0
            assert df.iloc[0]['step'] == 100
        else:  # jsonl or jsonl.gz
            data = read_jsonl(logger.save_path)
            assert len(data) == 1
            assert data[0]['id'] == 0
            assert data[0]['step'] == 100

    def test_empty_flush_and_close_no_error(
        self, phase_dir, rank, file_format, mock_logger
    ):
        """Verify flush() and close() don't error when buffer is empty."""
        logger = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format=file_format,
            logger=mock_logger,
        )
        try:
            logger.flush()
            logger.close()
        except Exception as e:
            pytest.fail(f"Flush or close raised an exception unexpectedly: {e}")

        assert not os.path.exists(logger.save_path)  # No file should be created

    def test_compression_setting(self, phase_dir, rank, mock_logger):
        """Verify specific compression settings are used (Parquet example)."""
        # Test with a specific compression different from default
        logger_gzip = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank,
            file_format='parquet',
            compression='gzip',  # Explicitly gzip
            buffer_size=1,
            logger=mock_logger,
        )
        logger_gzip.log({'a': 1}, step=1)
        logger_gzip.close()

        assert os.path.exists(logger_gzip.save_path)
        # Check parquet metadata for compression type
        meta = pq.read_metadata(logger_gzip.save_path)
        # Compression can be per column chunk, check the first one
        assert meta.row_group(0).column(0).compression == 'GZIP'

        # Test default compression (snappy for parquet)
        logger_default = SampleFileLogger(
            phase_dir=phase_dir,
            rank=rank + 10,  # Use different rank for different file
            file_format='parquet',
            compression=None,  # Use default
            buffer_size=1,
            logger=mock_logger,
        )
        logger_default.log({'b': 2}, step=1)
        logger_default.close()

        assert os.path.exists(logger_default.save_path)
        meta_default = pq.read_metadata(logger_default.save_path)
        assert meta_default.row_group(0).column(0).compression == 'SNAPPY'


# --- Test Class for SampleHandler ---


class TestSampleHandler:
    @pytest.fixture(autouse=True)
    def setup_logging_phases(self, monkeypatch):
        """Ensure LOGGING_PHASES is defined for the handler."""
        # If LOGGING_PHASES is imported, this might not be strictly necessary,
        # but it makes the dependency explicit for the test module.
        # If it's defined globally in the original module, this is fine.
        # If it needs to be dynamically set/mocked:
        # monkeypatch.setattr("sample_logging.LOGGING_PHASES", ["train", "eval", "predict"], raising=False)
        pass  # Assuming LOGGING_PHASES is accessible

    def test_init_creates_phase_loggers(
        self, log_dir, mock_dist_manager, file_format, mock_logger
    ):
        """Verify SampleFileLoggers are created for each phase + general."""
        handler = SampleHandler(
            dist_manager=mock_dist_manager,
            log_dir=log_dir,
            sample_file_format=file_format,
            sample_buffer_size=50,
            logger=mock_logger,
        )

        expected_phases = set(LOGGING_PHASES)
        assert len(handler._file_loggers) == 2
        assert set(handler._file_loggers.keys()) == expected_phases

        rank = mock_dist_manager.global_rank
        extension = SampleFileLogger.SUPPORTED_FORMATS[file_format]['extension']

        for phase in expected_phases:
            assert phase in handler._file_loggers
            file_logger = handler._file_loggers[phase]
            assert isinstance(file_logger, SampleFileLogger)
            assert file_logger.rank == rank
            assert file_logger.file_format == file_format
            assert file_logger.buffer_size == 50

            # Check directory structure
            expected_phase_dir = os.path.join(log_dir, phase)
            expected_samples_dir = os.path.join(expected_phase_dir, 'samples')
            expected_filepath = os.path.join(
                expected_samples_dir, f"rank{rank}.{extension}"
            )

            assert os.path.isdir(expected_samples_dir)
            assert file_logger.save_path == expected_filepath

        handler.close()  # Cleanup

    def test_log_sample_routes_to_correct_phase_file(
        self, log_dir, mock_dist_manager, file_format, mock_logger
    ):
        """Verify samples are logged to the file corresponding to their phase."""
        handler = SampleHandler(
            dist_manager=mock_dist_manager,
            log_dir=log_dir,
            sample_file_format=file_format,
            sample_buffer_size=1,  # Flush immediately
            logger=mock_logger,
        )

        rank = mock_dist_manager.global_rank
        extension = SampleFileLogger.SUPPORTED_FORMATS[file_format]['extension']

        # Log to specific phases
        handler.log_sample(
            phase='train', sample_data={'type': 'train_data', 'val': 1}, step=1
        )
        handler.log_sample(
            phase='eval', sample_data={'type': 'eval_data', 'val': 2}, step=2
        )

        # Log to general phase (unknown phase)
        handler.log_sample(
            phase='unknown_phase',
            sample_data={'type': 'general_data', 'val': 3},
            step=3,
        )

        # Log without phase (should also go to general)
        # Assuming the class handles phase=None by defaulting to GENERAL_PHASE
        # Update: The provided code defaults unknown phases to GENERAL_PHASE, let's test that.
        # handler.log_sample(phase=None, sample_data={"type": "general_none", "val": 4}, step=4) # Test None if applicable

        handler.close()  # Ensure all data is written

        # Verify train file
        train_path = os.path.join(
            log_dir, 'train', 'samples', f"rank{rank}.{extension}"
        )
        assert os.path.exists(train_path)
        if file_format == 'parquet':
            df_train = read_parquet(train_path)
            assert len(df_train) == 1
            assert df_train.iloc[0]['type'] == 'train_data'
            assert df_train.iloc[0]['step'] == 1
        else:
            data_train = read_jsonl(train_path)
            assert len(data_train) == 1
            assert data_train[0]['type'] == 'train_data'
            assert data_train[0]['step'] == 1

        # Verify eval file
        eval_path = os.path.join(
            log_dir, 'eval', 'samples', f"rank{rank}.{extension}"
        )
        assert os.path.exists(eval_path)
        if file_format == 'parquet':
            df_eval = read_parquet(eval_path)
            assert len(df_eval) == 1
            assert df_eval.iloc[0]['type'] == 'eval_data'
            assert df_eval.iloc[0]['step'] == 2
        else:
            data_eval = read_jsonl(eval_path)
            assert len(data_eval) == 1
            assert data_eval[0]['type'] == 'eval_data'
            assert data_eval[0]['step'] == 2

        # Verify predict file is empty (or doesn't exist)
        predict_path = os.path.join(
            log_dir, 'predict', 'samples', f"rank{rank}.{extension}"
        )
        assert not os.path.exists(predict_path)

    def test_flush_calls_flush_on_all_loggers(
        self, log_dir, mock_dist_manager, file_format, mock_logger
    ):
        """Verify handler.flush() flushes all underlying file loggers."""
        handler = SampleHandler(
            dist_manager=mock_dist_manager,
            log_dir=log_dir,
            sample_file_format=file_format,
            sample_buffer_size=10,  # Don't flush automatically
            logger=mock_logger,
        )
        rank = mock_dist_manager.global_rank
        extension = SampleFileLogger.SUPPORTED_FORMATS[file_format]['extension']

        # Log data to multiple phases, but less than buffer size
        handler.log_sample(phase='train', sample_data={'id': 1}, step=1)
        handler.log_sample(phase='eval', sample_data={'id': 2}, step=1)
        handler.log_sample(
            phase='unknown', sample_data={'id': 3}, step=1
        )  # -> general

        # Check files don't exist yet
        train_path = os.path.join(
            log_dir, 'train', 'samples', f"rank{rank}.{extension}"
        )
        eval_path = os.path.join(
            log_dir, 'eval', 'samples', f"rank{rank}.{extension}"
        )
        assert not os.path.exists(train_path)
        assert not os.path.exists(eval_path)

        # Call handler's flush
        handler.flush()

        # Check files now exist and have content
        assert os.path.exists(train_path)
        assert os.path.exists(eval_path)

        # Quick content check (just length)
        if file_format == 'parquet':
            assert len(read_parquet(train_path)) == 1
            assert len(read_parquet(eval_path)) == 1
        else:
            assert len(read_jsonl(train_path)) == 1
            assert len(read_jsonl(eval_path)) == 1

        handler.close()

    def test_close_calls_close_on_all_loggers(
        self, log_dir, mock_dist_manager, file_format, mock_logger
    ):
        """Verify handler.close() flushes and closes all loggers."""
        handler = SampleHandler(
            dist_manager=mock_dist_manager,
            log_dir=log_dir,
            sample_file_format=file_format,
            sample_buffer_size=10,  # Don't flush automatically
            logger=mock_logger,
        )
        rank = mock_dist_manager.global_rank
        extension = SampleFileLogger.SUPPORTED_FORMATS[file_format]['extension']

        # Log data
        handler.log_sample(phase='train', sample_data={'id': 10}, step=5)
        handler.log_sample(phase='eval', sample_data={'id': 20}, step=5)

        # Spy on the close method of one of the file loggers
        # We mock close to check if it's called, but still want original flush behavior
        original_close = handler._file_loggers['train'].close
        close_called = False

        def mock_close():
            nonlocal close_called
            close_called = True
            original_close()  # Call the original close to flush data

        handler._file_loggers['train'].close = mock_close

        # Call handler's close
        handler.close()

        # Check files exist (implicitly checks flush happened)
        train_path = os.path.join(
            log_dir, 'train', 'samples', f"rank{rank}.{extension}"
        )
        eval_path = os.path.join(
            log_dir, 'eval', 'samples', f"rank{rank}.{extension}"
        )
        assert os.path.exists(train_path)
        assert os.path.exists(eval_path)

        # Check our spy method was called
        assert close_called

        # Check handler cleared its logger dict
        assert not handler._file_loggers

    def test_handler_handles_file_logger_init_error(
        self, log_dir, mock_dist_manager, file_format, mock_logger, monkeypatch
    ):
        """Verify handler raises error if a SampleFileLogger fails init."""

        # Mock SampleFileLogger.__init__ to raise an error for a specific phase
        original_init = SampleFileLogger.__init__

        def faulty_init(self, phase_dir, *args, **kwargs):
            if 'eval' in phase_dir:  # Fail specifically for eval phase
                raise OSError('Disk full simulation')
            original_init(self, phase_dir, *args, **kwargs)

        monkeypatch.setattr(SampleFileLogger, '__init__', faulty_init)

        with pytest.raises(RuntimeError, match='Failed setup for phase eval'):
            SampleHandler(
                dist_manager=mock_dist_manager,
                log_dir=log_dir,
                sample_file_format=file_format,
                logger=mock_logger,
            )

        # No need for handler.close() as init failed

    def test_log_sample_warning_for_missing_logger(
        self, log_dir, mock_dist_manager, file_format, mock_logger, caplog
    ):
        """Verify a warning is logged if trying to log to a phase without a logger."""
        handler = SampleHandler(
            dist_manager=mock_dist_manager,
            log_dir=log_dir,
            sample_file_format=file_format,
            logger=mock_logger,
        )

        # Manually remove a logger after init to simulate a failure or unexpected state
        if 'eval' in handler._file_loggers:
            del handler._file_loggers['eval']

        with caplog.at_level(logging.WARNING):
            handler.log_sample(phase='eval', sample_data={'id': 1}, step=1)

        assert "No logger configured for phase 'eval'" in caplog.text

        handler.close()
