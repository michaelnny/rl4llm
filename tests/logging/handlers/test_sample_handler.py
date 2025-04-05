import gzip
import json
import logging
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl4llm.constants import LOGGING_PHASES
from rl4llm.logging.handlers.sample_handler import (
    SampleFileLogger,
    SampleHandler,
)


class MockDistributedManager:
    def __init__(self, rank=0, world_size=1):
        self.global_rank = rank
        self.world_size = world_size
        self.is_master = rank == 0


def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Read JSONL or gzipped JSONL file."""
    open_func = gzip.open if filepath.endswith('.gz') else open
    try:
        with open_func(filepath, 'rt', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def read_parquet(filepath: str) -> pd.DataFrame:
    """Read Parquet file and return a DataFrame."""
    try:
        return pd.read_parquet(filepath)
    except (FileNotFoundError, pa.lib.ArrowIOError):
        return pd.DataFrame()


@pytest.fixture
def mock_logger():
    """Return a test logger."""
    return logging.getLogger('TestLogger')


@pytest.fixture(params=[0, 1])
def rank(request):
    """Return the rank for distributed testing."""
    return request.param


@pytest.fixture
def mock_dist_manager(rank):
    """Return a mocked DistributedManager instance."""
    return MockDistributedManager(rank=rank, world_size=2)


@pytest.fixture(params=['parquet', 'jsonl', 'jsonl.gz'])
def file_format(request):
    """Return one of the supported file formats."""
    return request.param


@pytest.fixture
def phase_dir(tmp_path):
    """Return a temporary phase directory path."""
    return str(tmp_path / 'test_phase')


@pytest.fixture
def log_dir(tmp_path):
    """Return a temporary base log directory."""
    path = tmp_path / 'logs'
    path.mkdir()
    return str(path)


@pytest.mark.parametrize(
    'file_format', ['parquet', 'jsonl', 'jsonl.gz'], indirect=True
)
@pytest.mark.parametrize('rank', [0, 1], indirect=True)
def test_sample_file_logger_lifecycle(
    phase_dir, rank, file_format, mock_logger
):
    """Test full lifecycle of SampleFileLogger: init, log, flush, close."""
    logger = SampleFileLogger(
        phase_dir=phase_dir,
        rank=rank,
        file_format=file_format,
        buffer_size=2,
        logger=mock_logger,
    )

    logger.log({'id': 0, 'foo': 'bar'}, step=1)
    logger.log({'id': 1, 'foo': 'baz'}, step=2)  # triggers flush
    logger.log({'id': 2, 'foo': 'qux'}, step=3)  # remains in buffer

    logger.close()  # should flush the last one

    if file_format == 'parquet':
        df = read_parquet(logger.save_path)
        assert len(df) == 3
        assert sorted(df['id'].tolist()) == [0, 1, 2]
    else:
        data = read_jsonl(logger.save_path)
        assert len(data) == 3
        assert sorted([d['id'] for d in data]) == [0, 1, 2]


@pytest.mark.parametrize('file_format', ['parquet'], indirect=True)
def test_compression_behavior(phase_dir, rank, file_format, mock_logger):
    """Test compression settings for Parquet."""
    logger = SampleFileLogger(
        phase_dir=phase_dir,
        rank=rank,
        file_format=file_format,
        compression='gzip',
        buffer_size=1,
        logger=mock_logger,
    )
    logger.log({'a': 1}, step=1)
    logger.close()
    assert (
        pq.read_metadata(logger.save_path).row_group(0).column(0).compression
        == 'GZIP'
    )

    logger_default = SampleFileLogger(
        phase_dir=phase_dir,
        rank=rank + 10,
        file_format=file_format,
        buffer_size=1,
        logger=mock_logger,
    )
    logger_default.log({'b': 2}, step=2)
    logger_default.close()
    assert (
        pq.read_metadata(logger_default.save_path)
        .row_group(0)
        .column(0)
        .compression
        == 'SNAPPY'
    )


@pytest.mark.parametrize(
    'file_format', ['parquet', 'jsonl', 'jsonl.gz'], indirect=True
)
def test_sample_handler_basic_logging(
    log_dir, mock_dist_manager, file_format, mock_logger
):
    """Test that SampleHandler logs to correct phase files."""
    handler = SampleHandler(
        dist_manager=mock_dist_manager,
        log_dir=log_dir,
        sample_file_format=file_format,
        sample_buffer_size=1,
        logger=mock_logger,
    )
    handler.log_sample('train', {'type': 'train'}, step=1)
    handler.log_sample('eval', {'type': 'eval'}, step=2)
    handler.log_sample('unknown', {'type': 'general'}, step=3)
    handler.close()

    def assert_exists_and_has_data(path):
        assert os.path.exists(path)
        if file_format == 'parquet':
            assert len(read_parquet(path)) > 0
        else:
            assert len(read_jsonl(path)) > 0

    rank = mock_dist_manager.global_rank
    ext = SampleFileLogger.SUPPORTED_FORMATS[file_format]['extension']
    for phase in ['train', 'eval']:
        assert_exists_and_has_data(
            os.path.join(log_dir, phase, 'samples', f'rank{rank}.{ext}')
        )


@pytest.mark.parametrize('file_format', ['jsonl'], indirect=True)
def test_sample_handler_flush_and_close(
    log_dir, mock_dist_manager, file_format, mock_logger
):
    """Test flush and close of SampleHandler calls corresponding loggers."""
    handler = SampleHandler(
        dist_manager=mock_dist_manager,
        log_dir=log_dir,
        sample_file_format=file_format,
        sample_buffer_size=10,
        logger=mock_logger,
    )
    handler.log_sample('train', {'id': 1}, step=1)
    handler.flush()
    handler.close()

    rank = mock_dist_manager.global_rank
    path = os.path.join(log_dir, 'train', 'samples', f'rank{rank}.jsonl')
    assert os.path.exists(path)
    assert len(read_jsonl(path)) == 1
