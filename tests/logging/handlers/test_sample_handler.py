import logging
import os
import random

import pandas as pd
import pytest

from rl4llm.logging.handlers.sample import SampleFileLogger, SampleHandler


# --- Helper: Fake distributed manager for testing SampleHandler ---
class FakeDistManager:
    def __init__(self, is_master=True, world_size=1, global_rank=0):
        self.is_master = is_master
        self.world_size = world_size
        self.global_rank = global_rank

    def barrier(self):
        pass

    def gather_object(self, obj, dst=0):
        return [obj]


# === Tests for SampleFileLogger ===


def test_invalid_file_format(tmp_path):
    # Check that an unsupported file_format raises ValueError.
    save_dir = str(tmp_path / 'logdir')
    with pytest.raises(ValueError, match='Unsupported file format'):
        SampleFileLogger(save_dir=save_dir, rank=0, file_format='txt')


def test_get_filepath(tmp_path):
    # Verify that _get_filepath constructs the expected filepath.
    save_dir = str(tmp_path / 'logdir')
    test_logger = logging.getLogger('test')
    sfl = SampleFileLogger(
        save_dir=save_dir, rank=1, file_format='jsonl', logger=test_logger
    )
    tag = 'test/tag'
    filepath = sfl._get_filepath(tag)
    expected = os.path.join(save_dir, 'samples', 'test_tag_rank1.jsonl')
    assert filepath == expected


def test_log_and_flush_jsonl(tmp_path):
    # Using file_format "jsonl" with a small buffer size to force flush.
    save_dir = str(tmp_path / 'logdir')
    test_logger = logging.getLogger('test')
    sfl = SampleFileLogger(
        save_dir=save_dir,
        rank=0,
        file_format='jsonl',
        buffer_size=2,
        logger=test_logger,
    )
    tag = 'sample'
    # Log two samples so the buffer reaches buffer_size and flushes automatically.
    sfl.log(tag, {'a': 1}, step=10)
    sfl.log(tag, {'a': 2}, step=20)
    # After flush, the buffer for "sample" should be empty.
    assert sfl._buffers.get(tag, []) == []
    filepath = sfl._get_filepath(tag)
    # Check that a file was created.
    assert os.path.exists(filepath)
    # Read back the JSONL file.
    df = pd.read_json(filepath, lines=True)
    # Expect two rows with the logged steps.
    assert len(df) == 2
    assert set(df['step']) == {10, 20}


def test_flush_empty_buffer(tmp_path):
    # Ensure that calling flush on a tag with an empty buffer does not error.
    save_dir = str(tmp_path / 'logdir')
    test_logger = logging.getLogger('test')
    sfl = SampleFileLogger(
        save_dir=save_dir, rank=0, file_format='jsonl', logger=test_logger
    )
    # No sample was logged; flush should complete without writing.
    sfl.flush()  # Should not raise any exception.


def test_close_sample_file_logger(tmp_path):
    # Test that calling close flushes and then clears the internal buffers.
    save_dir = str(tmp_path / 'logdir')
    test_logger = logging.getLogger('test')
    sfl = SampleFileLogger(
        save_dir=save_dir,
        rank=0,
        file_format='jsonl',
        buffer_size=2,
        logger=test_logger,
    )
    tag = 'sample'
    sfl.log(tag, {'a': 1}, step=10)
    sfl.close()
    # After close, all buffers should be empty.
    assert sfl._buffers == {}


# === Tests for SampleHandler ===


@pytest.fixture
def fake_dist_manager_master():
    return FakeDistManager(is_master=True, world_size=1, global_rank=0)


@pytest.fixture
def fake_dist_manager_non_master():
    # Simulate a non-master process in a distributed setting.
    return FakeDistManager(is_master=False, world_size=2, global_rank=1)


@pytest.fixture
def sample_handler(tmp_path, fake_dist_manager_master):
    log_dir = str(tmp_path / 'logdir')
    return SampleHandler(
        dist_manager=fake_dist_manager_master,
        log_dir=log_dir,
        phases=['phase1'],
        sample_file_format='jsonl',
        sample_buffer_size=2,  # small buffer for testing flush behavior
        logger=logging.getLogger('test'),
    )


def test_sample_handler_initialization(tmp_path, fake_dist_manager_master):
    log_dir = str(tmp_path / 'logdir')
    sh = SampleHandler(
        dist_manager=fake_dist_manager_master,
        log_dir=log_dir,
        phases=['phase1', 'phase2'],
        sample_file_format='jsonl',
        sample_buffer_size=2,
        logger=logging.getLogger('test'),
    )
    # Expect file loggers for each provided phase and the general phase.
    expected_phases = {'phase1', 'phase2', sh.GENERAL_PHASE}
    assert set(sh._file_loggers.keys()) == expected_phases


def test_log_sample_file_logging(sample_handler):
    # Test that log_sample logs to the appropriate file logger and buffers a backend sample.
    sample_handler.log_sample('test_tag', {'val': 42}, step=1, phase='phase1')
    # Also verify that the file logger for "phase1" has buffered the sample.
    file_logger = sample_handler._file_loggers['phase1']
    assert 'test_tag' in file_logger._buffers
    assert len(file_logger._buffers['test_tag']) == 1


def test_flush_handler_calls_file_logger_flush(monkeypatch, sample_handler):
    # Monkey-patch the flush method of each file logger to record that it is called.
    flush_calls = {}

    def fake_flush(self):
        flush_calls[self] = True

    for fl in sample_handler._file_loggers.values():
        monkeypatch.setattr(fl, 'flush', fake_flush.__get__(fl, type(fl)))
    sample_handler.flush()
    # Verify that each file logger's flush method was called.
    for fl in sample_handler._file_loggers.values():
        assert flush_calls.get(fl, False) is True


def test_close_handler_clears_file_loggers(sample_handler):
    # Test that close flushes and clears all underlying file loggers.
    sample_handler.close()
    assert sample_handler._file_loggers == {}
