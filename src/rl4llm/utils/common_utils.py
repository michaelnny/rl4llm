import glob
import gzip
import json
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

logger = logging.getLogger()


class DummyLogger:
    def __init__(self):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def exception(self, msg, *args, **kwargs):
        pass

    def log(self, msg, *args, **kwargs):
        pass


def setup_logger(log_file: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """Creates custom logger"""

    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Create a formatter and set it for the console handler
    formatter = logging.Formatter(
        fmt='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Hide default INFO log from httpx._client.py
    logging.getLogger('httpx').setLevel(logging.WARNING)

    # If a log file is provided, add a file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_checkpoint_folders(ckpt_path: str) -> list[str]:
    """Get all checkpoint folders sorted by modification time (newest first)."""
    if not os.path.exists(ckpt_path):
        return []

    # Get all subdirectories in the checkpoint path
    folders = glob.glob(os.path.join(ckpt_path, 'checkpoint_*'))
    # Sort by modification time, newest first
    folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return folders


def cleanup_old_checkpoints(ckpt_path: str, keep_n: int):
    """Remove all but the N most recent checkpoint folders."""
    folders = get_checkpoint_folders(ckpt_path)

    # Keep 'final' checkpoint and N most recent checkpoints
    for folder in folders[keep_n:]:
        try:
            shutil.rmtree(folder)
        except OSError as e:
            print(f"Error removing checkpoint {folder}: {e}")


def set_seed(seed: int = 157):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_file_exist(file_path: str):
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")


def is_texts_similar(text1: str, text2: str, threshold: float = 0.9) -> bool:
    """Checks if two texts is similar

    Args:
        text1 (str): The left-side text to check.
        text2 (str): The right-side text to check.
        threshold (float): Similarity threshold, default 0.95.

    Returns:
        Bool: indicates if the two texts are similar with in the specific threshold.
    """
    assert threshold > 0
    assert text1
    assert text2
    score = SequenceMatcher(None, text1, text2).ratio()
    return score >= threshold


def load_from_json_file(file_path: str) -> Dict:
    """Loads json file content"""
    assert file_path.endswith('.json')
    if not os.path.exists(file_path):
        raise ValueError(f"File not exists {file_path}")

    try:
        with open(file_path, 'r') as f:
            content = json.loads(f.read())
        return content
    except Exception as e:
        logger.error(f"Failed to load json file: {file_path}")
        return None


def save_to_json_file(data: Dict, file_path: str) -> None:
    """Save data to json file"""
    assert file_path.endswith('.json')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        # Use json.dump to write the dictionary to the file in JSON format
        json.dump(data, f, indent=4)  # 'indent=4' ensures proper formatting


def load_yaml_config_file(file_path: str) -> Dict:
    """Load configuration from yaml file"""
    logger.info(f"Loading yaml config file: {file_path}")
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config_file(config: Dict, save_path: str):
    """Save configuration to yaml file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_to_jsonl_file(data: List[Dict], file_path: str) -> None:
    """
    Save data to jsonl file, automatically using compression if file ends with .gz

    Args:
        data: List of dictionaries to save
        file_path: Path to save file (.jsonl or .jsonl.gz)

    Raises:
        ValueError: If file path has invalid extension
    """
    if not (file_path.endswith('.jsonl') or file_path.endswith('.jsonl.gz')):
        raise ValueError('File must have .jsonl or .jsonl.gz extension')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Choose the appropriate opener based on file extension
    opener = gzip.open if file_path.endswith('.gz') else open
    mode = 'wt' if file_path.endswith('.gz') else 'w'

    with opener(file_path, mode, encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def load_from_jsonl_file(file_path: str) -> List[Dict]:
    """
    Load data from jsonl file, automatically detecting if it's compressed

    Args:
        file_path: Path to .jsonl or .jsonl.gz file

    Returns:
        List of dictionaries parsed from the JSONL file

    Raises:
        ValueError: If file doesn't exist or has invalid extension
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not exists {file_path}")

    # Check file extension
    if not (file_path.endswith('.jsonl') or file_path.endswith('.jsonl.gz')):
        raise ValueError('File must have .jsonl or .jsonl.gz extension')

    # Choose the appropriate opener based on file extension
    opener = gzip.open if file_path.endswith('.gz') else open

    with opener(file_path, 'rt', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def merge_jsonl_files(input_files: List[str], output_file: str):
    """
    Merge multiple JSONL files into a single file

    Args:
        input_files: A list of input JSONL files to merge
        output_file: Path to the output merged file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    loaded_data = []
    for input_file in input_files:
        loaded_data.extend(load_from_jsonl_file(input_file))

    save_to_jsonl_file(loaded_data, output_file)


def save_to_parquet_file(data: List[Dict], file_path: str, compression: str = 'snappy') -> None:
    """
    Save data to a Parquet file

    Args:
        data: List of dictionaries to save
        file_path: Path to save the Parquet file
        compression: Compression codec to use (snappy, gzip, brotli, etc.)
    """
    # Convert list of dictionaries to pandas DataFrame
    df = pd.DataFrame(data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save to Parquet
    df.to_parquet(file_path, compression=compression, index=False)


def load_from_parquet_file(file_path: str, columns: Optional[List[str]] = None) -> List[Dict]:
    """
    Load data from a Parquet file

    Args:
        file_path: Path to the Parquet file
        columns: Optional list of columns to load (for selective loading)

    Returns:
        List of dictionaries parsed from the Parquet file

    Raises:
        ValueError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    # Read Parquet file into DataFrame
    df = pd.read_parquet(file_path, columns=columns)

    # Convert DataFrame to list of dictionaries
    return df.to_dict(orient='records')


def get_runtime_device():
    """Get the runtime device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def clean_up_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
