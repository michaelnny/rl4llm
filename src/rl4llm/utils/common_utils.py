import gzip
import json
import logging
import os
import random
import time
from collections import defaultdict
from difflib import SequenceMatcher
from threading import Lock
from typing import Dict, List

import numpy as np
import torch
import yaml


logger = logging.getLogger()


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


class Timer:
    """Context manager to measure elapsed"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        if self.elapsed_time is None:
            raise RuntimeError('Timer has not been stopped yet')
        return self.elapsed_time
