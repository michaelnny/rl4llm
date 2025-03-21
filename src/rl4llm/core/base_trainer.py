"""Base trainer with common functionality"""

import gc
import logging
import os
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.utils import DummyLogger, MetricsCollector, compute_grad_norm, save_yaml_config_file

from .data_types import ClassifierConfig, GRPOConfig


class BaseTrainer(ABC):
    """
    Base trainer with common functionality.
    """

    def __init__(
        self,
        config: Union[GRPOConfig, ClassifierConfig],
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: Optional[logging.Logger] = None,
        rank: Optional[int] = 0,
    ):
        """
        Initialize the BaseTrainer with training components.

        Args:
            config (Any): Configuration object for training parameters.
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding/decoding text.
            device (torch.device): Device (CPU/GPU) for computation.
            torch_dtype (torch.dtype): Data type for PyTorch tensors (e.g., float32).
            artifacts_path (str): Directory path for saving logs and checkpoints.
            logger (Optional[logging.Logger]): Logger for training events; defaults to DummyLogger if None.
        """

        self.config = config
        self.device = device
        self.torch_dtype = torch_dtype
        self.tokenizer = tokenizer
        self.artifacts_path = artifacts_path
        self.logger = logger or DummyLogger()  # Use dummy logger to make coding easier

        self.rank = rank
        self.is_master = rank == 0

        self._initialize()

    def _initialize(self):
        """Initialize training components"""

        # Setup special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Initialize metrics
        self._metrics = MetricsCollector()

        # Setup directories and logging
        self._setup_directories()

        if self.is_master:
            # Initialize TensorBoard writer
            self._writer = SummaryWriter(self._tb_log_dir)
        else:
            self._writer = None

    def _setup_directories(self):
        """Helper method to create necessary directories"""
        self._tb_log_dir = os.path.join(self.artifacts_path, 'tb_logs')
        self._checkpoint_dir = os.path.join(self.artifacts_path, 'checkpoints')

        for path in [self._tb_log_dir, self._checkpoint_dir]:
            os.makedirs(path, exist_ok=True)

    def _create_custom_llm_generator(self, policy_model: PreTrainedModel) -> CustomLLMGenerator:
        """Create a custom generator wrapped around the policy model, which supports group temperature and stochastic sampling"""
        # Try replacing the end token with "Wait" for some samples
        source_tokens = []
        # Determine which tokens should be replaced based on format
        if self.config.xml_format:
            source_tokens.append(self.tokenizer.encode('</think>')[0])
            source_tokens.append(self.tokenizer.encode(' </think>')[0])
            source_tokens.append(self.tokenizer.encode(':</think>')[0])
            source_tokens.append(self.tokenizer.encode('.</think>')[0])
        else:
            source_tokens.append(self.eos_token_id)

        self.special_tokens = ['Wait']
        target_tokens = [self.tokenizer.encode(f' {kwd}')[0] for kwd in self.special_tokens]

        # we should only make the replacement for reasoning tokens
        prevent_patterns = [
            self.tokenizer.encode('</think>'),
            self.tokenizer.encode(' </think>'),
            self.tokenizer.encode('<answer>'),
        ]

        return CustomLLMGenerator(
            model=policy_model,
            tokenizer=self.tokenizer,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            prevent_patterns=prevent_patterns,
        )

    def on_exit(self):
        if self._writer:
            self._writer.close()

    def train(self, log_hyper_params: Optional[Dict] = None):
        """Start to train the model.

        Args:
            log_hyper_params (Dict[str, Any], optional): Hyperparameters to log.
        """

        # log the params we use for this training run
        if log_hyper_params and self.is_master:
            save_yaml_config_file(log_hyper_params, os.path.join(self.artifacts_path, 'config.yaml'))
            self._log_hyper_params_to_tensorboard(log_hyper_params)

        self._train()

    def _train(self):
        """Train the model"""
        raise NotImplementedError

    def _evaluate(self):
        """Evaluate the model"""
        raise NotImplementedError

    def get_grad_norm(self, model: PreTrainedModel) -> torch.Tensor:
        """Compute gradient norm for the given model"""
        return compute_grad_norm(model)

    def _log_hyper_params_to_tensorboard(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard.

        Args:
            config (Dict[str, Any]): Hyperparameters dictionary.
        """
        if self._writer and config:
            try:
                config_str = yaml.dump(config, sort_keys=False, indent=4)
                self._writer.add_text('config/parameters', f"```yaml\n{config_str}\n```", 0)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")

    def _log_stats_to_tensorboard(self, stats: Dict[str, Any], step: int) -> None:
        """Log stats to tensorboard"""
        if self._writer:
            try:
                for name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self._writer.add_scalar(f"{name}", value, step)
            except Exception as e:
                self.logger.warning(f"Failed to log stats to TensorBoard: {e}")

    def _clean_up(self) -> None:
        """Clean up GPU cache"""
        torch.cuda.empty_cache()
        gc.collect()
