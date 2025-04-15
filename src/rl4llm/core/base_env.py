"""Implements base MDP ENV for collect samples for RL"""

import logging
import random
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import PaddingStrategy

from rl4llm.constants import LOGGER_NAME
from rl4llm.utils.dataset_utils import shard_dataset

logger = logging.getLogger(LOGGER_NAME)


class EpisodeData(BaseModel):
    """LLM ENV rollout episode"""

    prompt_tokens: torch.Tensor = Field(..., description='Prompt token ids')
    prompt_text: str = Field(..., description='Prompt full text')
    prompt_length: int = Field(..., description='Prompt token size')
    completion_tokens: torch.Tensor = Field(
        ..., description='Completion token ids'
    )
    completion_text: str = Field(..., description='Completion full text')
    completion_length: int = Field(..., description='Completion token size')
    reward_dict: Dict[str, float] = Field(
        ..., description='Rewards for the episode'
    )
    raw_data: Optional[Dict] = Field(None, description='Raw sample data')
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description='Timestamp when the data was generated',
    )

    @model_validator(mode='after')
    def check_tensor_shapes(cls, values):
        if values.prompt_tokens.dim() != 1:
            raise ValueError(
                f"Prompt tokens tensor must be 1D vector: {values.prompt_tokens.shape}"
            )
        if values.completion_tokens.dim() != 1:
            raise ValueError(
                f"Completion tokens tensor must be 1D vector: {values.completion_tokens.shape}"
            )

        return values

    class Config:
        arbitrary_types_allowed = True


class EnvState(BaseModel):
    """Environment state for LLM generation"""

    prompt: List[str] = Field(..., description='Prompt full text')
    input_ids: torch.Tensor = Field(..., description='Prompt token ids')
    attention_mask: torch.Tensor = Field(
        ..., description='Attention mask for the prompt token ids'
    )
    ground_truth: List[str | float | int] = Field(
        ..., description='Ground truth to the problem'
    )
    raw_data: Optional[List[Dict]] = Field(None, description='Raw sample data')

    class Config:
        arbitrary_types_allowed = True


class BaseRewardFunction:
    """
    Base class for reward functions.
    """

    # Define the validation pattern as a constant for clarity
    _VALID_NAME_PATTERN = r'^[a-zA-Z0-9_\-]+$'

    def __init__(self, name: str):
        """
        Initializes the reward function and validates its name.

        Args:
            name: The name for the reward function. Must contain only
                  alphanumeric characters (a-z, A-Z, 0-9), underscores (_),
                  or hyphens (-), and must not be empty.

        Raises:
            TypeError: If the name is not a string.
            ValueError: If the name is empty or does not match the required pattern.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Reward function name must be a string, got {type(name)}."
            )
        if not name:
            raise ValueError('Reward function name cannot be empty.')

        if not re.match(self._VALID_NAME_PATTERN, name):
            raise ValueError(
                f"Invalid reward function name: '{name}'. "
                f"Name must match the pattern: '{self._VALID_NAME_PATTERN}' "
                f"(only alphanumeric, underscore, hyphen allowed)."
            )
        self.name = name

    def __call__(
        self,
        completions: List[str],
        ground_truths: List[str],
        **kwargs: Dict[str, Any],
    ) -> List[float]:
        """Implements the reward function.

        Args:
            completions (List[str]): LLM generated completion texts.
            ground_truths (List[Union[str | float | int]]): Ground truth for the problem.
            **kwargs (Dict[str, Any]): Any additional data.

        Returns:
            List[float]: A list of scalar rewards.
        """
        raise NotImplementedError(
            'Reward functions must implement the __call__ method.'
        )


class BaseEnv(ABC):
    """
    Base Environment for generating training samples with LLM models.

    Key Features:
    - Pre-tokenizes the dataset for efficiency.
    - Supports repeating prompts (`group_size`) for generating multiple completions
      per original prompt.
    - Handles batching and distributed data loading.
    - Calculates rewards based on provided functions.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        reward_functions: List[BaseRewardFunction],
        batch_size: int,
        group_size: int,
        max_prompt_length: Optional[int] = 1024,
        rank: int = 0,
        world_size: int = 1,
        seed: Optional[int] = 42,
        shuffle_dataset: bool = True,
        num_workers: int = 0,
    ):
        """
        Initializes the Env.

        Args:
            dataset: The dataset containing prompts and ground truths.
                     Expected columns: 'prompt' (str), 'ground_truth' (str),
                     and potentially other metadata columns.
            tokenizer: The tokenizer to use for processing text.
            reward_functions: A list of reward function instances.
            batch_size: The number of *original* prompts to process per batch.
                        The effective batch size for the LLM will be batch_size * group_size.
            group_size: The number of times each prompt should be repeated in a batch
                        to generate multiple completions for the same prompt.
            max_prompt_length: The maximum length allowed for tokenized prompts.
                               Prompts longer than this will be truncated. Defaults to 1024.
            rank: The rank of the current process in distributed training. Defaults to 0.
            world_size: The total number of processes in distributed training. Defaults to 1.
            seed: The random seed for reproducibility. Defaults to 42.
            shuffle_dataset: Whether to shuffle the dataset before iterating. Defaults to True.
            num_workers: Number of worker processes for the DataLoader. Defaults to 0.
        """
        if batch_size < 1:
            raise ValueError('Batch size must be at least 1')
        if group_size < 1:
            raise ValueError('Group size must be at least 1')
        if not reward_functions or not all(
            isinstance(fn, BaseRewardFunction) for fn in reward_functions
        ):
            raise ValueError(
                'reward_functions must be a non-empty list of BaseRewardFunction instances'
            )
        if not isinstance(dataset, Dataset):
            raise TypeError('dataset must be a datasets.Dataset instance.')
        if not all(
            col in dataset.column_names for col in ['prompt', 'ground_truth']
        ):
            raise ValueError(
                "Dataset must contain 'prompt' and 'ground_truth' columns."
            )

        self.tokenizer = tokenizer
        self.reward_functions = reward_functions
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_prompt_length = max_prompt_length
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_dataset = shuffle_dataset
        self.num_workers = num_workers

        self.epoch = 0

        self._setup_tokenizer()
        self._set_seed()

        # Pre-tokenize the dataset
        tokenized_dataset = self._tokenize_dataset(dataset)

        # Shard the tokenized dataset for distributed training
        self.sharded_dataset = shard_dataset(
            tokenized_dataset, self.world_size, self.rank
        )

        logger.info(
            f"Env - Rank {self.rank} has {len(self.sharded_dataset)} samples"
        )

        # Setup DataLoader
        self.loader = DataLoader(
            self.sharded_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_dataset,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.dataset_iterator = iter(self.loader)

    def shuffle(self):
        """
        Shuffles the dataset and resets the DataLoader iterator.

        This method shuffles the sharded dataset using a seed based on the initial seed,
        current epoch, and rank, ensuring reproducibility and diversity across ranks
        in distributed training.
        """
        self.epoch += 1
        shuffle_seed = self.seed + self.epoch + self.rank
        self.sharded_dataset = self.sharded_dataset.shuffle(seed=shuffle_seed)
        self.loader = DataLoader(
            self.sharded_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.dataset_iterator = iter(self.loader)

    def _setup_tokenizer(self):
        """Configures the tokenizer with padding settings."""
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            logger.warning(
                'Tokenizer does not have a pad token. Setting to eos_token.'
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.tokenizer.pad_token is None:
                raise ValueError(
                    'Tokenizer needs a pad_token or eos_token for padding.'
                )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            logger.warning(
                'Tokenizer does not have an EOS token defined. Generation might rely solely on max_length.'
            )

    def _set_seed(self):
        """Sets random seeds for reproducibility across libraries."""
        if self.seed is not None:
            seed_val = self.seed + self.rank
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_val)
            logger.info(f"Rank {self.rank}: Random seed set to {seed_val}")

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenizes the 'prompt' column of the dataset.

        Args:
            dataset: The input dataset with a 'prompt' column.

        Returns:
            A new dataset with 'input_ids' and 'attention_mask' columns,
            containing unpadded tokenized prompts. Original columns are kept.
        """
        logger.info(f"Rank {self.rank}: Starting dataset tokenization...")

        def tokenize_fn(examples):
            tokenized = self.tokenizer(
                examples['prompt'],
                truncation=True,
                max_length=self.max_prompt_length,
                padding=False,
            )
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
            }

        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self.num_workers if self.num_workers > 0 else None,
            desc=f"Rank {self.rank} Tokenizing prompts",
        )
        logger.info(f"Rank {self.rank}: Dataset tokenization complete.")
        return tokenized_dataset

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collates a batch of pre-tokenized samples and pads them.

        Args:
            batch: A list of dictionaries, each representing a pre-tokenized sample
                   from the dataset (containing 'input_ids', 'attention_mask',
                   'ground_truth', 'prompt', etc.).

        Returns:
            A dictionary containing padded tensors for 'input_ids' and 'attention_mask',
            and lists for other data like 'ground_truth', 'prompt', 'raw_data'.
        """
        if not batch:
            return {}
        input_ids_list = [item['input_ids'] for item in batch]
        padded = self.tokenizer.pad(
            {'input_ids': input_ids_list},
            padding=PaddingStrategy.LONGEST,
            padding_side='left',
            return_tensors='pt',
            return_attention_mask=True,
        )
        return {
            'input_ids': padded['input_ids'],
            'attention_mask': padded['attention_mask'],
            'ground_truth': [item['ground_truth'] for item in batch],
            'prompt': [item['prompt'] for item in batch],
            'raw_data': [
                {
                    k: v
                    for k, v in item.items()
                    if k not in ['input_ids', 'attention_mask']
                }
                for item in batch
            ],
        }

    def _reset(self) -> Optional[EnvState]:
        """
        Resets the environment by sampling a new batch and preparing the initial state.

        Handles dataset exhaustion by resetting the iterator. Repeats samples
        according to `self.group_size`.

        Returns:
            EnvState: The initial state for the new batch (prompts repeated
                      `group_size` times), or None if the dataset is empty after reset.
        """
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            logger.info(
                f"Rank {self.rank}: Dataset iterator exhausted. Resetting DataLoader."
            )
            self.dataset_iterator = iter(self.loader)
            try:
                batch = next(self.dataset_iterator)
            except StopIteration:
                logger.error(
                    f"Rank {self.rank}: DataLoader yielded no batches even after reset."
                )
                return None
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error getting next batch", exc_info=True
            )
            raise e
        return self._prepare_grouped_state(batch)

    def _prepare_grouped_state(self, batch: Dict[str, Any]) -> EnvState:
        """
        Prepares the EnvState by repeating batch items `group_size` times.

        Args:
            batch: A collated batch dictionary from `_collate_fn`.

        Returns:
            An EnvState object ready for the LLM, with repeated and padded data.
        """
        num_samples = len(batch['ground_truth'])
        if num_samples == 0:
            logger.warning(
                f"Rank {self.rank}: Received an empty batch in _prepare_grouped_state."
            )
            return self.reset()

        expanded_input_ids = batch['input_ids'].repeat_interleave(
            self.group_size, dim=0
        )
        expanded_attention_mask = batch['attention_mask'].repeat_interleave(
            self.group_size, dim=0
        )
        expanded_prompts = [
            p for p in batch['prompt'] for _ in range(self.group_size)
        ]
        expanded_ground_truths = [
            gt for gt in batch['ground_truth'] for _ in range(self.group_size)
        ]
        expanded_raw_data = [
            rd for rd in batch['raw_data'] for _ in range(self.group_size)
        ]

        return EnvState(
            prompt=expanded_prompts,
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            ground_truth=expanded_ground_truths,
            raw_data=expanded_raw_data,
        )

    def _calculate_rewards(
        self, completions: List[str], ground_truths: List[str]
    ) -> Dict[str, List[float]]:
        """
        Calculates rewards for each completion based on the configured reward functions.

        Args:
            completions: List of generated completion strings.
            ground_truths: List of corresponding ground truth strings.

        Returns:
            A dictionary where keys are reward function names and values are lists
            of reward scores for the batch.
        """
        if len(completions) != len(ground_truths):
            raise ValueError(
                f"Mismatch between completions ({len(completions)}) and ground truths ({len(ground_truths)})"
            )
        rewards_dict = {}
        for fn in self.reward_functions:
            try:
                rewards = fn(completions, ground_truths)
                if not isinstance(rewards, list) or len(rewards) != len(
                    completions
                ):
                    raise ValueError(
                        f"Reward function '{fn.name}' output mismatch."
                    )
                rewards_dict[fn.name] = rewards
            except Exception as e:
                logger.error(
                    f"Reward function '{fn.name}' failed: {e}", exc_info=True
                )
                rewards_dict[fn.name] = [0.0] * len(completions)
        return rewards_dict

    @abstractmethod
    @torch.inference_mode()
    def rollout(
        self,
        llm: PreTrainedModel,
        sampling_params: Dict,
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[EpisodeData]:
        """
        Performs a rollout step: gets a batch, generates completions, calculates rewards,
        and returns structured episode data.

        Args:
            llm: The pre-trained language model to use for generation.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens).
            **kwargs: Additional custom arguments.

        Returns:
            A list of EpisodeData objects, one for each generated sample in the batch
            (batch_size * group_size samples). Returns an empty list if the dataset is exhausted.
        """
        pass
