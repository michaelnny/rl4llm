"""Implements MDP ENV for collect samples for RL"""

import logging
import random
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

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

    # @model_validator(mode='after')
    # def check_tensor_shapes(cls, values):
    #     if values.prompt_tokens.dim() != 1:
    #         raise ValueError(
    #             f"Prompt tokens tensor must be 1D vector: {values.prompt_tokens.shape}"
    #         )
    #     if values.completion_tokens.dim() != 1:
    #         raise ValueError(
    #             f"Completion tokens tensor must be 1D vector: {values.completion_tokens.shape}"
    #         )

    #     return values

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


class LLMEnv:
    """
    Environment for generating LLM training samples with batching and multiple return sequences.
    (Optimized Version)

    Manages interaction flow, prompt generation, response processing,
    and reward calculation based on configuration.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        reward_functions: List[BaseRewardFunction],
        rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
        seed: Optional[int] = 42,
        max_prompt_length: Optional[int] = None,
    ):
        if batch_size < 1:
            raise ValueError('Batch size must be at least 1')
        if not reward_functions or not all(
            isinstance(fn, BaseRewardFunction) for fn in reward_functions
        ):
            raise ValueError(
                'reward_functions must be a non-empty list of BaseRewardFunction instances'
            )

        self.seed = seed
        if self.seed is not None:
            # Seed setting should ideally happen outside the class or once globally
            # But keeping it here as per original code for consistency
            random.seed(self.seed + rank)
            np.random.seed(self.seed + rank)
            torch.manual_seed(self.seed + rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed + rank)

        self.reward_functions = reward_functions
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            logger.warning(
                'Tokenizer does not have a pad token. Setting to eos_token.'
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                raise ValueError(
                    'Tokenizer needs a pad_token or eos_token for padding.'
                )
        # Ensure pad_token_id is available
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            # Warning: Some models might not have a distinct EOS token.
            # Generation might rely solely on max_length.
            logger.warning('Tokenizer does not have an EOS token defined.')

        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

        # Consider adding num_workers > 0 if data loading is slow
        shared_dataset = shard_dataset(
            dataset,
            self.world_size,
            self.rank,
        )
        self.loader = DataLoader(
            shared_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            # num_workers=4, # Example: Use multiple workers
        )
        self.dataset_iterator = iter(self.loader)
        self.max_prompt_length = max_prompt_length or getattr(
            self.tokenizer, 'model_max_length', 512
        )
        if self.max_prompt_length is None:
            logger.warning(
                'model_max_length not found in tokenizer. Consider setting max_prompt_length.'
            )
            self.max_prompt_length = 512

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, List]:
        """Collates list of dicts into a dict of lists."""
        if not batch:
            return {}
        # More robust collation
        keys = batch[0].keys()
        collated = {key: [item.get(key) for item in batch] for key in keys}
        return collated

    def reset(self) -> Optional[EnvState]:
        """
        Resets the environment by sampling a new batch of data.

        Returns:
            EnvState: The initial state for the new batch, or None if dataset exhausted.
        """
        try:
            item_batch = next(self.dataset_iterator)
            if not item_batch:
                logger.warning('DataLoader returned an empty batch.')
                return self.reset()
            return self._prepare_new_episode_state(item_batch)
        except StopIteration:
            logger.info('Dataset iterator exhausted. Resetting DataLoader.')
            self.dataset_iterator = iter(self.loader)
            item_batch = next(self.dataset_iterator)
            return self._prepare_new_episode_state(item_batch)
        except Exception as e:
            logger.error(f"Error getting next batch: {e}", exc_info=True)
            raise

    def _prepare_new_episode_state(
        self, item_batch: Dict[str, List]
    ) -> EnvState:
        """
        Prepares the initial EnvState from a batch of data items.
        """
        if (
            not isinstance(item_batch, dict)
            or 'prompt' not in item_batch
            or 'ground_truth' not in item_batch
        ):
            raise ValueError(
                f"Invalid batch data format. Expected dict with 'prompt' and 'ground_truth' lists, got {type(item_batch)}"
            )

        prompts = [str(p) for p in item_batch['prompt']]
        ground_truths = item_batch[
            'ground_truth'
        ]  # Assume they are already strings or appropriate type

        if len(prompts) != len(ground_truths):
            raise ValueError(
                f"Batch size mismatch between 'prompt' ({len(prompts)}) and 'ground_truth' ({len(ground_truths)})."
            )
        if len(prompts) == 0:
            raise ValueError('Batch contains zero samples.')

        # Tokenize the batch of prompts
        # Padding side 'left' is crucial for decoder-only models like GPT
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding='longest',
            padding_side='left',
            truncation=False,
            max_length=self.max_prompt_length,
            return_attention_mask=True,
        )

        prompt_len = inputs['input_ids'].shape[1]
        if self.max_prompt_length and prompt_len > self.max_prompt_length:
            logger.warning(f"Skip sample with prompt tokens size {prompt_len}")
            return self.reset()

        # Store raw data per sample
        raw_data_list = []
        batch_keys = list(item_batch.keys())
        num_samples = len(prompts)
        for i in range(num_samples):
            raw_data_list.append(
                {key: item_batch[key][i] for key in batch_keys}
            )

        state = EnvState(
            prompt=prompts,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            ground_truth=ground_truths,
            raw_data=raw_data_list,
        )
        return state

    def _generate_completions(
        self,
        llm: PreTrainedModel,
        gen_args: Dict,
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Generates completions using the LLM."""
        input_ids = state.input_ids.to(llm.device)
        attention_mask = state.attention_mask.to(llm.device)
        gen_args_copy = gen_args.copy()
        gen_args_copy.pop('input_ids', None)
        gen_args_copy.pop('attention_mask', None)
        gen_args_copy['return_dict_in_generate'] = True
        output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_args_copy,
        )
        return output.sequences

    def _process_completions(
        self, full_sequences: torch.Tensor, state: EnvState, group_size: int
    ) -> Tuple[List[str], List, List, List, List, List[int]]:
        """Processes generated sequences: calculates lengths, decodes, and expands data."""
        # Move to CPU early to save GPU memory, since batch sizes are typically small
        completion_ids_full = full_sequences[
            :, state.input_ids.shape[1] :
        ].cpu()
        num_sequences = completion_ids_full.shape[0]

        actual_lengths = []
        for seq in completion_ids_full:
            # Find first EOS or end of non-padding tokens
            length = seq.shape[0]
            for i, token in enumerate(seq):
                if token == self.eos_token_id or token == self.pad_token_id:
                    length = i + (1 if token == self.eos_token_id else 0)
                    break
            actual_lengths.append(length)

        # Decode completions efficiently with batch_decode
        sequences_to_decode = [
            seq[:length]
            for seq, length in zip(completion_ids_full, actual_lengths)
        ]
        completion_texts = self.tokenizer.batch_decode(
            sequences_to_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Expand batch data using list comprehension (efficient enough for small batches)
        expanded_prompts = [p for p in state.prompt for _ in range(group_size)]
        expanded_ground_truths = [
            gt for gt in state.ground_truth for _ in range(group_size)
        ]
        expanded_raw_data = [
            rd for rd in state.raw_data for _ in range(group_size)
        ]
        expanded_prompt_tokens = [
            pt.cpu() for pt in state.input_ids for _ in range(group_size)
        ]

        # Verify lengths match
        if len(expanded_prompts) != num_sequences:
            raise RuntimeError(
                f"Expanded data size {len(expanded_prompts)} does not match generated sequences {num_sequences}"
            )

        return (
            completion_texts,
            expanded_prompts,
            expanded_ground_truths,
            expanded_raw_data,
            expanded_prompt_tokens,
            actual_lengths,
        )

    def _calculate_rewards(
        self, completion_texts: List[str], expanded_ground_truths: List[str]
    ) -> Dict[str, List[float]]:
        """Calculates rewards for each completion."""
        reward_dict_batch = {}
        for reward_fn in self.reward_functions:
            try:
                rewards = reward_fn(completion_texts, expanded_ground_truths)
                if not isinstance(rewards, list) or len(rewards) != len(
                    completion_texts
                ):
                    raise ValueError(
                        f"Reward function '{reward_fn.name}' output mismatch"
                    )
                reward_dict_batch[reward_fn.name] = rewards
            except Exception as e:
                logger.error(f"Reward function '{reward_fn.name}' failed: {e}")
                reward_dict_batch[reward_fn.name] = [0.0] * len(
                    completion_texts
                )
        return reward_dict_batch

    @torch.inference_mode()
    def rollout(
        self,
        llm: PreTrainedModel,
        gen_args: Dict,
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[EpisodeData]:
        """Performs a rollout: generates completions and calculates rewards."""
        group_size = gen_args.get('num_return_sequences', 1)
        if group_size < 1:
            raise ValueError('num_return_sequences must be at least 1')

        # Get initial state
        state = self.reset()
        if state is None:
            return []

        # Generate and process completions
        full_sequences = self._generate_completions(
            llm, gen_args, state, **kwargs
        )
        (
            completion_texts,
            expanded_prompts,
            expanded_ground_truths,
            expanded_raw_data,
            expanded_prompt_tokens,
            actual_lengths,
        ) = self._process_completions(full_sequences, state, group_size)
        reward_dict_batch = self._calculate_rewards(
            completion_texts, expanded_ground_truths
        )

        # Construct episodes
        results = []
        for i in range(len(completion_texts)):
            # Extract completion tokens (already unpadded)
            completion_tokens = full_sequences[
                i,
                state.input_ids.shape[1] : state.input_ids.shape[1]
                + actual_lengths[i],
            ].cpu()

            # Get padded prompt tokens and calculate actual length
            prompt_tokens_padded = expanded_prompt_tokens[i]
            prompt_len = (
                (prompt_tokens_padded != self.pad_token_id).sum().item()
            )

            # Extract unpadded prompt tokens (last prompt_len tokens since padding is on the left)
            prompt_tokens_unpadded = prompt_tokens_padded[-prompt_len:]

            sample = EpisodeData(
                prompt_text=expanded_prompts[i],
                prompt_tokens=prompt_tokens_unpadded,
                prompt_length=prompt_len,
                completion_text=completion_texts[i],
                completion_tokens=completion_tokens,
                completion_length=actual_lengths[i],
                reward_dict={k: v[i] for k, v in reward_dict_batch.items()},
                raw_data=expanded_raw_data[i],
            )
            results.append(sample)

        return results
