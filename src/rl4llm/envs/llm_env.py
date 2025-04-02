import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.constants import LOGGER_NAME

# from rl4llm.logging import LoggingManager
from rl4llm.utils.dataset_utils import shard_dataset

logger = logging.getLogger(LOGGER_NAME)

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)  # Basic logging setup


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

        self._seed = seed
        if self._seed is not None:
            # Seed setting should ideally happen outside the class or once globally
            # But keeping it here as per original code for consistency
            random.seed(
                self._seed + rank
            )  # Add rank for different seeds per process
            np.random.seed(self._seed + rank)
            torch.manual_seed(self._seed + rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed + rank)

        self._reward_functions = reward_functions
        self._tokenizer = tokenizer
        if self._tokenizer.pad_token is None:
            logger.warning(
                'Tokenizer does not have a pad token. Setting to eos_token.'
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token
            if self._tokenizer.pad_token is None:
                raise ValueError(
                    'Tokenizer needs a pad_token or eos_token for padding.'
                )
        # Ensure pad_token_id is available
        self._pad_token_id = self._tokenizer.pad_token_id
        self._eos_token_id = self._tokenizer.eos_token_id
        if self._eos_token_id is None:
            # Warning: Some models might not have a distinct EOS token.
            # Generation might rely solely on max_length.
            logger.warning('Tokenizer does not have an EOS token defined.')

        self._batch_size = batch_size
        self._world_size = world_size
        self._rank = rank

        # Consider adding num_workers > 0 if data loading is slow
        # and pin_memory=True if using GPU
        shared_dataset = shard_dataset(
            dataset,
            self._world_size,
            self._rank,
        )
        self._loader = DataLoader(
            shared_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle can add overhead, disable if order doesn't matter
            collate_fn=self._collate_fn,
            # num_workers=4, # Example: Use multiple workers
            # pin_memory=torch.cuda.is_available(), # Example: Pin memory if using CUDA
        )
        self._dataset_iterator = iter(self._loader)
        self._max_prompt_length = max_prompt_length or getattr(
            self._tokenizer, 'model_max_length', 512
        )  # Provide a default if not set
        if self._max_prompt_length is None:
            logger.warning(
                'model_max_length not found in tokenizer. Consider setting max_prompt_length.'
            )
            self._max_prompt_length = 512  # Set a reasonable default

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
            item_batch = next(self._dataset_iterator)
            if not item_batch:  # Handle empty batch from collate_fn
                logger.warning('DataLoader returned an empty batch.')
                # Attempt to get the next batch recursively or handle end of epoch
                return (
                    self.reset()
                )  # Simple retry, might lead to infinite loop if dataset is empty
            return self._prepare_initial_state(item_batch)
        except StopIteration:
            logger.info('Dataset iterator exhausted. Resetting DataLoader.')
            # Optionally check if the dataset is truly empty before resetting
            if len(self._loader.dataset) == 0:
                logger.warning('Dataset appears to be empty.')
                return None  # Signal exhaustion
            self._dataset_iterator = iter(self._loader)
            try:
                item_batch = next(self._dataset_iterator)
                if not item_batch:
                    logger.warning(
                        'DataLoader returned an empty batch after reset.'
                    )
                    return None  # Or handle differently
                return self._prepare_initial_state(item_batch)
            except StopIteration:
                logger.error(
                    'Failed to get batch even after resetting iterator.'
                )
                return None  # Indicate failure or exhaustion
        except Exception as e:
            logger.error(f"Error getting next batch: {e}", exc_info=True)
            raise

    def _prepare_initial_state(self, item_batch: Dict[str, List]) -> EnvState:
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
        inputs = self._tokenizer(
            prompts,
            return_tensors='pt',
            padding='longest' if self._batch_size > 1 else False,
            padding_side='left',  # Use left padding for generation
            truncation=True,
            max_length=self._max_prompt_length,
            return_attention_mask=True,
        )

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

    @torch.inference_mode()
    def rollout(
        self, llm: PreTrainedModel, gen_args: Dict
    ) -> List[EpisodeData]:
        """
        Performs a rollout step: generates completions and calculates rewards. (Optimized)
        """
        group_size = gen_args.get('num_return_sequences', 1)
        if group_size < 1:
            raise ValueError('num_return_sequences must be at least 1')

        # 1. Get the initial state
        s_t = self.reset()
        if s_t is None:  # Handle case where dataset is exhausted
            return []
        input_batch_size = s_t.input_ids.shape[0]
        prompt_length = s_t.input_ids.shape[1]
        device = llm.device

        # 2. Generate sequences (on GPU)
        input_ids = s_t.input_ids.to(device)
        attention_mask = s_t.attention_mask.to(device)

        # Ensure generation args don't conflict and set essential ones
        gen_args_copy = gen_args.copy()
        gen_args_copy.pop('input_ids', None)
        gen_args_copy.pop('attention_mask', None)
        # Ensure pad token ID is set correctly for generation
        # Some models might need eos_token_id as well, depending on stopping criteria
        gen_args_copy['pad_token_id'] = self._pad_token_id
        # If the model doesn't automatically stop at EOS, you might need:
        # gen_args_copy['eos_token_id'] = self._eos_token_id

        # Set output_scores=True if needed by reward functions later, otherwise False
        # Set return_dict_in_generate=True for easier access to outputs
        gen_args_copy['output_scores'] = gen_args_copy.get(
            'output_scores', False
        )
        gen_args_copy['return_dict_in_generate'] = True

        output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_args_copy,
        )

        # Shape: (input_batch_size * group_size, full_sequence_length)
        # Keep sequences on the device for now
        full_sequences = output.sequences
        output_batch_size = full_sequences.shape[0]

        # Verification check
        expected_output_size = input_batch_size * group_size
        if output_batch_size != expected_output_size:
            # This can happen with sampling methods if some sequences finish early
            # across different groups for the same prompt, but generate() usually
            # pads them to the same length within the batch.
            # A mismatch often indicates a deeper issue or misunderstanding.
            logger.warning(
                f"Unexpected output batch size. Expected {expected_output_size}, Got {output_batch_size}. "
                f"Input batch: {input_batch_size}, num_return_sequences: {group_size}. "
                'This might occur with beam search if num_beams != num_return_sequences or complex sampling. '
                'Assuming output size is correct and proceeding.'
            )
            # If the discrepancy is consistent, you might need to adjust group_size dynamically,
            # but it's better to understand the root cause in generate() args.
            # For now, we'll proceed assuming output_batch_size is the actual number generated.
            # Recalculate group_size based on output ONLY IF divisible, otherwise error.
            if output_batch_size % input_batch_size == 0:
                group_size = output_batch_size // input_batch_size
                logger.warning(
                    f"Adjusted group_size to {group_size} based on output."
                )
            else:
                # This is problematic. The output doesn't align with input structure.
                raise RuntimeError(
                    f"Output batch size {output_batch_size} is not compatible with input batch size {input_batch_size}."
                )

        # 3. Process outputs efficiently
        # Get completion tokens (still on device)
        # Slicing is cheap on GPU
        completion_ids_full = full_sequences[
            :, prompt_length:
        ]  # (output_batch_size, completion_max_len)

        # --- Efficiently find actual sequence lengths (including first EOS) ---
        # Create a mask for non-padding tokens
        non_padding_mask = completion_ids_full != self._pad_token_id

        # Find the index of the first EOS token, if present
        # Create a tensor where EOS tokens are marked, others are marked with a large number
        eos_indices = torch.full_like(
            completion_ids_full, completion_ids_full.shape[1]
        )
        if self._eos_token_id is not None:
            eos_mask = completion_ids_full == self._eos_token_id
            # Scatter large number where there is no EOS, keep original index where there is EOS
            eos_indices = torch.where(
                eos_mask,
                torch.arange(
                    completion_ids_full.shape[1], device=device
                ).unsqueeze(0),
                completion_ids_full.shape[1],
            )

        # Find the minimum index for each sequence (first EOS)
        first_eos_idx = torch.min(
            eos_indices, dim=1
        ).values  # (output_batch_size,)

        # Find the length based on non-padding tokens (for sequences without EOS)
        non_padding_len = non_padding_mask.sum(dim=1)  # (output_batch_size,)

        # The actual length is the minimum of (index of first EOS + 1) and (length of non-padding tokens)
        # We add 1 to first_eos_idx because the index is 0-based and we want to include the EOS token
        actual_lengths = torch.min(
            first_eos_idx + 1, non_padding_len
        )  # (output_batch_size,)
        # Ensure lengths are at least 0, although they should be >= 0 anyway
        actual_lengths = torch.clamp(actual_lengths, min=0)

        # --- Decode using batch_decode ---
        # We need the completion tokens on CPU for batch_decode
        completion_ids_cpu = completion_ids_full.cpu()
        decoded_texts = []
        # Decode each sequence up to its actual length
        # batch_decode handles padding/eos correctly if skip_special_tokens=True
        # However, to be precise and match the `actual_lengths` logic, we slice first.
        sequences_to_decode = [
            comp_tensor[:length]
            for comp_tensor, length in zip(completion_ids_cpu, actual_lengths)
        ]
        decoded_texts = self._tokenizer.batch_decode(
            sequences_to_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,  # Often recommended
        )
        completion_texts = decoded_texts

        # 4. Prepare data for reward calculation and EpisodeData construction
        # Expand original batch data efficiently
        expanded_prompts = [p for p in s_t.prompt for _ in range(group_size)]
        expanded_ground_truths = [
            gt for gt in s_t.ground_truth for _ in range(group_size)
        ]
        expanded_raw_data = [
            rd for rd in s_t.raw_data for _ in range(group_size)
        ]
        # Expand prompt tokens (keep on CPU as they are for storage)
        expanded_prompt_tokens = [
            pt.cpu() for pt in s_t.input_ids for _ in range(group_size)
        ]

        # Verify expanded list lengths match output_batch_size
        # (Adjust check if group_size was dynamically changed)
        if len(expanded_prompts) != output_batch_size:
            # This might happen if group_size was adjusted. Re-expand based on actual output.
            # This indicates the generation process didn't produce the expected num_return_sequences for all inputs.
            logger.warning(
                f"Mismatch after expansion. Expected {output_batch_size}, got {len(expanded_prompts)}. Re-expanding based on actual output."
            )
            # This assumes the output sequences maintain the original input order grouping.
            expanded_prompts = [
                s_t.prompt[i // group_size] for i in range(output_batch_size)
            ]
            expanded_ground_truths = [
                s_t.ground_truth[i // group_size]
                for i in range(output_batch_size)
            ]
            expanded_raw_data = [
                s_t.raw_data[i // group_size] for i in range(output_batch_size)
            ]
            expanded_prompt_tokens = [
                s_t.input_ids[i // group_size].cpu()
                for i in range(output_batch_size)
            ]

        # 5. Calculate rewards (batched)
        # This part remains largely the same, assuming reward functions expect lists
        reward_dict_batch = {}
        for reward_fn in self._reward_functions:
            try:
                rewards = reward_fn(
                    completion_texts,  # List[str] (output_batch_size)
                    expanded_ground_truths,  # List[str] (output_batch_size)
                    # Optionally pass more context if needed by reward functions
                    # prompts=expanded_prompts,
                    # raw_data=expanded_raw_data,
                )
                if (
                    not isinstance(rewards, list)
                    or len(rewards) != output_batch_size
                ):
                    raise ValueError(
                        f"Reward function '{reward_fn.name}' did not return a list of size {output_batch_size}"
                    )
                reward_dict_batch[reward_fn.name] = rewards
            except Exception as e:
                logger.error(
                    f"Error calculating reward with {reward_fn.name}: {e}",
                    exc_info=True,
                )
                reward_dict_batch[reward_fn.name] = [0.0] * output_batch_size

        # 6. Construct EpisodeData efficiently
        results: List[EpisodeData] = []
        # Move actual_lengths to CPU once for the loop
        actual_lengths_cpu = (
            actual_lengths.cpu().numpy()
        )  # Use numpy for potentially faster indexing in loop

        for i in range(output_batch_size):
            # Slice the original completion tokens (already on CPU) using the calculated length
            # No redundant EOS finding needed here
            completion_len = actual_lengths_cpu[i]
            # Ensure slicing doesn't go out of bounds (shouldn't happen with clamp earlier)
            completion_tokens_for_sample = completion_ids_cpu[
                i, :completion_len
            ]

            # Get the corresponding prompt tokens (already expanded and on CPU)
            prompt_tokens_for_sample = expanded_prompt_tokens[i]
            # Calculate prompt length (consider attention mask for actual length if needed)
            # Simple length of the tensor (including padding if not masked)
            prompt_len = len(prompt_tokens_for_sample)
            # More accurate: prompt_len = (prompt_tokens_for_sample != self._pad_token_id).sum().item()
            # Or use the original attention mask if stored appropriately per sample

            sample = EpisodeData(
                prompt_text=expanded_prompts[i],
                prompt_tokens=prompt_tokens_for_sample,  # Store CPU tensor
                prompt_length=prompt_len,  # Store length
                completion_text=completion_texts[i],
                completion_tokens=completion_tokens_for_sample,  # Store CPU tensor
                completion_length=completion_len,  # Store length
                reward_dict={k: v[i] for k, v in reward_dict_batch.items()},
                raw_data=expanded_raw_data[i],
                # Optionally include generation scores if output_scores=True
                # scores=output.scores[i] if output.scores else None
            )
            results.append(sample)

        return results


# class LLMEnv:
#     """
#     Environment for generating LLM training samples with batching and multiple return sequences.

#     Manages interaction flow, prompt generation, response processing,
#     and reward calculation based on configuration.
#     """

#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: int,
#         tokenizer: PreTrainedTokenizer,
#         reward_functions: List[BaseRewardFunction],
#         rank: Optional[int] = 0,
#         world_size: Optional[int] = 1,
#         seed: Optional[int] = 42,
#         # Add tokenizer args if needed, e.g., max_length
#         max_prompt_length: Optional[int] = None,
#     ):
#         """
#         Initializes the LLM Environment.

#         Args:
#             dataset: The dataset containing prompts and ground truths.
#                      Expected to yield dictionaries with 'prompt' and 'ground_truth' keys.
#             batch_size: The number of prompts to process in one batch *before* considering num_return_sequences.
#             tokenizer: The tokenizer for processing text.
#             reward_functions: A list of reward function instances.
#             rank: Current rank.
#             world_size: World size.
#             seed: Optional random seed for reproducibility.
#             max_prompt_length: Optional maximum length for tokenized prompts. Defaults to tokenizer's model_max_length.
#         """
#         if batch_size < 1:
#             raise ValueError('Batch size must be at least 1')
#         if not reward_functions or not all(
#             isinstance(fn, BaseRewardFunction) for fn in reward_functions
#         ):
#             raise ValueError(
#                 'reward_functions must be a non-empty list of BaseRewardFunction instances'
#             )

#         self._seed = seed
#         if self._seed is not None:
#             random.seed(self._seed)
#             np.random.seed(self._seed)
#             torch.manual_seed(self._seed)
#             if torch.cuda.is_available():
#                 torch.cuda.manual_seed_all(self._seed)

#         self._reward_functions = reward_functions
#         self._tokenizer = tokenizer
#         # Ensure pad token is set for batching
#         if self._tokenizer.pad_token is None:
#             logger.warning(
#                 'Tokenizer does not have a pad token. Setting to eos_token.'
#             )
#             self._tokenizer.pad_token = self._tokenizer.eos_token
#             if self._tokenizer.pad_token is None:
#                 raise ValueError(
#                     'Tokenizer needs a pad_token or eos_token for padding.'
#                 )

#         self._batch_size = batch_size
#         self._world_size = world_size
#         self._rank = rank

#         shared_dataset = shard_dataset(
#             dataset,
#             self._world_size,
#             self._rank,
#         )

#         # Use a custom collate_fn if dataset items are not dictionaries
#         self._loader = DataLoader(
#             shared_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             collate_fn=self._collate_fn,
#         )
#         self._dataset_iterator = iter(self._loader)
#         self._max_prompt_length = (
#             max_prompt_length or self._tokenizer.model_max_length
#         )

#     def _collate_fn(self, batch: List[Dict]) -> Dict[str, List]:
#         """Collates list of dicts into a dict of lists."""
#         collated = {key: [item[key] for item in batch] for key in batch[0]}
#         return collated

#     def reset(self) -> EnvState:
#         """
#         Resets the environment by sampling a new batch of data.

#         Returns:
#             EnvState: The initial state for the new batch.
#         """
#         try:
#             item_batch = next(self._dataset_iterator)
#             return self._prepare_initial_state(item_batch)
#         except StopIteration:
#             logger.info('Dataset iterator exhausted. Resetting DataLoader.')
#             self._dataset_iterator = iter(self._loader)
#             item_batch = next(self._dataset_iterator)
#             return self._prepare_initial_state(item_batch)
#         except Exception as e:
#             logger.error(f"Error getting next batch: {e}")
#             raise

#     def _prepare_initial_state(self, item_batch: Dict[str, List]) -> EnvState:
#         """
#         Prepares the initial EnvState from a batch of data items.

#         Args:
#             item_batch: A dictionary where keys are 'prompt', 'ground_truth', etc.,
#                         and values are lists of corresponding data for the batch.

#         Returns:
#             EnvState: The prepared initial state for the batch.
#         """
#         if (
#             not isinstance(item_batch, dict)
#             or 'prompt' not in item_batch
#             or 'ground_truth' not in item_batch
#         ):
#             raise ValueError(
#                 f"Invalid batch data format. Expected dict with 'prompt' and 'ground_truth' lists, got {type(item_batch)}"
#             )
#         if not isinstance(item_batch['prompt'], list) or not isinstance(
#             item_batch['ground_truth'], list
#         ):
#             raise ValueError(
#                 "'prompt' and 'ground_truth' values in the batch must be lists."
#             )
#         if len(item_batch['prompt']) != len(item_batch['ground_truth']):
#             raise ValueError(
#                 "Batch size mismatch between 'prompt' and 'ground_truth'."
#             )

#         # Ensure prompts are strings
#         prompts = [str(p) for p in item_batch['prompt']]

#         # Tokenize the batch of prompts with padding and truncation
#         inputs = self._tokenizer(
#             prompts,
#             return_tensors='pt',
#             # Pad to the longest sequence in the batch
#             padding='longest' if self._batch_size > 1 else False,
#             padding_side='left' if self._batch_size > 1 else None,
#             truncation=True,
#             max_length=self._max_prompt_length,
#             return_attention_mask=True,
#         )

#         # Store the actual length of the padded prompts
#         # prompt_length = inputs["input_ids"].shape[1]

#         # Store raw data per sample, not just the whole batch dict
#         raw_data_list = []
#         batch_keys = list(item_batch.keys())
#         num_samples = len(item_batch['prompt'])
#         for i in range(num_samples):
#             raw_data_list.append(
#                 {key: item_batch[key][i] for key in batch_keys}
#             )

#         state = EnvState(
#             prompt=prompts,  # List of prompt strings
#             input_ids=inputs[
#                 'input_ids'
#             ],  # Tensor (batch_size, prompt_seq_len)
#             attention_mask=inputs[
#                 'attention_mask'
#             ],  # Tensor (batch_size, prompt_seq_len)
#             ground_truth=item_batch[
#                 'ground_truth'
#             ],  # List of ground truth strings
#             raw_data=raw_data_list,  # List of raw data dicts
#             # prompt_length=prompt_length,
#         )

#         return state

#     @torch.inference_mode()
#     def rollout(
#         self, llm: PreTrainedModel, gen_args: Dict
#     ) -> List[EpisodeData]:
#         """
#         Performs a rollout step: generates completions for a batch of prompts
#         and calculates rewards.

#         Args:
#             llm: The language model used for generation.
#             gen_args: Dictionary of arguments passed to `llm.generate()`.
#                       Must include `num_return_sequences`.

#         Returns:
#             List[EpisodeData]: A list containing data for each generated episode
#                                (prompt-completion pair with rewards). The total
#                                number of episodes is batch_size * num_return_sequences.
#         """
#         group_size = gen_args.get('num_return_sequences', 1)
#         if group_size < 1:
#             raise ValueError('num_return_sequences must be at least 1')

#         # 1. Get the initial state (batched prompts)
#         s_t = self.reset()
#         input_batch_size = s_t.input_ids.shape[0]  # Actual batch size processed
#         device = llm.device  # Use the model's device

#         # 2. Generate sequences
#         # Move inputs to the correct device
#         input_ids = s_t.input_ids.to(device)
#         attention_mask = s_t.attention_mask.to(device)

#         # Ensure generation args don't conflict with required args
#         gen_args_copy = gen_args.copy()
#         gen_args_copy.pop('input_ids', None)
#         gen_args_copy.pop('attention_mask', None)

#         output = llm.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             **gen_args_copy,
#         )

#         # Shape: (input_batch_size * group_size, full_sequence_length)
#         full_sequences = output.sequences.cpu()
#         output_batch_size = full_sequences.shape[0]

#         # Verification check
#         expected_output_size = input_batch_size * group_size
#         if output_batch_size != expected_output_size:
#             logger.warning(
#                 f"Unexpected output batch size from generate(). "
#                 f"Expected {expected_output_size} (batch={input_batch_size}, num_return={group_size}), "
#                 f"Got {output_batch_size}. Check generation parameters (e.g., sampling)."
#             )
#             # Adjust group_size if necessary, though this might indicate an issue
#             if output_batch_size % input_batch_size == 0:
#                 group_size = output_batch_size // input_batch_size
#                 logger.warning(
#                     f"Adjusting group_size to {group_size} based on output."
#                 )
#             else:
#                 # This case is problematic, maybe raise error or handle differently
#                 raise RuntimeError(
#                     f"Output batch size {output_batch_size} is not divisible by input batch size {input_batch_size}."
#                 )

#         # 3. Process outputs
#         prompt_length = s_t.input_ids.shape[1]
#         completion_ids = full_sequences[:, prompt_length:]
#         preliminary_lengths = (
#             completion_ids != self._tokenizer.pad_token_id
#         ).sum(dim=1)

#         # We will determine the final slice length for each sample individually later.
#         # The EOS truncation loop below is primarily for cleaning up the text decoding
#         # by ensuring text generation stops after the first EOS.
#         # The token sequence should retain the EOS.
#         completion_ids_for_decode = (
#             completion_ids.clone()
#         )  # Clone for decoding modifications
#         for i in range(completion_ids_for_decode.size(0)):
#             # Find positions where we have EOS
#             eos_positions = (
#                 completion_ids_for_decode[i] == self._tokenizer.eos_token_id
#             ).nonzero(as_tuple=True)[0]
#             if len(eos_positions) > 0:
#                 first_eos = eos_positions[0].item()
#                 # Truncate everything *after* (but not including) the first EOS for *decoding*
#                 completion_ids_for_decode[i, first_eos + 1 :] = (
#                     self._tokenizer.pad_token_id
#                 )

#         # Calculate the lengths for *decoding* by ignoring padding.
#         decode_lengths = (
#             completion_ids_for_decode != self._tokenizer.pad_token_id
#         ).sum(dim=1)

#         # Decode completions using the cleaned sequences
#         # Use the lengths derived from the cleaned sequences for batch_decode
#         decoded_texts = []
#         for i in range(output_batch_size):
#             decoded_texts.append(
#                 self._tokenizer.decode(
#                     completion_ids_for_decode[i, : decode_lengths[i]],
#                     skip_special_tokens=True,
#                 )
#             )
#         completion_texts = decoded_texts  # Assign to the variable used later
#         # --- End of section processing completion tokens ---

#         # 4. Prepare data for reward calculation and EpisodeData construction
#         # Expand original batch data to match the output batch size
#         expanded_prompts = []
#         expanded_ground_truths = []
#         expanded_raw_data = []
#         expanded_prompt_tokens = []  # Store original prompt tokens per output

#         for i in range(input_batch_size):
#             for _ in range(group_size):
#                 expanded_prompts.append(s_t.prompt[i])
#                 expanded_ground_truths.append(s_t.ground_truth[i])
#                 expanded_raw_data.append(s_t.raw_data[i])
#                 # Get the original (potentially unpadded) prompt tokens for this sample
#                 # We use the input_ids before padding/truncation if available,
#                 # otherwise, use the padded ones and slice based on attention mask?
#                 # Simplest: just store the padded input_ids for the corresponding prompt.
#                 expanded_prompt_tokens.append(
#                     s_t.input_ids[i].cpu()
#                 )  # Store on CPU

#         # Verify expanded list lengths
#         assert len(expanded_prompts) == output_batch_size
#         assert len(expanded_ground_truths) == output_batch_size
#         assert len(expanded_raw_data) == output_batch_size
#         assert len(expanded_prompt_tokens) == output_batch_size

#         # 5. Calculate rewards
#         reward_dict_batch = {}  # Stores rewards for the entire output batch
#         for reward_fn in self._reward_functions:
#             try:
#                 # Pass expanded lists matching the completions
#                 rewards = reward_fn(
#                     completion_texts,  # List[str] (output_batch_size)
#                     expanded_ground_truths,  # List[str] (output_batch_size)
#                 )
#                 if (
#                     not isinstance(rewards, list)
#                     or len(rewards) != output_batch_size
#                 ):
#                     raise ValueError(
#                         f"Reward function '{reward_fn.name}' did not return a list of size {output_batch_size}"
#                     )
#                 reward_dict_batch[reward_fn.name] = rewards
#             except Exception as e:
#                 logger.error(
#                     f"Error calculating reward with {reward_fn.name}: {e}"
#                 )
#                 # Handle error, e.g., assign default reward or re-raise
#                 reward_dict_batch[reward_fn.name] = [
#                     0.0
#                 ] * output_batch_size  # Example: default reward

#         # 6. Construct EpisodeData for each generated sequence
#         results: List[EpisodeData] = []
#         for i in range(output_batch_size):
#             # --- Start Modification ---
#             # Use the original completion_ids (before modification for decoding)
#             current_completion_ids_original = completion_ids[i]

#             # Find the first EOS token in the *original* completion tokens for this sample
#             eos_positions = (
#                 current_completion_ids_original == self._tokenizer.eos_token_id
#             ).nonzero(as_tuple=True)[0]

#             if len(eos_positions) > 0:
#                 # If an EOS token is present, slice up to and *including* it
#                 first_eos_idx = eos_positions[0].item()
#                 # The length includes the EOS token at index first_eos_idx
#                 actual_len_incl_eos = first_eos_idx + 1
#                 # Slice the original completion tokens to include EOS
#                 completion_tokens_for_sample = current_completion_ids_original[
#                     :actual_len_incl_eos
#                 ]
#             else:
#                 # If no EOS token, the sequence likely finished due to max_length.
#                 # Use the preliminary length calculated before any modifications.
#                 # This captures all non-pad tokens generated.
#                 actual_len_no_eos = preliminary_lengths[i].item()
#                 completion_tokens_for_sample = current_completion_ids_original[
#                     :actual_len_no_eos
#                 ]

#             sample = EpisodeData(
#                 prompt_text=expanded_prompts[i],
#                 prompt_tokens=expanded_prompt_tokens[i],
#                 prompt_length=len(expanded_prompt_tokens[i]),
#                 completion_text=completion_texts[i],
#                 completion_tokens=completion_tokens_for_sample,
#                 completion_length=len(completion_tokens_for_sample),
#                 reward_dict={k: v[i] for k, v in reward_dict_batch.items()},
#                 raw_data=expanded_raw_data[i],
#             )
#             results.append(sample)

#         return results
