import logging
import re
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

# from rl4llm.logging import LoggingManager
from rl4llm.utils.dataset_utils import shard_dataset
from rl4llm.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)  # Basic logging setup


class EpisodeData(BaseModel):
    """LLM ENV rollout episode"""

    prompt_tokens: torch.Tensor = Field(..., description="Prompt token ids")
    prompt_text: str = Field(..., description="Prompt full text")
    prompt_length: int = Field(..., description="Prompt token size")
    completion_tokens: torch.Tensor = Field(..., description="Completion token ids")
    completion_text: str = Field(..., description="Completion full text")
    completion_length: int = Field(..., description="Completion token size")
    reward_dict: Dict[str, float] = Field(..., description="Rewards for the episode")
    raw_data: Optional[Dict] = Field(None, description="Raw sample data")

    @model_validator(mode="after")
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

    prompt: List[str] = Field(..., description="Prompt full text")
    input_ids: torch.Tensor = Field(..., description="Prompt token ids")
    attention_mask: torch.Tensor = Field(
        ..., description="Attention mask for the prompt token ids"
    )
    ground_truth: List[str | float | int] = Field(
        ..., description="Ground truth to the problem"
    )
    raw_data: Optional[List[Dict]] = Field(None, description="Raw sample data")

    class Config:
        arbitrary_types_allowed = True


class BaseRewardFunction:
    """
    Base class for reward functions.
    """

    # Define the validation pattern as a constant for clarity
    _VALID_NAME_PATTERN = r"^[a-zA-Z0-9_\-]+$"

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
            raise TypeError(f"Reward function name must be a string, got {type(name)}.")
        if not name:
            raise ValueError("Reward function name cannot be empty.")

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
            "Reward functions must implement the __call__ method."
        )


class LLMEnv:
    """
    Environment for generating LLM training samples with batching and multiple return sequences.

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
        # Add tokenizer args if needed, e.g., max_length
        max_prompt_length: Optional[int] = None,
    ):
        """
        Initializes the LLM Environment.

        Args:
            dataset: The dataset containing prompts and ground truths.
                     Expected to yield dictionaries with 'prompt' and 'ground_truth' keys.
            batch_size: The number of prompts to process in one batch *before* considering num_return_sequences.
            tokenizer: The tokenizer for processing text.
            reward_functions: A list of reward function instances.
            rank: Current rank.
            world_size: World size.
            seed: Optional random seed for reproducibility.
            max_prompt_length: Optional maximum length for tokenized prompts. Defaults to tokenizer's model_max_length.
        """
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if not reward_functions or not all(
            isinstance(fn, BaseRewardFunction) for fn in reward_functions
        ):
            raise ValueError(
                "reward_functions must be a non-empty list of BaseRewardFunction instances"
            )

        self._seed = seed
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)

        self._reward_functions = reward_functions
        self._tokenizer = tokenizer
        # Ensure pad token is set for batching
        if self._tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
            self._tokenizer.pad_token = self._tokenizer.eos_token
            if self._tokenizer.pad_token is None:
                raise ValueError(
                    "Tokenizer needs a pad_token or eos_token for padding."
                )

        self._batch_size = batch_size
        self._world_size = world_size
        self._rank = rank

        shared_dataset = shard_dataset(
            dataset,
            self._world_size,
            self._rank,
        )

        # Use a custom collate_fn if dataset items are not dictionaries
        self._loader = DataLoader(
            shared_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        self._dataset_iterator = iter(self._loader)
        self._max_prompt_length = max_prompt_length or self._tokenizer.model_max_length

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, List]:
        """Collates list of dicts into a dict of lists."""
        collated = {key: [item[key] for item in batch] for key in batch[0]}
        return collated

    def reset(self) -> EnvState:
        """
        Resets the environment by sampling a new batch of data.

        Returns:
            EnvState: The initial state for the new batch.
        """
        try:
            item_batch = next(self._dataset_iterator)
            return self._prepare_initial_state(item_batch)
        except StopIteration:
            logger.info("Dataset iterator exhausted. Resetting DataLoader.")
            self._dataset_iterator = iter(self._loader)
            item_batch = next(self._dataset_iterator)
            return self._prepare_initial_state(item_batch)
        except Exception as e:
            logger.error(f"Error getting next batch: {e}")
            raise

    def _prepare_initial_state(self, item_batch: Dict[str, List]) -> EnvState:
        """
        Prepares the initial EnvState from a batch of data items.

        Args:
            item_batch: A dictionary where keys are 'prompt', 'ground_truth', etc.,
                        and values are lists of corresponding data for the batch.

        Returns:
            EnvState: The prepared initial state for the batch.
        """
        if (
            not isinstance(item_batch, dict)
            or "prompt" not in item_batch
            or "ground_truth" not in item_batch
        ):
            raise ValueError(
                f"Invalid batch data format. Expected dict with 'prompt' and 'ground_truth' lists, got {type(item_batch)}"
            )
        if not isinstance(item_batch["prompt"], list) or not isinstance(
            item_batch["ground_truth"], list
        ):
            raise ValueError(
                "'prompt' and 'ground_truth' values in the batch must be lists."
            )
        if len(item_batch["prompt"]) != len(item_batch["ground_truth"]):
            raise ValueError("Batch size mismatch between 'prompt' and 'ground_truth'.")

        # Ensure prompts are strings
        prompts = [str(p) for p in item_batch["prompt"]]

        # Tokenize the batch of prompts with padding and truncation
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            # Pad to the longest sequence in the batch
            padding="longest" if self._batch_size > 1 else False,
            padding_side="left" if self._batch_size > 1 else None,
            truncation=True,
            max_length=self._max_prompt_length,
            return_attention_mask=True,
        )

        # Store the actual length of the padded prompts
        # prompt_length = inputs["input_ids"].shape[1]

        # Store raw data per sample, not just the whole batch dict
        raw_data_list = []
        batch_keys = list(item_batch.keys())
        num_samples = len(item_batch["prompt"])
        for i in range(num_samples):
            raw_data_list.append({key: item_batch[key][i] for key in batch_keys})

        state = EnvState(
            prompt=prompts,  # List of prompt strings
            input_ids=inputs["input_ids"],  # Tensor (batch_size, prompt_seq_len)
            attention_mask=inputs[
                "attention_mask"
            ],  # Tensor (batch_size, prompt_seq_len)
            ground_truth=item_batch["ground_truth"],  # List of ground truth strings
            raw_data=raw_data_list,  # List of raw data dicts
            # prompt_length=prompt_length,
        )

        return state

    def rollout(self, llm: PreTrainedModel, gen_args: Dict) -> List[EpisodeData]:
        """
        Performs a rollout step: generates completions for a batch of prompts
        and calculates rewards.

        Args:
            llm: The language model used for generation.
            gen_args: Dictionary of arguments passed to `llm.generate()`.
                      Must include `num_return_sequences`.

        Returns:
            List[EpisodeData]: A list containing data for each generated episode
                               (prompt-completion pair with rewards). The total
                               number of episodes is batch_size * num_return_sequences.
        """
        group_size = gen_args.get("num_return_sequences", 1)
        if group_size < 1:
            raise ValueError("num_return_sequences must be at least 1")

        # 1. Get the initial state (batched prompts)
        s_t = self.reset()
        input_batch_size = s_t.input_ids.shape[0]  # Actual batch size processed
        device = llm.device  # Use the model's device

        # 2. Generate sequences
        # Move inputs to the correct device
        input_ids = s_t.input_ids.to(device)
        attention_mask = s_t.attention_mask.to(device)

        # Ensure generation args don't conflict with required args
        gen_args_copy = gen_args.copy()
        gen_args_copy.pop("input_ids", None)
        gen_args_copy.pop("attention_mask", None)

        with torch.no_grad():  # Important for inference
            output = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self._tokenizer.pad_token_id,  # Ensure pad token is used if needed during generation
                **gen_args_copy,
            )

        # Shape: (input_batch_size * group_size, full_sequence_length)
        full_sequences = output.sequences.cpu()  # Move to CPU for decoding/processing
        output_batch_size = full_sequences.shape[0]

        # Verification check
        expected_output_size = input_batch_size * group_size
        if output_batch_size != expected_output_size:
            logger.warning(
                f"Unexpected output batch size from generate(). "
                f"Expected {expected_output_size} (batch={input_batch_size}, num_return={group_size}), "
                f"Got {output_batch_size}. Check generation parameters (e.g., sampling)."
            )
            # Adjust group_size if necessary, though this might indicate an issue
            if output_batch_size % input_batch_size == 0:
                group_size = output_batch_size // input_batch_size
                logger.warning(f"Adjusting group_size to {group_size} based on output.")
            else:
                # This case is problematic, maybe raise error or handle differently
                raise RuntimeError(
                    f"Output batch size {output_batch_size} is not divisible by input batch size {input_batch_size}."
                )

        # 3. Process outputs
        prompt_length = s_t.input_ids.shape[1]  # Use the stored padded prompt length

        # Shape: (output_batch_size, completion_length)
        # Ensure slicing doesn't go out of bounds if generation is shorter than prompt
        completion_ids = full_sequences[
            :, min(prompt_length, full_sequences.shape[1]) :
        ]

        # Calculate actual completion lengths (excluding padding)
        # Assuming pad_token_id is used for padding *after* generation ends
        completion_lengths = (completion_ids != self._tokenizer.pad_token_id).sum(dim=1)

        # Decode completions
        completion_texts = self._tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # 4. Prepare data for reward calculation and EpisodeData construction
        # Expand original batch data to match the output batch size
        expanded_prompts = []
        expanded_ground_truths = []
        expanded_raw_data = []
        expanded_prompt_tokens = []  # Store original prompt tokens per output

        for i in range(input_batch_size):
            for _ in range(group_size):
                expanded_prompts.append(s_t.prompt[i])
                expanded_ground_truths.append(s_t.ground_truth[i])
                expanded_raw_data.append(s_t.raw_data[i])
                # Get the original (potentially unpadded) prompt tokens for this sample
                # We use the input_ids before padding/truncation if available,
                # otherwise, use the padded ones and slice based on attention mask?
                # Simplest: just store the padded input_ids for the corresponding prompt.
                expanded_prompt_tokens.append(s_t.input_ids[i].cpu())  # Store on CPU

        # Verify expanded list lengths
        assert len(expanded_prompts) == output_batch_size
        assert len(expanded_ground_truths) == output_batch_size
        assert len(expanded_raw_data) == output_batch_size
        assert len(expanded_prompt_tokens) == output_batch_size

        # 5. Calculate rewards
        reward_dict_batch = {}  # Stores rewards for the entire output batch
        for reward_fn in self._reward_functions:
            try:
                # Pass expanded lists matching the completions
                rewards = reward_fn(
                    completion_texts,  # List[str] (output_batch_size)
                    expanded_ground_truths,  # List[str] (output_batch_size)
                )
                if not isinstance(rewards, list) or len(rewards) != output_batch_size:
                    raise ValueError(
                        f"Reward function '{reward_fn.name}' did not return a list of size {output_batch_size}"
                    )
                reward_dict_batch[reward_fn.name] = rewards
            except Exception as e:
                logger.error(f"Error calculating reward with {reward_fn.name}: {e}")
                # Handle error, e.g., assign default reward or re-raise
                reward_dict_batch[reward_fn.name] = [
                    0.0
                ] * output_batch_size  # Example: default reward

        # 6. Construct EpisodeData for each generated sequence
        results: List[EpisodeData] = []
        for i in range(output_batch_size):
            # Get the original prompt tokens corresponding to this output
            # original_batch_idx = i // group_size # Index into the original s_t batch
            # prompt_tokens_for_sample = s_t.input_ids[original_batch_idx].cpu()

            # Ensure completion tokens don't include padding beyond the actual length
            actual_completion_len = completion_lengths[i].item()
            completion_tokens_for_sample = completion_ids[i, :actual_completion_len]

            sample = EpisodeData(
                prompt_text=expanded_prompts[i],
                prompt_tokens=expanded_prompt_tokens[
                    i
                ],  # Padded prompt tokens from input
                prompt_length=prompt_length,  # Padded length
                completion_text=completion_texts[i],
                completion_tokens=completion_tokens_for_sample,  # Tokens up to actual length
                completion_length=actual_completion_len,
                reward_dict={k: v[i] for k, v in reward_dict_batch.items()},
                raw_data=expanded_raw_data[i],  # Raw data for the original prompt
            )
            results.append(sample)

        return results
