"""Implements base MDP ENV for collect samples for RL"""

import logging
import random
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from rl4llm.constants import LOGGER_NAME
from rl4llm.utils.dataset_utils import shard_dataset

logger = logging.getLogger(LOGGER_NAME)


RewardTransform: TypeAlias = Optional[
    Callable[[Dict[str, List[float]]], torch.Tensor]
]


class ChatMessage(BaseModel):
    role: str = Field(..., description='Role of the chat turn')
    content: str = Field(..., description='Chat content')
    # tool_calls: Optional[List[Dict]] = None
    # tool_call_id: Optional[str] = None

    @model_validator(mode='after')
    def check_role(cls, model_instance):
        supported_roles = ['user', 'assistant', 'system', 'tool']
        if model_instance.role not in supported_roles:
            raise ValueError(
                f"Invalid role {model_instance.role}, only support {supported_roles}"
            )
        return model_instance


class EpisodeData(BaseModel):
    """LLM ENV rollout episode"""

    states: torch.Tensor = Field(
        ..., description='Token sequences from t=0 to T-1'
    )
    actions: torch.Tensor = Field(
        ..., description='Token sequences from t=1 to T'
    )
    loss_mask: torch.Tensor = Field(
        ..., description='Mask for completion tokens (t=1 to T)'
    )
    terminal_reward: float = Field(..., description='Final transformed reward')
    ground_truth: Union[str, float, int] = Field(
        ..., description='Ground truth'
    )
    reward_dict: Dict[str, float] = Field(..., description='Individual rewards')
    chat_history: List[ChatMessage] = Field(
        ..., description='Full chat history'
    )
    prompt_length: int = Field(..., description='Initial prompt token size')
    completion_length: int = Field(
        ..., description='Generated completion token size'
    )
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @model_validator(mode='after')
    def check_tensor_shapes(cls, model_instance):
        if (
            model_instance.states.shape != model_instance.actions.shape
            or model_instance.states.shape != model_instance.loss_mask.shape
        ):
            raise ValueError(
                f"Tensor shape mismatch: states={model_instance.states.shape}, "
                f"actions={model_instance.actions.shape}, loss_mask={model_instance.loss_mask.shape}"
            )
        return model_instance

    class Config:
        arbitrary_types_allowed = True


class SampleState(BaseModel):
    """Represents the state of a single sample during interaction."""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4().hex),
        description='Unique ID of the sample',
    )
    messages: List[ChatMessage] = Field(
        ..., description='Current message history for this sample'
    )
    ground_truth: Union[str, float, int] = Field(
        ..., description='Ground truth for this sample'
    )
    init_msg_size: int = Field(
        ..., description='Number of messages in the initial prompt'
    )
    current_step: int = Field(
        default=0, description='Number of interaction steps taken'
    )
    done: bool = Field(
        default=False,
        description='Whether this sample has finished interaction',
    )

    class Config:
        arbitrary_types_allowed = True


class EnvState(BaseModel):
    """Environment state holding individual states for each sample in the batch."""

    sample_states: List[SampleState] = Field(
        ..., description='List of individual sample states'
    )

    class Config:
        arbitrary_types_allowed = True


class BaseRewardFunction(ABC):
    """Base class for reward functions."""

    _VALID_NAME_PATTERN = r'^[a-zA-Z0-9_\-]+$'

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError(
                f"Reward function name must be a string, got {type(name)}."
            )
        if not name:
            raise ValueError('Reward function name cannot be empty.')
        if not re.match(self._VALID_NAME_PATTERN, name):
            raise ValueError(
                f"Invalid reward function name: '{name}'. Pattern: '{self._VALID_NAME_PATTERN}'"
            )
        self.name = name

    @abstractmethod
    def __call__(
        self,
        messages: List[ChatMessage],
        ground_truth: Union[str, float, int],
        **kwargs: Any,
    ) -> float:
        """
        Calculates reward based on the final state (full chat history) for a single sample.

        Args:
            messages: The full chat history for a single sample.
            ground_truth: The corresponding ground truth for the sample.
            **kwargs: Additional data.

        Returns:
            float: Scalar reward for the sample.
        """
        raise NotImplementedError


# Helper function
def find_subsequence(main_list, sub_list):
    """Finds the start index of the first occurrence of sub_list within main_list."""
    if not sub_list or not main_list:
        return -1
    len_sub = len(sub_list)
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i : i + len_sub] == sub_list:
            return i
    return -1


class BaseMDPEnv(ABC):
    """
    Base MDP Environment for generating training samples with LLM models.

    Handles common functionalities like data loading, batching, sharding,
    reward calculation, and final data conversion.

    Subclasses should primarily override `_run_interaction_loop` to define
    the specific generation process (single-step, multi-step, tool use).
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        reward_functions: List[BaseRewardFunction],
        batch_size: int,
        group_size: int,
        max_steps: int = 1,
        rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
        seed: Optional[int] = 42,
        shuffle_dataset: Optional[bool] = True,
        num_workers: Optional[int] = 0,
        reward_transform_fn: Optional[RewardTransform] = None,
    ):
        if world_size < 1:
            raise ValueError('world_size must be >= 1')
        if rank >= world_size:
            raise ValueError('Rank must be less than world_size')
        if batch_size < 1:
            raise ValueError('Batch size must be >= 1')
        if group_size < 1:
            raise ValueError('Group size must be >= 1')
        if max_steps < 1:
            raise ValueError('Max steps must be >= 1')
        if not reward_functions:
            raise ValueError('reward_functions cannot be empty')
        if not all(
            isinstance(fn, BaseRewardFunction) for fn in reward_functions
        ):
            raise ValueError(
                'All reward_functions must be instances of BaseRewardFunction'
            )
        if not isinstance(dataset, Dataset):
            raise TypeError('dataset must be a datasets.Dataset')
        if not all(
            col in dataset.column_names for col in ['messages', 'ground_truth']
        ):
            raise ValueError(
                "Dataset needs 'messages' and 'ground_truth' columns."
            )
        # Ensure 'messages' column contains lists of ChatMessage-like dicts
        if not isinstance(dataset[0]['messages'], list) or not all(
            isinstance(m, dict) and 'role' in m and 'content' in m
            for m in dataset[0]['messages']
        ):
            raise ValueError(
                "'messages' column should contain lists of {'role': str, 'content': str} dicts."
            )
        if len(reward_functions) > 1 and not reward_transform_fn:
            raise ValueError(
                'Multiple reward functions provided without a reward_transform_fn.'
            )

        self.reward_transform_fn = (
            reward_transform_fn
            if reward_transform_fn
            else lambda r_dict: torch.tensor(list(r_dict.values())[0])
        )

        self.tokenizer = tokenizer
        self.reward_functions = reward_functions
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_steps = max_steps  # Store max interaction steps
        self.rank = rank
        self.world_size = world_size
        self.seed = seed + rank
        self.shuffle_dataset = shuffle_dataset
        self.num_workers = num_workers
        self.epoch = 0

        self._setup_tokenizer()
        self.random_state = np.random.RandomState(self.seed)
        self.sharded_dataset = shard_dataset(
            dataset, self.world_size, self.rank
        )
        logger.info(
            f"Env - Rank {self.rank} has {len(self.sharded_dataset)} samples"
        )

        self.loader = DataLoader(
            self.sharded_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_dataset,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )
        self.dataset_iterator = iter(self.loader)

    def _seed_worker(self, worker_id):
        """Ensure reproducibility in dataloader workers."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _setup_tokenizer(self):
        """Configures the tokenizer."""
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            logger.warning('Tokenizer missing pad token; using eos_token.')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer needs pad_token_id.')
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        # Ensure chat template exists
        try:
            _ = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': 'test'}], tokenize=False
            )
        except Exception as e:
            raise ValueError(
                'Tokenizer must have a chat template defined. '
                f"Error during template test: {e}"
            )

    def _collate_fn(self, batch_list: List[Dict]) -> Dict[str, Any]:
        """Collates samples without padding yet."""
        if not batch_list:
            return {}
        # Simple collation, assumes dataset provides dicts
        keys = batch_list[0].keys()
        collated = {k: [item[k] for item in batch_list] for k in keys}
        return collated

    def _get_next_batch(self):
        """Fetches the next batch from the DataLoader, resetting if exhausted."""
        try:
            return next(self.dataset_iterator)
        except StopIteration:
            logger.info(
                f"Rank {self.rank}: Dataset exhausted. Resetting DataLoader."
            )
            self.epoch += 1
            self.dataset_iterator = iter(self.loader)
            try:
                return next(self.dataset_iterator)
            except StopIteration:
                logger.error(f"Rank {self.rank}: DataLoader empty after reset.")
                return None

    def _reset(self):
        """Resets the environment by sampling a new batch and preparing the initial state."""
        try:
            raw_batch = self._get_next_batch()
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error getting batch", exc_info=True
            )
            raise e
        if raw_batch is None:
            return None
        try:
            return self._prepare_initial_state(raw_batch)
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error preparing initial state from batch",
                exc_info=True,
            )
            raise e

    def _prepare_initial_state(self, raw_batch: Dict[str, Any]) -> EnvState:
        """Prepares EnvState containing a list of SampleState objects."""
        num_samples = len(raw_batch['ground_truth'])
        if num_samples == 0:
            logger.warning(f"Rank {self.rank}: Empty batch received.")
            return self._reset()  # Or handle appropriately

        sample_states = []
        for i in range(num_samples):
            try:
                # Parse messages for the original sample
                parsed_msgs = [
                    ChatMessage(**msg) for msg in raw_batch['messages'][i]
                ]
                ground_truth = raw_batch['ground_truth'][i]
                init_msg_count = len(parsed_msgs)

                # Repeat group_size times
                for _ in range(self.group_size):
                    # Create a deep copy of messages for each SampleState
                    # to avoid shared mutable state issues.
                    initial_messages_copy = [
                        msg.model_copy(deep=True) for msg in parsed_msgs
                    ]

                    sample_state = SampleState(
                        messages=initial_messages_copy,
                        ground_truth=ground_truth,
                        init_msg_size=init_msg_count,
                        current_step=0,
                        done=False,
                    )
                    sample_states.append(sample_state)

            except Exception as e:
                logger.error(
                    f"Failed to parse messages or create SampleState for raw sample {i}: {raw_batch['messages'][i]}. Error: {e}"
                )
                continue  # Or raise an exception ???

        if not sample_states:
            logger.error(
                f"Rank {self.rank}: No valid SampleStates created from the batch."
            )
            return None  # Or raise an exception ???

        return EnvState(sample_states=sample_states)

    def _calculate_rewards(
        self, batch_sample_states: List[SampleState]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculates rewards using configured functions."""
        num_samples = len(batch_sample_states)
        if num_samples == 0:
            return torch.empty(0, dtype=torch.float32), {}

        # Initialize rewards_dict with zero tensors
        rewards_dict: Dict[str, torch.Tensor] = {
            fn.name: torch.zeros(num_samples, dtype=torch.float32)
            for fn in self.reward_functions
        }

        # Calculate rewards
        for i, sample_state in enumerate(batch_sample_states):
            for fn in self.reward_functions:
                reward = fn(sample_state.messages, sample_state.ground_truth)
                if not isinstance(reward, (float, int)):
                    raise ValueError(
                        f"Reward func '{fn.name}' must return float/int, got {type(reward)}"
                    )
                rewards_dict[fn.name][i] = float(reward)

        return self._transform_rewards(rewards_dict), rewards_dict

    def _transform_rewards(
        self, rewards_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Transforms multiple rewards into a single tensor."""
        if not rewards_dict:
            return torch.zeros(
                self.batch_size * self.group_size, dtype=torch.float32
            )

        if self.reward_transform_fn:
            transformed = self.reward_transform_fn(rewards_dict)
            if not isinstance(transformed, torch.Tensor):
                transformed = torch.tensor(transformed, dtype=torch.float32)

            expected_len = len(next(iter(rewards_dict.values())))
            if transformed.shape != (expected_len,):
                raise ValueError(
                    f"Expected shape ({expected_len},), got {transformed.shape}"
                )
            return transformed.float()

        return next(iter(rewards_dict.values()))

    def _convert_to_episodes(self, env_state: EnvState) -> List[EpisodeData]:
        """Converts the final list of SampleStates into EpisodeData list."""

        if not env_state.sample_states:
            return []

        # 1. Calculate rewards in batch
        batch_terminal_rewards_tensor, batch_rewards_dict = (
            self._calculate_rewards(env_state.sample_states)
        )

        num_samples = len(env_state.sample_states)
        if batch_terminal_rewards_tensor.size(0) != num_samples:
            logger.error(
                f"Mismatch between number of sample states ({num_samples}) "
                f"and number of terminal rewards ({batch_terminal_rewards_tensor.size(0)}). "
                'Skipping episode conversion for this batch.'
            )
            return []

        batch_terminal_rewards = batch_terminal_rewards_tensor.tolist()

        # 2. Create EpisodeData for each sample
        results = []
        for i, sample_state in enumerate(env_state.sample_states):
            terminal_reward = batch_terminal_rewards[i]
            reward_dict = {k: v[i] for k, v in batch_rewards_dict.items()}

            # --- Apply tokenization and masking logic PER SAMPLE ---
            try:
                if len(sample_state.messages) < 2:
                    raise RuntimeError('Sample resulted in messages length < 2')

                messages_as_dicts = [
                    msg.model_dump() for msg in sample_state.messages
                ]
                # We use 'continue_final_message' to skip add EOS to the last turn
                # in some cases the chat template will add other tokens after the EOS token
                formatted_chat_history = self.tokenizer.apply_chat_template(
                    messages_as_dicts,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
                full_sequence_ids = self.tokenizer.encode(
                    formatted_chat_history, add_special_tokens=False
                )

                if not isinstance(full_sequence_ids, list):
                    full_sequence_ids = full_sequence_ids.squeeze(0).tolist()

                # Ensure the last assistant's turn ends with EOS token id
                if full_sequence_ids[-1] != self.tokenizer.eos_token_id:
                    full_sequence_ids.append(self.tokenizer.eos_token_id)

                loss_mask = [0] * len(full_sequence_ids)
                # Ensure we use the last EOS token id for training
                loss_mask[-1] = 1
                prompt_token_len = -1
                current_pos = 0

                for msg_idx, msg in enumerate(messages_as_dicts):
                    prefix_msg = self.tokenizer.apply_chat_template(
                        messages_as_dicts[: msg_idx + 1],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    prefix_tokens = self.tokenizer.encode(
                        prefix_msg, add_special_tokens=False
                    )

                    if not isinstance(prefix_tokens, list):
                        prefix_tokens = prefix_tokens.squeeze(0).tolist()
                    message_tokens = prefix_tokens[current_pos:]

                    if msg_idx == sample_state.init_msg_size - 1:
                        prompt_token_len = len(prefix_tokens)

                    # Only process assistant's turns after the initial prompt messages
                    if (
                        msg['role'] == 'assistant'
                        and msg_idx >= sample_state.init_msg_size
                    ):
                        # --- Masking llm generated content ---
                        content = msg.get('content')
                        if content:
                            content_tokens = self.tokenizer.encode(
                                content, add_special_tokens=False
                            )
                            # Is this the most reliable way of construct the loss mask???
                            if content_tokens:
                                # # Also add EOS to intermediate turns from assistant's generation
                                # if (
                                #     content_tokens[-1]
                                #     != self.tokenizer.eos_token_id
                                # ):
                                #     content_tokens.append(
                                #         self.tokenizer.eos_token_id
                                #     )

                                content_start_in_msg = find_subsequence(
                                    message_tokens, content_tokens
                                )
                                if content_start_in_msg != -1:
                                    global_content_start = (
                                        current_pos + content_start_in_msg
                                    )
                                    global_content_end = (
                                        global_content_start
                                        + len(content_tokens)
                                    )
                                    for j in range(
                                        global_content_start, global_content_end
                                    ):
                                        if j < len(loss_mask):
                                            loss_mask[j] = 1
                                else:
                                    raise RuntimeError(
                                        f"Could not precisely locate content tokens for assistant msg {msg_idx}."
                                    )

                    current_pos = len(prefix_tokens)
                # --- End Masking Logic ---

                if prompt_token_len == -1:
                    prompt_token_len = 0  # Handle cases with no prompt messages

                full_sequence_tensor = torch.tensor(
                    full_sequence_ids, dtype=torch.long
                )
                loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.bool)

                states = full_sequence_tensor[:-1]
                actions = full_sequence_tensor[1:]
                final_loss_mask = loss_mask_tensor[1:]
                completion_len = final_loss_mask.sum().item()

                ep = EpisodeData(
                    states=states,
                    actions=actions,
                    loss_mask=final_loss_mask,
                    terminal_reward=terminal_reward,
                    reward_dict=reward_dict,
                    ground_truth=sample_state.ground_truth,
                    chat_history=sample_state.messages,
                    prompt_length=prompt_token_len,
                    completion_length=completion_len,
                )
                results.append(ep)

            except Exception as e:
                raise RuntimeError(
                    f"Failed converting Sample to EpisodeData: {e}",
                )

        return results

    def _convert_to_batch_prompts(self, env_state: EnvState) -> List[str]:
        """Converts a batch messages to chat-style prompt for generation"""

        batch_prompts = []
        for d in env_state.sample_states:
            messages = d.messages
            # Merge consecutive assistant's turns into a single turn
            if not messages:
                merged_messages = []
            else:
                merged_messages = [messages[0]]  # Start with the first message
                for i in range(1, len(messages)):
                    current_msg_obj = messages[i]
                    last_merged_obj = merged_messages[-1]

                    if (
                        current_msg_obj.role == 'assistant'
                        and last_merged_obj.role == 'assistant'
                    ):
                        # Merge content. Using a newline as a separator.
                        merged_content = (
                            last_merged_obj.content + current_msg_obj.content
                        )
                        # Replace the last message object with a new one containing the merged content
                        merged_messages[-1] = ChatMessage(
                            role='assistant', content=merged_content
                        )
                        logger.debug(
                            f"Merged assistant turn: '{last_merged_obj.content}' + '{current_msg_obj.content}' -> '{merged_content}'"
                        )
                    else:
                        merged_messages.append(current_msg_obj)

            # Convert merged messages to dictionaries
            messages_as_dicts = [msg.model_dump() for msg in merged_messages]

            # Handle continue generation if the message ends with assistant's turn
            continue_gen = False
            if merged_messages[-1].role == 'assistant':
                continue_gen = True

            # # A simple hack to avoid add default system prompt
            # # Note: This will still render system tokens if the template includes them,
            # # but the content part of the system prompt will be empty.
            # # A better way might be to set the chat_template for the tokenizer
            # has_system_prompt = (
            #     messages_as_dicts and messages_as_dicts[0]['role'] == 'system'
            # )
            # if not has_system_prompt:
            #     messages_as_dicts = [
            #         {'role': 'system', 'content': ''}
            #     ] + messages_as_dicts

            prompt = self.tokenizer.apply_chat_template(
                messages_as_dicts,
                tokenize=False,
                add_generation_prompt=not continue_gen,
                continue_final_message=continue_gen,
            )
            batch_prompts.append(prompt)

        return batch_prompts

    @torch.inference_mode()
    def _run_interaction_loop(
        self,
        env_state: EnvState,
        llm: Any,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        """
        LLM interaction loop for single-step or multi-step MDPs or tool use.

        Args:
            env_state: The starting state containing a list of SampleState objects.
            llm: The language model or inference client.
            generation_config: Configuration for generation (max_new_tokens, etc.).
            **kwargs: Additional arguments (unused in default).

        Returns:
            EnvState: The final state after generation, with updated SampleState.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def rollout(
        self,
        llm: Any,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[EpisodeData]:
        """
        Performs a rollout: resets env, runs interaction, converts to episodes.

        Args:
            llm: The HF language model or inference client.
            sampling_params: Sampling parameters for generation.
            **kwargs: Additional custom arguments passed to _run_interaction_loop.

        Returns:
            List[EpisodeData]: Data for each sample in the batch. Empty list if dataset exhausted.
        """
        # 1. Get initial state for the batch
        initial_state = self._reset()
        if initial_state is None:
            logger.warning(
                f"Rank {self.rank}: Reset returned None, likely end of dataset."
            )
            return []

        # 2. Run the interaction loop, should be handled by the subclass
        try:
            final_state = self._run_interaction_loop(
                initial_state, llm, sampling_params, **kwargs
            )

            # Ensure every sample state's messages ends with assistant's turn
            if any(
                [
                    d.messages[-1].role != 'assistant'
                    for d in final_state.sample_states
                ]
            ):
                raise RuntimeError(
                    f"Rank {self.rank}: Error during _run_interaction_loop: expect all messages ends with assistant's turn"
                )
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error during _run_interaction_loop: {e}",
                exc_info=True,
            )
            return []

        # 3. Convert the final state to training episodes
        try:
            episodes = self._convert_to_episodes(final_state)
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error during _convert_to_episodes: {e}",
                exc_info=True,
            )
            # Decide how to handle: return empty, partial, or raise?
            return []

        logger.debug(
            f"Rank {self.rank}: Rollout generated {len(episodes)} episodes."
        )
        return episodes
