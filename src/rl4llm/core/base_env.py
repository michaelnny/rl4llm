"""Implements base MDP ENV for collect samples for RL"""

import logging
import random
import re
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
    TypedDict,
    Union,
)

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from rl4llm.utils.dataset_utils import shard_dataset

logger = logging.getLogger(__name__)


RewardTransform: TypeAlias = Optional[Callable[[Dict[str, List[float]]], torch.Tensor]]


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the chat turn")
    content: str = Field(..., description="Chat content")
    # Add optional fields for tool use if needed later
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    @model_validator(mode="after")
    def check_role(cls, values):
        # Allow 'tool' role if implementing tool use later
        supported_roles = ["user", "assistant", "system", "tool"]
        if values.role not in supported_roles:
            raise ValueError(
                f"Invalid role {values.role}, only support {supported_roles}"
            )
        return values

    class Config:
        # Allow extra fields if needed for tool calls etc.
        extra = "allow"


class EpisodeData(BaseModel):
    """LLM ENV rollout episode"""

    states: torch.Tensor = Field(..., description="Token sequences from t=0 to T-1")
    actions: torch.Tensor = Field(..., description="Token sequences from t=1 to T")
    loss_mask: torch.Tensor = Field(
        ..., description="Mask for completion tokens (t=1 to T)"
    )
    terminal_reward: float = Field(..., description="Final transformed reward")
    ground_truth: Union[str, float, int] = Field(..., description="Ground truth")
    reward_dict: Dict[str, float] = Field(..., description="Individual rewards")
    chat_history: List[ChatMessage] = Field(..., description="Full chat history")
    prompt_length: int = Field(..., description="Initial prompt token size")
    completion_length: int = Field(..., description="Generated completion token size")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())

    @model_validator(mode="after")
    def check_tensor_shapes(cls, values):
        if (
            values.states.shape != values.actions.shape
            or values.states.shape != values.loss_mask.shape
        ):
            raise ValueError(
                f"Tensor shape mismatch: states={values.states.shape}, "
                f"actions={values.actions.shape}, loss_mask={values.loss_mask.shape}"
            )
        return values

    class Config:
        arbitrary_types_allowed = True


class EnvState(BaseModel):
    """Environment state for LLM generation"""

    # Represents the state *before* the next generation step
    batch_messages: List[List[ChatMessage]] = Field(
        ..., description="Batch list of chat messages"
    )
    batch_ground_truth: List[str | float | int] = Field(
        ..., description="Batch list of ground truth"
    )
    batch_init_prompt_size: List[int] = Field(
        ..., description="Batch list of initial prompt message count"
    )
    # Optional: Add fields needed for multi-step control (e.g., current step, done flags)
    batch_done: Optional[List[bool]] = None
    batch_current_step: Optional[List[int]] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for subclasses


class BaseRewardFunction(ABC):  # Made abstract
    """Base class for reward functions."""

    _VALID_NAME_PATTERN = r"^[a-zA-Z0-9_\-]+$"

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Reward function name must be a string, got {type(name)}.")
        if not name:
            raise ValueError("Reward function name cannot be empty.")
        if not re.match(self._VALID_NAME_PATTERN, name):
            raise ValueError(
                f"Invalid reward function name: '{name}'. Pattern: '{self._VALID_NAME_PATTERN}'"
            )
        self.name = name

    @abstractmethod
    def __call__(
        self,
        batch_messages: List[List[ChatMessage]],
        batch_ground_truths: List[Union[str, float, int]],
        **kwargs: Any,
    ) -> List[float]:
        """
        Calculates rewards based on the final state (full chat history).

        Args:
            batch_messages: List where each element is the full chat history for a sample.
            batch_ground_truths: Corresponding ground truths.
            **kwargs: Additional data.

        Returns:
            List[float]: Scalar rewards for each sample in the batch.
        """
        raise NotImplementedError


# Helper function (assuming it exists or is defined elsewhere)
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
        max_steps: int = 1,  # Max interaction steps (for multi-step control)
        rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
        seed: Optional[int] = 42,
        shuffle_dataset: Optional[bool] = True,
        num_workers: Optional[int] = 0,
        reward_transform_fn: Optional[RewardTransform] = None,
    ):
        if batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        if group_size < 1:
            raise ValueError("Group size must be >= 1")
        if max_steps < 1:
            raise ValueError("Max steps must be >= 1")
        if not reward_functions:
            raise ValueError("reward_functions cannot be empty")
        if not all(isinstance(fn, BaseRewardFunction) for fn in reward_functions):
            raise ValueError(
                "All reward_functions must be instances of BaseRewardFunction"
            )
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a datasets.Dataset")
        if not all(col in dataset.column_names for col in ["messages", "ground_truth"]):
            raise ValueError("Dataset needs 'messages' and 'ground_truth' columns.")
        # Ensure 'messages' column contains lists of ChatMessage-like dicts
        if not isinstance(dataset[0]["messages"], list) or not all(
            isinstance(m, dict) and "role" in m and "content" in m
            for m in dataset[0]["messages"]
        ):
            raise ValueError(
                "'messages' column should contain lists of {'role': str, 'content': str} dicts."
            )
        if len(reward_functions) > 1 and not reward_transform_fn:
            raise ValueError(
                "Multiple reward functions provided without a reward_transform_fn."
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
        self.sharded_dataset = shard_dataset(dataset, self.world_size, self.rank)
        logger.info(f"Env - Rank {self.rank} has {len(self.sharded_dataset)} samples")

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
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            logger.warning("Tokenizer missing pad token; using eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id.")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        # Ensure chat template exists
        try:
            _ = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}], tokenize=False
            )
        except Exception as e:
            raise ValueError(
                "Tokenizer must have a chat template defined. "
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
            logger.info(f"Rank {self.rank}: Dataset exhausted. Resetting DataLoader.")
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
            logger.error(f"Rank {self.rank}: Error getting batch", exc_info=True)
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
        """Prepares EnvState, repeating items `group_size` times."""
        num_samples = len(raw_batch["ground_truth"])
        if num_samples == 0:
            logger.warning(f"Rank {self.rank}: Empty batch received.")
            # Attempt to get another batch
            return self._reset()

        # Validate and parse messages
        batch_messages = []
        for msg_list in raw_batch["messages"]:
            try:
                # Convert dicts to ChatMessage objects for validation
                parsed_msgs = [ChatMessage(**msg) for msg in msg_list]
                batch_messages.append(parsed_msgs)
            except Exception as e:
                logger.error(f"Failed to parse messages: {msg_list}. Error: {e}")
                raise ValueError(f"Invalid message format in dataset: {e}")

        expanded_messages = [p for p in batch_messages for _ in range(self.group_size)]
        expanded_ground_truths = [
            gt for gt in raw_batch["ground_truth"] for _ in range(self.group_size)
        ]

        # Store the *count* of initial messages, not token length yet
        init_prompt_size = [len(p) for p in expanded_messages]

        return EnvState(
            batch_messages=expanded_messages,
            batch_ground_truth=expanded_ground_truths,
            batch_init_prompt_size=init_prompt_size,
        )

    def _calculate_rewards(
        self, batch_messages: List[List[ChatMessage]], batch_ground_truths: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Calculates rewards using configured functions."""
        if len(batch_messages) != len(batch_ground_truths):
            raise ValueError(
                f"Reward input size mismatch: {len(batch_messages)} vs {len(batch_ground_truths)}"
            )

        rewards_dict = {}
        for fn in self.reward_functions:
            try:
                # Pass ChatMessage objects directly
                rewards = fn(batch_messages, batch_ground_truths)
                if not isinstance(rewards, list) or len(rewards) != len(batch_messages):
                    raise ValueError(f"Reward func '{fn.name}' output mismatch.")
                rewards_dict[fn.name] = rewards
            except Exception as e:
                logger.error(f"Reward func '{fn.name}' failed: {e}", exc_info=True)
                # Provide default reward on failure
                rewards_dict[fn.name] = [0.0] * len(batch_messages)

        terminal_reward_tensor = self._transform_rewards(rewards_dict)
        return terminal_reward_tensor, rewards_dict

    def _transform_rewards(self, rewards_dict: Dict[str, List[float]]) -> torch.Tensor:
        """Transforms multiple rewards into a single tensor."""
        if not rewards_dict:
            return torch.zeros(self.batch_size * self.group_size)  # Handle empty case

        if self.reward_transform_fn:
            try:
                transformed = self.reward_transform_fn(rewards_dict)
                if not isinstance(transformed, torch.Tensor):
                    # Try converting if it's list/numpy
                    try:
                        transformed = torch.tensor(transformed, dtype=torch.float32)
                    except Exception:
                        raise TypeError(
                            f"Reward transform function must return a torch.Tensor, got {type(transformed)}"
                        )
                # Ensure shape matches batch size
                expected_len = len(next(iter(rewards_dict.values())))
                if transformed.shape != (expected_len,):
                    raise ValueError(
                        f"Transformed reward shape mismatch. Expected ({expected_len},), got {transformed.shape}"
                    )
                return transformed.float()  # Ensure float
            except Exception as e:
                logger.error(f"Reward transformation failed: {e}", exc_info=True)
                # Fallback: use the first reward function's output
                first_reward_key = next(iter(rewards_dict.keys()))
                logger.warning(
                    f"Falling back to using reward '{first_reward_key}' due to transform error."
                )
                return torch.tensor(rewards_dict[first_reward_key], dtype=torch.float32)
        else:
            # Should have been handled in __init__, but as a safeguard
            first_reward_key = next(iter(rewards_dict.keys()))
            return torch.tensor(rewards_dict[first_reward_key], dtype=torch.float32)

    def _convert_to_episodes(self, final_state: EnvState) -> List[EpisodeData]:
        """Converts the final state after interaction into EpisodeData list."""

        # Calculate rewards based on the final messages
        batch_terminal_rewards_tensor, batch_rewards_dict = self._calculate_rewards(
            final_state.batch_messages, final_state.batch_ground_truth
        )
        # Convert tensor back to list for easier per-sample processing
        batch_terminal_rewards = batch_terminal_rewards_tensor.tolist()

        results = []
        effective_batch_size = len(final_state.batch_messages)

        for i in range(effective_batch_size):
            messages = final_state.batch_messages[i]  # List[ChatMessage]
            ground_truth = final_state.batch_ground_truth[i]
            init_prompt_msg_count = final_state.batch_init_prompt_size[i]
            terminal_reward = batch_terminal_rewards[i]
            # Extract per-sample reward dict
            reward_dict = {k: v[i] for k, v in batch_rewards_dict.items()}

            # Convert ChatMessage objects back to dicts for apply_chat_template if needed
            # (Some tokenizers might expect dicts, others might handle Pydantic models)
            messages_as_dicts = [msg.model_dump() for msg in messages]

            try:
                # 1. Tokenize the entire conversation
                full_sequence_ids = self.tokenizer.apply_chat_template(
                    messages_as_dicts, tokenize=True, add_generation_prompt=False
                )
                if not isinstance(full_sequence_ids, list):
                    full_sequence_ids = full_sequence_ids.tolist()  # Ensure it's a list

                # 2. Initialize loss mask
                loss_mask = [0] * len(full_sequence_ids)

                # 3. Identify assistant content tokens for loss calculation
                current_pos = 0
                prompt_token_len = -1  # Track prompt length in tokens

                for msg_idx, msg in enumerate(messages_as_dicts):
                    # Tokenize sequence up to *including* current message
                    prefix_tokens = self.tokenizer.apply_chat_template(
                        messages_as_dicts[: msg_idx + 1],
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    if not isinstance(prefix_tokens, list):
                        prefix_tokens = prefix_tokens.tolist()

                    message_tokens = prefix_tokens[current_pos:]

                    # Record prompt token length after processing the last prompt message
                    if msg_idx == init_prompt_msg_count - 1:
                        prompt_token_len = len(prefix_tokens)

                    # Mask loss only for assistant messages *after* the initial prompt
                    if msg["role"] == "assistant" and msg_idx >= init_prompt_msg_count:
                        content = msg.get("content")
                        if (
                            content
                        ):  # Handle potential empty content (e.g., tool call start)
                            # Tokenize content *without* special tokens
                            content_tokens = self.tokenizer.encode(
                                content, add_special_tokens=False
                            )

                            if content_tokens:
                                # Find content within the message's tokens
                                content_start_in_msg = find_subsequence(
                                    message_tokens, content_tokens
                                )

                                if content_start_in_msg != -1:
                                    global_content_start = (
                                        current_pos + content_start_in_msg
                                    )
                                    global_content_end = global_content_start + len(
                                        content_tokens
                                    )

                                    # Set mask to 1 for these tokens
                                    for j in range(
                                        global_content_start, global_content_end
                                    ):
                                        if j < len(loss_mask):
                                            loss_mask[j] = 1
                                        else:
                                            logger.warning(
                                                f"Index {j} out of bounds for loss_mask (len {len(loss_mask)})."
                                            )
                                else:
                                    # This is tricky - chat templates add roles/separators.
                                    # If find_subsequence fails, it might be due to template structure.
                                    # A less precise but often workable fallback: assume the *end* of message_tokens corresponds to content.
                                    # This heuristic might incorrectly mask template tokens if content is short.
                                    approx_content_start = len(message_tokens) - len(
                                        content_tokens
                                    )
                                    if approx_content_start >= 0:
                                        logger.warning(
                                            f"Could not precisely locate content tokens for assistant msg {msg_idx}. Using end-of-message heuristic."
                                        )
                                        global_content_start = (
                                            current_pos + approx_content_start
                                        )
                                        global_content_end = global_content_start + len(
                                            content_tokens
                                        )
                                        for j in range(
                                            global_content_start, global_content_end
                                        ):
                                            if j < len(loss_mask):
                                                loss_mask[j] = 1
                                    else:
                                        logger.error(
                                            f"Failed to locate or approximate content tokens for assistant msg {msg_idx}. Content: '{content}', Content Tokens: {content_tokens}, Message Tokens: {message_tokens}"
                                        )
                                        # Decide: raise error or skip masking for this message? Skipping for now.
                                        # raise ValueError(f"Could not locate content tokens for assistant message {msg_idx}")

                    # Update position for next iteration
                    current_pos = len(prefix_tokens)

                # --- Verification ---
                if len(full_sequence_ids) != len(loss_mask):
                    raise ValueError(
                        f"Length mismatch: sequence {len(full_sequence_ids)} vs mask {len(loss_mask)}"
                    )
                if prompt_token_len == -1 and init_prompt_msg_count > 0:
                    # This happens if the prompt itself was empty after tokenization, or if init_prompt_msg_count was 0
                    logger.warning(
                        f"Could not determine prompt token length for sample {i}. Setting to 0."
                    )
                    prompt_token_len = 0
                elif init_prompt_msg_count == 0:
                    prompt_token_len = 0  # No initial prompt messages

                # Convert to tensors
                full_sequence_tensor = torch.tensor(full_sequence_ids, dtype=torch.long)
                loss_mask_tensor = torch.tensor(
                    loss_mask, dtype=torch.bool
                )  # Use bool for mask

                # Create states (0..N-1), actions (1..N), mask (1..N)
                if len(full_sequence_tensor) < 2:
                    logger.warning(
                        f"Sample {i} resulted in sequence length < 2. Skipping."
                    )
                    continue  # Cannot create state/action pairs

                states = full_sequence_tensor[:-1]
                actions = full_sequence_tensor[1:]
                final_loss_mask = loss_mask_tensor[1:]  # Align mask with actions

                completion_len = final_loss_mask.sum().item()

                ep = EpisodeData(
                    states=states,
                    actions=actions,
                    loss_mask=final_loss_mask,
                    terminal_reward=terminal_reward,
                    reward_dict=reward_dict,
                    ground_truth=ground_truth,
                    chat_history=messages,  # Store ChatMessage objects
                    prompt_length=prompt_token_len,
                    completion_length=completion_len,
                )
                results.append(ep)

            except Exception as e:
                logger.error(
                    f"Failed converting sample {i} to EpisodeData: {e}", exc_info=True
                )
                # Optionally skip this sample or re-raise
                continue

        return results

    def _convert_batch_message_to_prompt(
        self, batch_messages: List[List[ChatMessage]]
    ) -> List[str]:
        """Converts a batch messages to chat-style prompt for generation"""
        batch_prompts = []
        for messages in batch_messages:
            messages_as_dicts = [msg.model_dump() for msg in messages]
            # Apply chat template with assistant's generation prompt for generation
            prompt = self.tokenizer.apply_chat_template(
                messages_as_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_prompts.append(prompt)

        return batch_prompts

    @torch.inference_mode()
    def _run_interaction_loop(
        self,
        initial_state: EnvState,
        llm: Any,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        """
        Default interaction loop: Performs a single generation step.

        Suitable for single-step MDPs where the full completion is generated at once.
        Subclasses for multi-step MDPs or tool use should override this method.

        Args:
            initial_state: The starting state from _prepare_initial_state.
            llm: The language model.
            generation_config: Configuration for generation (max_new_tokens, etc.).
            **kwargs: Additional arguments (unused in default).

        Returns:
            EnvState: The final state after generation, with updated batch_messages.
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
            llm: The language model.
            sampling_params: Sampling parameters generation.
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

        # 2. Run the interaction loop (delegated to potentially overridden method)
        try:
            final_state = self._run_interaction_loop(
                initial_state, llm, sampling_params, **kwargs
            )
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error during _run_interaction_loop: {e}",
                exc_info=True,
            )
            # Decide how to handle: return empty, partial, or raise? Returning empty for now.
            return []

        # 3. Convert the final state to training episodes
        try:
            episodes = self._convert_to_episodes(final_state)
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error during _convert_to_episodes: {e}",
                exc_info=True,
            )
            # Decide how to handle: return empty, partial, or raise? Returning empty for now.
            return []

        logger.debug(f"Rank {self.rank}: Rollout generated {len(episodes)} episodes.")
        return episodes
