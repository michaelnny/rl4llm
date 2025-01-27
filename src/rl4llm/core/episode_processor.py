"""Handles post-actor samples processing before feed to RL """

import logging
import random
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np

from transformers import PreTrainedTokenizer
from rl4llm.types import Episode, ProcessedEpisode

logger = logging.getLogger(__name__)


# Global worker to be initialized in each pool worker
_global_worker = None


def initializer(tokenizer):
    """Initializer for each pool worker."""
    global _global_worker
    _global_worker = WorkerProcessor(tokenizer)


def process_single_episode_wrapper(episode) -> Tuple[ProcessedEpisode, str]:
    """Wrapper function that uses the global worker."""
    try:
        return (
            _global_worker.process_single_episode(episode),
            None,
        )
    except Exception as e:
        return episode, str(e)  # Return error message if something fails


class WorkerProcessor:
    """Worker class for processing individual episodes."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

    def process_single_episode(self, episode: Episode) -> ProcessedEpisode:
        """Process a single episode with token and reward handling."""

        assert episode.transitions

        # Tokenize each turn, but we need the loss mask for each turn, where 0s for user's turn, and 1s for assistant's turn
        all_token_ids = []
        all_rewards = []
        all_temperatures = []
        all_masks = []

        for i, t in enumerate(episode.transitions):
            is_first_turn = i == 0

            # Add user's turn
            user_prompt = t.state.user_prompt
            user_content = user_prompt
            if is_first_turn:
                # add original question to first user turn
                try:
                    user_content = user_prompt.format(question=episode.question)
                except Exception:
                    user_content = f"{user_prompt}\n\nQuestion:\n{episode.question}"

            user_token_ids = self.tokenizer.apply_chat_template(
                (
                    [{'role': 'system', 'content': t.state.system_prompt}, {'role': 'user', 'content': user_content}]
                    if is_first_turn and t.state.system_prompt
                    else [{'role': 'user', 'content': user_content}]
                ),
                tokenize=True,
                add_generation_prompt=True,
            )
            all_token_ids.append(user_token_ids)
            all_rewards.append(np.zeros_like(user_token_ids, dtype=float))
            all_temperatures.append(np.ones_like(user_token_ids, dtype=float))
            all_masks.append(np.zeros_like(user_token_ids))

            # Add assistant's turn
            assistant_token_ids = self._text_to_token_ids(t.action.text)
            assistant_token_ids = self._handle_special_tokens(assistant_token_ids, is_intermediate=t.is_done)
            assistant_rewards = np.zeros_like(assistant_token_ids, dtype=float)
            assistant_rewards[-1] = t.reward
            assistant_masks = np.ones_like(assistant_token_ids)
            assistant_temperatures = np.ones_like(assistant_token_ids, dtype=float) * t.action.temperature

            if t.action.exploring_steps is not None and t.action.exploring_steps > 0:
                assistant_masks[: t.action.exploring_steps] = 0

            all_token_ids.append(assistant_token_ids)
            all_rewards.append(assistant_rewards)
            all_temperatures.append(assistant_temperatures)
            all_masks.append(assistant_masks)

        # Build final sequences
        token_ids = np.concatenate(all_token_ids, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        temperatures = np.concatenate(all_temperatures, axis=0)
        loss_masks = np.concatenate(all_masks, axis=0)

        # Validate shapes
        shapes = {
            'token_ids': token_ids.shape,
            'temperatures': temperatures.shape,
            'rewards': rewards.shape,
            'masks': loss_masks.shape,
        }
        if not all(shape == token_ids.shape for shape in shapes.values()):
            raise ValueError(f"Inconsistent shapes in sequence components: {shapes}")

        full_chat_text = self.tokenizer.decode(token_ids)
        logger.debug(f"\n\nEpisode sample:\n{full_chat_text}")

        return ProcessedEpisode(
            token_ids=token_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            temperatures=temperatures,
        )

    def _text_to_token_ids(self, text: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(text, truncation=True, padding=False, add_special_tokens=False))

    def _handle_special_tokens(self, token_ids: np.ndarray, is_intermediate: bool) -> np.ndarray:
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Remove all BOS, EOS, PAD tokens
        token_ids = token_ids[(token_ids != bos_id) & (token_ids != eos_id) & (token_ids != pad_id)]

        if not is_intermediate:
            # Append a single EOS token to final sequences
            token_ids = np.concatenate((token_ids, np.array([eos_id])))

        return token_ids


class EpisodeProcessor:
    """Handles episode augmentation and parallel processing to turn text into tokens."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize the processor.

        Args:
            tokenizer: Tokenizer for text processing
            num_workers: Number of worker processes to use
        """

        assert tokenizer is not None, 'Tokenizer must be provided'

        self.tokenizer = tokenizer

        self.num_workers = num_workers or cpu_count()
        self.pool = None  # Pool will be initialized in process_episodes

    def close_pool(self):
        """Close the multiprocessing pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def process_episodes(
        self,
        episodes: List[Episode],
    ) -> List[ProcessedEpisode]:
        """
        Process all episodes in parallel.
        """
        if not episodes:
            logger.warning('No episodes provided')
            return []

        logger.info(f"Processing total of {len(episodes)} episodes")

        if self.pool is None:
            # Initialize the pool once if not already initialized
            self.pool = Pool(
                processes=self.num_workers,
                initializer=initializer,
                initargs=(self.tokenizer,),
            )

        # Use imap_unordered for better performance and memory usage
        processed_episodes = []
        failed_episodes = []

        for result, error in self.pool.imap_unordered(process_single_episode_wrapper, episodes):
            if error is None:
                processed_episodes.append(result)
            else:
                failed_episodes.append((result, error))  # Collect failed episodes

        # Handle failed episodes (logging, retrying, etc.)
        if failed_episodes:
            self._log_failed_episodes(failed_episodes)

        return processed_episodes

    def _log_failed_episodes(self, failed_episodes: List[tuple[Episode, str]]) -> None:
        """Log information about failed episode processing."""
        logger.error(f"Failed to process {len(failed_episodes)} episodes:")
        for episode, error in failed_episodes:
            logger.error(f"Episode {episode.question[:50]}...: {error}")
