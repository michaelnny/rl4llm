"""SFT trainer to optimize policy model and bootstrap value head"""

import math
import random
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing_extensions import Self

from rl4llm.core.base_trainer import BaseTrainer
from rl4llm.core.helper import masked_mean
from rl4llm.data import MathAugmenter
from rl4llm.types import Episode, SFTConfig, SFTSample
from rl4llm.utils import assert_file_exist, load_from_jsonl_file


class SFTTrainer(BaseTrainer):
    """Implements code for a SFT leaner for language models."""

    def __init__(self):
        super().__init__()
        self.train_loader: Optional[DataLoader] = None

    @classmethod
    def from_config(cls, config_path: str) -> Self:
        trainer = super().from_config(config_path)
        trainer._setup_sft_specific()
        trainer._load_and_prepare_sft_datasets()
        return trainer

    def _setup_sft_specific(self) -> None:
        """Initialize SFT-specific components."""
        if 'training_config' not in self.config:
            raise ValueError('Config must contain training_config')
        if 'datasets' not in self.config:
            raise ValueError('Config must contain datasets')

        # Create SFT config
        sft_config = self.config['training_config']
        self.train_cfg = SFTConfig(**sft_config)

    def _load_and_prepare_sft_datasets(self) -> None:
        """Load and prepare datasets for training."""
        datasets_config = self.config['datasets']
        prompt_templates = self.config['prompt_templates']

        # Validate inputs
        if not datasets_config or not prompt_templates:
            raise ValueError('Missing datasets_config or prompt_templates')

        # Check file existence
        for item in datasets_config:
            assert_file_exist(item['path'])

        # Create prompt mapping
        state_prompt_map = {item['state_id']: item['user_prompt'] for item in prompt_templates}

        # Process all datasets
        train_episodes: List[Episode] = []

        for item in datasets_config:
            try:
                # Load episodes
                items = load_from_jsonl_file(item['path'])
                episodes = [Episode(**item) for item in items]
                train_episodes.extend(episodes)

            except Exception as e:
                self.logger.error(f"Error processing dataset {item['name']}: {str(e)}")
                continue

        if len(train_episodes) == 0:
            raise RuntimeError('Got no samples for trining')

        num_correct_samples = len([ep for ep in train_episodes if ep.graded_reward >= 1.0])
        correct_rate = num_correct_samples / len(train_episodes)
        self.logger.info(
            f"Processed {len(train_episodes)} episodes, correct vs incorrect: {correct_rate:.2f}:{1 - correct_rate:.2f}"
        )

        augmented_episodes = self._augment_sample_episodes(train_episodes)
        train_episodes.extend(augmented_episodes)

        # Convert episodes to training samples
        train_samples = [self._convert_episode_to_train_sample(ep, state_prompt_map) for ep in train_episodes]

        # Create data loader with DistributedSampler
        train_sampler = DistributedSampler(train_samples, rank=self.local_rank, seed=self.seed, shuffle=True)
        self.train_loader = DataLoader(
            train_samples,
            batch_size=self.batch_size_per_gpu,
            sampler=train_sampler,  # Use DistributedSampler
            pin_memory=True,
            collate_fn=partial(self._train_collate_fn, pad_id=self.pad_token_id),
            num_workers=self.config.get('training', {}).get('num_workers', 0),
            drop_last=True,
        )

    def _augment_sample_episodes(self, episodes: List[Episode]) -> List[Episode]:
        """Augment to generate more incorrect samples."""

        augmented_episodes = []

        math_augmenter = MathAugmenter()
        for ep in episodes:
            if ep.graded_reward < 1.0:
                continue
            try:
                # for episode, generate 2 incorrect samples by randomly inject noise into the 'reasoning' step
                aug_texts = []
                for _ in range(2):
                    copied_ep = deepcopy(ep)
                    first_t = copied_ep.transitions[0]
                    origin_text = first_t.action.text
                    augmented_text, _ = math_augmenter.augment_text(origin_text, max_replacements=random.randint(3, 8))
                    if augmented_text != origin_text and augmented_text not in aug_texts:
                        first_t.action.text = augmented_text
                        first_t.reward = 0.0
                        copied_ep.transitions = [first_t]  # only use the 'reasoning' step
                        copied_ep.graded_reward = 0.0
                        augmented_episodes.append(copied_ep)
                        aug_texts.append(augmented_text)

                # augmented_episodes.append(copied_ep)
            except Exception as e:
                self.logger.warning(f"Failed to process episode: {str(e)}")
                continue

        return augmented_episodes

    def _convert_episode_to_train_sample(self, episode: Episode, state_prompt_map: Dict[str, str]) -> SFTSample:
        """Convert a single episode to a training sample."""
        episode_token_ids = []
        episode_rewards = []
        episode_loss_masks = []

        # Process each transition in the episode
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

            episode_token_ids.append(user_token_ids)
            episode_rewards.append(np.zeros_like(user_token_ids, dtype=float))
            episode_loss_masks.append(np.zeros_like(user_token_ids))

            assistant_token_ids = self._text_to_token_ids(t.action.text)
            assistant_token_ids = self._handle_special_tokens(assistant_token_ids, is_intermediate=t.is_done)
            assistant_rewards = np.zeros_like(assistant_token_ids, dtype=float)
            assistant_rewards[-1] = t.reward

            assistant_masks = np.ones_like(assistant_token_ids)
            episode_token_ids.append(assistant_token_ids)
            episode_rewards.append(assistant_rewards)
            episode_loss_masks.append(assistant_masks)

        # Concatenate and validate sequences
        token_ids = np.concatenate(episode_token_ids)
        rewards = np.concatenate(episode_rewards)
        loss_masks = np.concatenate(episode_loss_masks)
        correctness = np.array([episode.graded_reward >= 1.0])

        # Compute returns and create training sample
        mc_returns = self._compute_masked_mc_returns(rewards, loss_masks)

        return SFTSample(
            input_tokens=torch.from_numpy(token_ids[:-1]).to(dtype=torch.long),
            target_tokens=torch.from_numpy(token_ids[1:]).to(dtype=torch.long),
            mc_returns=torch.from_numpy(mc_returns[1:]).to(dtype=torch.float),
            loss_masks=torch.from_numpy(loss_masks[1:]).to(dtype=torch.bool),
            correctness=torch.from_numpy(correctness).to(dtype=torch.bool),
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

    def _compute_masked_mc_returns(self, rewards: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Computes monte carlo returns considering only assistant turns using NumPy arrays.

        Args:
            rewards (np.ndarray): Float array with rewards (0 for user), shape [seq_len]
            masks (np.ndarray): Binary mask (0 for user, 1 for assistant), shape [seq_len]

        Returns:
            np.ndarray: Array of the original shape, with discounted returns
                    for assistant turns and zeros for user turns
        """
        # Input validation
        assert rewards.ndim == masks.ndim == 1, 'Inputs must be 1-dimensional'
        assert len(rewards) == len(masks), 'Rewards and masks must have same length'

        # Initialize returns array
        returns = np.zeros_like(masks, dtype=np.float32)

        # Get assistant rewards using boolean indexing
        assistant_rewards = rewards[masks.astype(bool)]
        seq_len = len(assistant_rewards)
        # Handle empty case
        if seq_len == 0:
            return returns

        gamma = self.get_dynamic_discount(seq_len)

        # Initialize assistant returns
        assistant_returns = np.zeros_like(assistant_rewards)

        R = 0
        for t in reversed(range(len(assistant_rewards))):
            R = assistant_rewards[t] + gamma * R
            assistant_returns[t] = R

        # Place assistant returns back in the original array
        returns[masks.astype(bool)] = assistant_returns

        return returns

    def train(self) -> None:
        """Doing supervised fine-tuning over the collected samples"""
        self.policy_engine.train()
        self._clear_gpu_memory()

        # Calculate total number of steps
        total_samples = len(self.train_loader.dataset)
        total_steps = math.ceil(total_samples / self.batch_size) * self.train_cfg.num_epochs

        pbar = tqdm(desc='Training steps', unit='batch', total=total_steps)
        # Initialize accumulated batch stats
        steps_count = 0
        accumulated_batch_stats = defaultdict(list)
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.train_cfg.num_epochs):
                for mini_batch in self.train_loader:
                    metrics = self._process_minibatch(mini_batch)
                    del mini_batch
                    # Accumulate batch stats
                    for name, values in metrics.items():
                        accumulated_batch_stats[name].extend(values)
                    # Logging stats
                    if self.policy_engine.is_gradient_accumulation_boundary():
                        self.update_count += 1
                        steps_count += 1
                        pbar.update(1)
                        # Compute aggregated batch stats
                        batch_stats = self._aggregate_stats(accumulated_batch_stats)
                        elapsed_time = pbar.format_dict.get('elapsed', 0)
                        batch_stats['step_time'] = round(elapsed_time / max(steps_count, 1), 4)
                        self._log_batch_stats(batch_stats)
                        accumulated_batch_stats.clear()
                        self.save_checkpoint()

        self.save_checkpoint(is_final=True)

        elapsed_time = pbar.format_dict.get('elapsed', 0)
        pbar.close()

        iter_stats = {
            'elapsed/time': round(elapsed_time, 4),
            'elapsed/step_time': round(elapsed_time / max(steps_count, 1), 4),
            'elapsed/updates': self.update_count,
        }

        self.logger.info(iter_stats)

    # --- Private Helper Methods ---

    def _process_minibatch(
        self,
        mini_batch: SFTSample,
    ):
        """Process one mini-batch and compute the loss and metrics"""
        input_tokens, attn_mask = self._prepare_model_inputs(mini_batch.input_tokens)
        outputs = self.policy_engine.forward(
            tokens=input_tokens,
            attn_mask=attn_mask,
            use_cache=False,
            return_values=True,
        )
        del input_tokens, attn_mask
        loss, metrics = self._compute_loss(
            pred_pi_logits=outputs.logits,
            pred_values=outputs.values,
            batch=mini_batch,
        )

        self.policy_engine.backward(loss)
        self.policy_engine.step()

        del outputs
        self._clear_gpu_memory()

        return metrics

    def _compute_loss(
        self,
        pred_pi_logits: torch.Tensor,
        pred_values: torch.Tensor,
        batch: SFTSample,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute PPO loss and other metrics.

        Args:
            pred_pi_logits (torch.Tensor): Predicted policy logits.
            pred_values (torch.Tensor): Predicted value estimates.
            batch (SFTSample): object containing batch data.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Tuple containing loss and stats in tensor forms.
        """
        device = pred_pi_logits.device

        target_tokens = batch.target_tokens.to(device)
        mc_returns = batch.mc_returns.to(device)
        loss_masks = batch.loss_masks.bool().to(device)
        correctness_masks = batch.correctness.bool().to(device)

        assert pred_pi_logits.dim() == 3  # [B, max_seq_len, vocab_size]
        assert target_tokens.dim() == loss_masks.dim() == 2  # [B, max_seq_len]
        assert pred_pi_logits.shape[0] == target_tokens.shape[0] == loss_masks.shape[0]

        B, T, *_ = pred_pi_logits.shape
        lm_losses = F.cross_entropy(pred_pi_logits.view(-1, pred_pi_logits.size(-1)), target_tokens.view(-1), reduction='none')
        assert not torch.any(torch.isnan(lm_losses))
        lm_losses = lm_losses.view(B, T)
        assert lm_losses.shape == loss_masks.shape

        # Value head loss
        value_losses = F.mse_loss(pred_values.to(self.compute_dtype), mc_returns.to(self.compute_dtype), reduction='none')

        # only using correct samples to compute LM loss, and skip the augmented tokens if any
        lm_losses = correctness_masks * masked_mean(lm_losses, loss_masks, dim=1)  # [batch_size]
        # using both incorrect and correct samples for value loss, we also include the augmented tokens
        value_losses = masked_mean(value_losses, loss_masks, dim=1)  # [batch_size]
        total_losses = lm_losses + self.train_cfg.value_loss_coef * value_losses
        loss = total_losses.mean()

        stats = {
            'loss/total': total_losses.detach().to(device=self.device, dtype=self.compute_dtype),
            'loss/lm': lm_losses.detach().to(device=self.device, dtype=self.compute_dtype),
            'loss/value': value_losses.detach().to(device=self.device, dtype=self.compute_dtype),
        }

        return loss, stats

    @staticmethod
    def _train_collate_fn(batch: List[SFTSample], pad_id: int) -> SFTSample:
        """
        Custom collate function to pad sequences.
        """

        batch_size = len(batch)
        max_seq_len = max([len(item.input_tokens) for item in batch])

        batch_input_tokens = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        batch_target_tokens = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        batch_mc_returns = torch.full((batch_size, max_seq_len), 0.0, dtype=torch.float)
        batch_loss_masks = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)
        batch_correctness = torch.full((batch_size,), 0, dtype=torch.bool)

        for i, item in enumerate(batch):
            seq_len = len(item.input_tokens)
            batch_input_tokens[i, :seq_len] = item.input_tokens
            batch_target_tokens[i, :seq_len] = item.target_tokens
            batch_mc_returns[i, :seq_len] = item.mc_returns
            batch_loss_masks[i, :seq_len] = item.loss_masks
            batch_correctness[i] = item.correctness

        return SFTSample(
            input_tokens=batch_input_tokens,
            target_tokens=batch_target_tokens,
            mc_returns=batch_mc_returns,
            loss_masks=batch_loss_masks,
            correctness=batch_correctness,
        )
