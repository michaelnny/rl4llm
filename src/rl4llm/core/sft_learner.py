"""PPO leaner class for model training using deepspeed engine."""

import logging
import math
import os
import random
import multiprocessing as mp
from copy import deepcopy
from collections import defaultdict
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.utils.data import DataLoader
from tqdm import tqdm

from rl4llm.core.base_ds_class import BaseDeepSpeedClass
from rl4llm.core.episode_processor import EpisodeProcessor
from rl4llm.core.helper import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    masked_mean,
    masked_normalize,
    masked_sum,
)
from rl4llm.data import MathAugmenter
from rl4llm.types import Episode, SFTConfig, SFTSample, ProcessedEpisode
from rl4llm.utils import TrainingTracker, assert_file_exist, load_from_jsonl_file


class SFTLearner(BaseDeepSpeedClass):
    """Implements the SFT for training the large language model using deepspeed."""

    def __init__(
        self,
        config: Dict[str, Any],
        local_rank: int,
        tracker: Optional[TrainingTracker] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, local_rank, tracker=tracker, logger=logger)
        self.policy_engine: deepspeed.DeepSpeedEngine = self._init_policy_engine()
        self.sample_processor: EpisodeProcessor = EpisodeProcessor(tokenizer=self.tokenizer)
        self.batch_size_per_gpu: int = self.config['deepspeed']['train_micro_batch_size_per_gpu']
        self.batch_size: int = self._calculate_batch_size()
        self.ckpt_dir: str = self.tracker.output_paths['checkpoints'] if self.tracker else "/tmp"
        self.train_cfg: SFTConfig = SFTConfig(**self.config['training_config'])
        self.update_count = 0
        self.iteration_count = 0
        self.episode_count = 0

    def get_policy_grad_norm(self) -> float:
        """Compute the norm of the policy model's gradients."""
        return self._get_grad_norm(self.policy_engine)

    def save_policy_model(
        self,
        tag: Optional[str] = None,
    ) -> None:
        """Save the policy model to the output directory."""

        self._save_hf_model(
            self.policy_engine,
            save_base_dir=self.ckpt_dir,
            step_count=self.update_count,
            tag=tag,
            keep_last_n=self.train_cfg.checkpoint_keep_n,
        )

    def on_exit(self):
        """Cleanup on exit."""
        if self.tracker is not None:
            self.tracker.flush()
            self.tracker.close()

        if self.train_cfg.checkpoint_enabled:
            self.save_policy_model(tag="final")

    def train(self) -> None:
        """Run SFT training epochs."""
        # episodes = self.sample_processor.process_episodes(episodes)

        train_loader = self._load_and_prepare_sft_datasets()

        self.policy_engine.train()
        self.policy_engine = self.policy_engine.to(self.device)  # Move engines to device
        torch.cuda.empty_cache()

        total_samples = len(train_loader.dataset)
        total_steps = math.ceil(total_samples / self.batch_size) * self.train_cfg.num_epochs

        pbar = tqdm(desc='Training steps', unit='batch', total=total_steps)

        # Initialize accumulated batch stats
        accumulated_batch_stats = defaultdict(list)
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.train_cfg.num_epochs):
                for mini_batch in train_loader:
                    metrics = self._process_minibatch(mini_batch)
                    del mini_batch
                    # Accumulate batch stats
                    for name, values in metrics.items():
                        accumulated_batch_stats[name].extend(values)
                    # Logging stats
                    if self.policy_engine.is_gradient_accumulation_boundary():
                        self.update_count += 1
                        pbar.update(1)
                        # Compute aggregated batch stats
                        batch_stats = self._aggregate_stats(accumulated_batch_stats)
                        elapsed_time = pbar.format_dict.get('elapsed', 0)
                        batch_stats['step_time'] = round(elapsed_time / max(self.update_count, 1), 4)
                        self._log_batch_stats(batch_stats)
                        accumulated_batch_stats.clear()
                        if self.train_cfg.checkpoint_enabled and self.update_count % self.train_cfg.checkpoint_interval == 0:
                            self.save_policy_model()

                self.iteration_count += 1
                if self.train_cfg.checkpoint_enabled:
                    self.save_policy_model(tag=f'epoch_{epoch}')

        elapsed_time = pbar.format_dict.get('elapsed', 0)
        pbar.close()

        iter_stats = {
            'elapsed/time': round(elapsed_time, 4),
            'elapsed/step_time': round(elapsed_time / max(self.update_count, 1), 4),
            'elapsed/updates': self.update_count,
        }

        self.logger.info(iter_stats)

    def _get_lr_by_group_name(self, name: str) -> float:
        """Get learning rate for a parameter group by name."""
        for group in self.policy_engine.optimizer.param_groups:
            if group['name'] == name:
                return group['lr']
        return self.policy_engine.optimizer.param_groups[0]['lr']

    def _get_common_stats(self) -> Dict:
        """Get common statistics like learning rates."""
        return {
            'policy/learning_rate': self._get_lr_by_group_name('policy'),
            'value/learning_rate': self._get_lr_by_group_name('value'),
        }

    def _process_minibatch(self, mini_batch: SFTSample) -> Dict[str, np.array]:
        """Process a mini-batch and compute loss and metrics."""
        input_tokens, attn_mask = self._prepare_model_inputs(mini_batch.input_tokens)
        outputs = self.policy_engine.forward(
            input_ids=input_tokens,
            attention_mask=attn_mask,
            return_dict=True,
            use_cache=False,
            return_values=True,  # compute values
        )
        loss, metrics = self._compute_loss(pred_pi_logits=outputs.logits, pred_values=outputs.values, batch=mini_batch)
        self.policy_engine.backward(loss)
        self.policy_engine.step()
        return metrics

    def _compute_loss(
        self, pred_pi_logits: torch.Tensor, pred_values: torch.Tensor, batch: SFTSample
    ) -> Tuple[torch.Tensor, Dict[str, np.array]]:
        """Compute language modeling loss, value loss, and other metrics.

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
        assert pred_values.dim() == 2  # [B, max_seq_len]
        assert target_tokens.dim() == loss_masks.dim() == 2  # [B, max_seq_len]
        assert pred_pi_logits.shape[0] == target_tokens.shape[0] == loss_masks.shape[0]
        assert mc_returns.shape == pred_values.shape

        B, T, *_ = pred_pi_logits.shape
        lm_losses = F.cross_entropy(pred_pi_logits.view(-1, pred_pi_logits.size(-1)), target_tokens.view(-1), reduction='none')
        lm_losses = lm_losses.view(B, T)
        assert lm_losses.shape == loss_masks.shape

        # Value head loss
        value_losses = F.mse_loss(pred_values.to(self.dtype), mc_returns.to(self.dtype), reduction="none")

        # only using correct samples to compute LM loss, and skip the augmented episode
        lm_losses = masked_mean(lm_losses, loss_masks, dim=1)[correctness_masks]  # [correct_batch_size]
        value_losses = masked_mean(value_losses, loss_masks, dim=1)  # [batch_size]

        loss = lm_losses.mean() + self.train_cfg.value_loss_coef * value_losses.mean()

        stats = {
            'loss/lm': lm_losses.detach().to(device=self.device, dtype=self.dtype),
            'loss/value': value_losses.detach().to(device=self.device, dtype=self.dtype),
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

    def _calculate_batch_size(self) -> int:
        """Calculates the effective batch size."""
        ds_config = self.config['deepspeed']
        if 'train_batch_size' in ds_config:
            return ds_config['train_batch_size']
        if 'gradient_accumulation_steps' not in ds_config:
            return self.batch_size_per_gpu

        return self.batch_size_per_gpu * ds_config['gradient_accumulation_steps']

    def _init_policy_engine(self) -> deepspeed.DeepSpeedEngine:
        """Initializes the DeepSpeed policy training engine."""
        model = self._load_policy_model()
        param_groups = self._get_params_groups(model, self.config['deepspeed']['optimizer'])
        return self._create_deepspeed_training_engine(model, param_groups)

    def _load_and_prepare_sft_datasets(self) -> DataLoader:
        """Load and prepare datasets for SFT training."""
        self.logger.info("Loading and preparing datasets for training...")
        datasets_config = self.config['datasets']
        prompt_templates = self.config['prompt_templates']

        max_seq_len = min(self.config['model'].get('max_seq_len', 4096), self.tokenizer.model_max_length)

        # Validate inputs
        if not datasets_config or not prompt_templates:
            raise ValueError('Missing datasets_config or prompt_templates')

        # Check file existence
        for item in datasets_config:
            assert_file_exist(item['path'])

        # Create prompt mapping
        user_prompt_map = {}
        system_prompt_map = {}

        for item in prompt_templates:
            state_id = item['state_id'].lower().strip()
            if 'user_prompt' in item and item['user_prompt']:
                user_prompt_map[state_id] = item['user_prompt']
            if 'system_prompt' in item and item['system_prompt']:
                system_prompt_map[state_id] = item['system_prompt']
            else:
                system_prompt_map[state_id] = None

        # Process all datasets
        train_episodes: List[Episode] = []

        num_correct_samples = 0

        for item in datasets_config:
            try:
                # Load episodes
                items = load_from_jsonl_file(item['path'])
                reward_scale = item.get('reward_scale', 1.0)
                episodes = [Episode(**item) for item in items]

                for ep in episodes:
                    if len(ep.transitions) > 1:
                        # only use the first transition
                        ep.transitions = ep.transitions[:1]

                    assert ep.transitions[0].state.state_id == 'reasoning'

                    for i, t in enumerate(ep.transitions):
                        state_id = t.state.state_id
                        # Replace with a simpler prompt
                        if state_id in user_prompt_map:
                            t.state.user_prompt = user_prompt_map[state_id]
                        if state_id in system_prompt_map:
                            t.state.system_prompt = system_prompt_map[state_id]

                num_correct_samples += len([ep for ep in episodes if ep.graded_reward >= 1.0])

                # Optionally, scale the rewards (e.g. GPT4 samples are not very good reasoning samples)
                if reward_scale > 0 and reward_scale < 1.0:
                    for ep in episodes:
                        if ep.graded_reward == 1.0:
                            ep.graded_reward *= reward_scale
                            for t in ep.transitions:
                                t.reward *= reward_scale

                train_episodes.extend(episodes)

            except Exception as e:
                self.logger.error(f"Error processing dataset {item['name']}: {str(e)}")
                continue

        if len(train_episodes) == 0:
            raise RuntimeError('Got no samples for trining')

        correct_rate = num_correct_samples / len(train_episodes)
        self.logger.info(
            f"Loaded {len(train_episodes)} episodes, correct vs incorrect: {correct_rate:.2f}:{1 - correct_rate:.2f}"
        )

        if self.train_cfg.augment_rate > 0:
            # Generate more synthetic samples
            math_augmenter = MathAugmenter()
            correct_episodes = [ep for ep in train_episodes if ep.graded_reward > 0.0]
            sampled_episodes = random.choices(correct_episodes, k=int(self.train_cfg.augment_rate * len(correct_episodes)))
            if len(sampled_episodes) > 0:
                self.logger.info(f"Augmenting {len(sampled_episodes)} samples")
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    # Parallel Augmentation
                    augmented_episodes = pool.map(
                        partial(self._augment_single_episode, math_augmenter=math_augmenter),
                        sampled_episodes,
                    )
                    self.logger.info(f"Generated {len(augmented_episodes)} augmented samples")
                    train_episodes.extend(augmented_episodes)

        # Turn episodes into SFT training samples
        processed_episodes = self.sample_processor.process_episodes(train_episodes)
        train_samples = []
        for ep in processed_episodes:
            token_ids = ep.token_ids
            rewards = ep.rewards
            loss_masks = ep.loss_masks
            is_correct = rewards[-1] > 0.0

            # Compute returns and create training sample
            mc_returns = self._compute_masked_mc_returns(rewards, loss_masks)

            if max_seq_len > 0 and len(token_ids) > max_seq_len:
                self.logger.warning(f"Truncating sequence of length {len(token_ids)} to max_seq_len={max_seq_len}")
                token_ids = token_ids[:max_seq_len]
                mc_returns = mc_returns[:max_seq_len]
                loss_masks = loss_masks[:max_seq_len]

            assert sum(loss_masks) > 0, "No assistant turns found in the episode"

            train_samples.append(
                SFTSample(
                    input_tokens=torch.from_numpy(token_ids[:-1]).to(dtype=torch.long),
                    target_tokens=torch.from_numpy(token_ids[1:]).to(dtype=torch.long),
                    mc_returns=torch.from_numpy(mc_returns[1:]).to(dtype=self.dtype),
                    loss_masks=torch.from_numpy(loss_masks[1:]).to(dtype=torch.bool),
                    correctness=torch.tensor([is_correct]).to(dtype=torch.bool),
                )
            )
        
        assert len(train_samples) > 0, "No training samples found"

        return DataLoader(
            train_samples,
            batch_size=self.batch_size_per_gpu,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=partial(self._train_collate_fn, pad_id=self.pad_token_id),
            drop_last=True,
        )

    @staticmethod
    def _augment_single_episode(episode: Episode, math_augmenter: MathAugmenter) -> Episode:
        """Helper function to augment a single episode by randomly replace numbers in the correct answer."""

        try:
            copied_ep = deepcopy(episode)
            first_t = copied_ep.transitions[0]
            origin_text = first_t.action.text
            augmented_text, _ = math_augmenter.augment_text(origin_text, max_replacements=random.randint(5, 10))
            if augmented_text != origin_text:
                first_t.action.text = augmented_text
                first_t.reward = 0.0
                copied_ep.transitions = [first_t]
                copied_ep.graded_reward = 0.0
                return copied_ep

        except Exception as e:
            print(f"Failed to process episode for augmentation: {str(e)}")
            return None

    def _compute_masked_mc_returns(self, rewards: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Computes monte carlo returns considering only assistant turns.

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

        gamma = self.train_cfg.gamma

        # Initialize assistant returns
        assistant_returns = np.zeros_like(assistant_rewards)

        R = 0
        for t in reversed(range(len(assistant_rewards))):
            R = assistant_rewards[t] + gamma * R
            assistant_returns[t] = R

        # Place assistant returns back in the original array
        returns[masks.astype(bool)] = assistant_returns

        return returns

    def _log_batch_stats(self, batch_stats: Dict[str, Any]):
        """Log batch statistics."""
        batch_stats.update(self._get_common_stats())
        if self.tracker:
            Thread(target=self.tracker.log_learner_step_stats, args=(batch_stats,)).start()

    def _log_iteration_stats(self, iter_stats: Dict[str, Any]):
        """Log iteration statistics."""
        iter_stats.update(self._get_common_stats())
        self.logger.info(f"Learner stats: {iter_stats}")
        if self.tracker:
            self.tracker.log_learner_iteration_stats(iter_stats)
