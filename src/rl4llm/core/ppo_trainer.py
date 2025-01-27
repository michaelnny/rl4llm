"""RL PPO trainer to optimize policy model"""

import logging
import math
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel
from typing_extensions import Self

from rl4llm.core.base_trainer import BaseTrainer
from rl4llm.core.episode_processor import EpisodeProcessor
from rl4llm.core.helper import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    masked_mean,
    masked_normalize,
    masked_sum,
)
from rl4llm.types import Episode, PPOConfig, PPOSample, ProcessedEpisode

logger = logging.getLogger()


class PPOTrainer(BaseTrainer):
    """Implements the Proximal Policy Optimization (PPO) algorithm for language models."""

    def __init__(self):
        super().__init__()

        self.ref_policy_model: Optional[PreTrainedModel] = None
        self.ref_policy_engine: Optional[DeepSpeedEngine] = None
        self.sample_processor: Optional[EpisodeProcessor] = None

        self.iter_count = 0
        self.episode_count = 0

    # --- Public Methods ---

    @classmethod
    def from_config(cls, config_path: str) -> Self:
        trainer = super().from_config(config_path)
        trainer._setup_ppo_specific()
        trainer._setup_reference_model()
        return trainer

    def _setup_ppo_specific(self) -> None:
        """Initialize PPO-specific components."""
        if 'training_config' not in self.config:
            raise ValueError('Config must contain training_config')
        if 'datasets' not in self.config:
            raise ValueError('Config must contain datasets')

        # Create PPO config
        ppo_config = self.config['training_config']
        self.train_cfg = PPOConfig(**ppo_config)

        self.sample_processor = EpisodeProcessor(tokenizer=self.tokenizer)

    def _setup_reference_model(self):
        """Setup reference model for PPO training."""
        ref_model = AutoModelForCausalLM.from_pretrained(**self.model_kwargs)
        self.disable_dropout(ref_model)
        self.freeze_model(ref_model)
        self.ref_policy_model = ref_model

        eval_ds_config = {
            "stage": self.config['deepspeed']['zero_optimization'].get('stage', 2),
            "stage3_param_persistence_threshold": "auto",
            "offload_param": {
                "device": "none",
                "pin_memory": True,
            },
            "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
        }

        self.ref_policy_engine, _, _, _ = deepspeed.initialize(
            model=self.ref_policy_model,
            model_parameters=None,
            config=eval_ds_config,
        )

        self.ref_policy_model = self.ref_policy_engine.module
        self._offload_reference_model()

    def get_episode_count(self) -> int:
        return self.episode_count

    def train(self, episodes: List[Episode]) -> None:
        """Run PPO updates M epochs using the collected samples"""

        episodes = self.sample_processor.process_episodes(episodes)
        transitions = self._prepare_ppo_transitions(episodes)

        self.policy_engine.train()
        self.policy_engine = self.policy_engine.to(self.device)
        self._clear_gpu_memory()

        data_loader = self._create_data_loader(transitions, batch_size=self.batch_size_per_gpu, shuffle=True, for_prepare=False)
        total_steps = math.ceil(self.train_cfg.num_epochs * len(episodes) / self.batch_size)
        pbar = tqdm(desc='Training steps', unit='batch', total=total_steps)

        steps_count = 0
        # Initialize iteration-level stat tracking
        accumulated_iter_stats = defaultdict(list)

        # Initialize accumulated batch stats
        accumulated_batch_stats = defaultdict(list)
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.train_cfg.num_epochs):
                for mini_batch in data_loader:
                    # forward compute loss and also call loss.backward()
                    metrics = self._process_minibatch(mini_batch)

                    # Accumulate batch stats
                    for name, values in metrics.items():
                        accumulated_batch_stats[name].extend(values)
                        accumulated_iter_stats[name].extend(values)

                    del mini_batch

                    # Logging stats
                    if self.policy_engine.is_gradient_accumulation_boundary():
                        self.update_count += 1
                        steps_count += 1
                        pbar.update(1)

                        # Compute aggregated batch stats
                        batch_stats = self._aggregate_stats(accumulated_batch_stats, True)
                        elapsed_time = pbar.format_dict.get('elapsed', 0)
                        batch_stats['step_time'] = round(elapsed_time / max(steps_count, 1), 4)
                        self._log_batch_stats(batch_stats)
                        accumulated_batch_stats.clear()
                        self.save_checkpoint()

        elapsed_time = pbar.format_dict.get('elapsed', 0)
        pbar.close()
        self.iteration_count += 1
        self.episode_count += len(episodes)

        # Compute and log iteration-level stats
        iter_stats = self._aggregate_stats(accumulated_iter_stats, True)
        iter_stats.update(
            {
                'elapsed/time': round(elapsed_time, 4),
                'elapsed/step_time': round(elapsed_time / max(steps_count, 1), 4),
                'elapsed/updates': self.update_count,
                'elapsed/episodes': self.episode_count,
            }
        )

        self._log_iteration_stats(iter_stats)

    # --- Private Helper Methods ---
    def _create_data_loader(
        self, samples: List[Any], batch_size: int, shuffle: bool = True, for_prepare: bool = False
    ) -> DataLoader:
        assert batch_size >= 1
        if for_prepare:
            collate_fn = partial(self._prep_collate_fn, pad_id=self.pad_token_id, dtype=self.compute_dtype)
        else:
            collate_fn = partial(self._train_collate_fn, pad_id=self.pad_token_id, dtype=self.compute_dtype)
        return DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.device.type == 'cuda',
            collate_fn=collate_fn,
        )

    def _process_minibatch(self, mini_batch: PPOSample):
        """Process one mini-batch and compute the loss and metrics"""
        states_tm1, attn_mask = self._prepare_model_inputs(mini_batch.states)
        outputs = self.policy_engine.forward(
            input_ids=states_tm1,
            attention_mask=attn_mask,
            return_dict=True,
            use_cache=False,
        )
        del states_tm1, attn_mask
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
        batch: PPOSample,
    ) -> Tuple[torch.Tensor, Dict[str, np.array]]:
        """Compute PPO loss and other metrics.

        Args:
            pred_pi_logits (torch.Tensor): Predicted policy logits.
            pred_values (torch.Tensor): Predicted value estimates.
            batch (PPOSample): PPOSample object containing batch data.

        Returns:
            Tuple[torch.Tensor, Dict[str, np.array]]: Tuple containing loss tensor and stats dictionary.
        """
        config = self.train_cfg
        device = pred_pi_logits.device

        # Move tensors to device
        actions = batch.actions.to(device)
        behavior_logprobs = batch.pi_logprobs.to(device)
        ref_logprobs = batch.ref_logprobs.to(device)
        old_values = batch.values.to(device)
        rewards = batch.rewards.to(device)
        loss_masks = batch.loss_masks.bool().to(device)
        loss_masks = loss_masks.detach()

        rewards *= loss_masks
        old_values *= loss_masks

        # Compute log probabilities and KL divergence
        entropies = compute_entropy_from_logits(pred_pi_logits)
        pi_logprobs = compute_logprobs_from_logits(pred_pi_logits, actions)

        # Compute token-level KL estimate for reward shaping
        kl = pi_logprobs - ref_logprobs
        kl *= loss_masks

        # Remove tiny KL noise
        # kl = torch.where((rewards != 0) | (torch.abs(kl) < 1e-5), torch.zeros_like(kl), kl)
        kl_score = -config.kl_coef * kl
        mixed_rewards = (rewards + kl_score) * loss_masks

        if self.train_cfg.normalize_rewards:
            mixed_rewards = masked_normalize(mixed_rewards, loss_masks)

        # Compute returns and advantages
        returns, advantages = self._compute_returns_and_advantages(rewards=mixed_rewards, values=old_values, masks=loss_masks)

        if self.train_cfg.normalize_advantages:
            advantages = masked_normalize(advantages, loss_masks)

        # PPO Clipped Policy Loss
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clip_adv = torch.clamp(ratio, 1.0 - config.policy_clip_eps, 1.0 + config.policy_clip_eps) * advantages.detach()
        pg_losses = torch.min(clip_adv * advantages.detach(), clip_adv)
        clipped = ratio.gt(1 + config.policy_clip_eps) | ratio.lt(1 - config.policy_clip_eps)
        pg_clipfrac = torch.as_tensor(clipped, dtype=ratio.dtype)
        approxkl = (pi_logprobs - behavior_logprobs).detach()

        # Value Function Loss
        vpred_clipped = torch.clamp(pred_values, old_values - config.value_clip_eps, old_values + config.value_clip_eps)
        vf_losses1 = torch.square(pred_values - returns)
        vf_losses2 = torch.square(vpred_clipped - returns)
        value_losses = 0.5 * torch.max(vf_losses1, vf_losses2).float()
        vf_clipfrac = torch.gt(vf_losses2, vf_losses1)
        value_error = (pred_values - returns).pow(2).detach()

        # Apply mask and mean over seq_length dimension
        pg_losses = masked_mean(pg_losses, loss_masks, dim=1)  # [batch_size]
        entropies = masked_mean(entropies, loss_masks, dim=1)  # [batch_size]
        value_losses = masked_mean(value_losses, loss_masks, dim=1)  # [batch_size]
        advantages = masked_mean(advantages, loss_masks, dim=1)  # [batch_size]
        returns = masked_mean(returns, loss_masks, dim=1)  # [batch_size]
        pred_values = masked_mean(pred_values, loss_masks, dim=1)  # [batch_size]
        old_values = masked_mean(old_values, loss_masks, dim=1)  # [batch_size]
        value_error = masked_mean(value_error, loss_masks, dim=1)  # [batch_size]
        approxkl = masked_mean(approxkl, loss_masks, dim=1)  # [batch_size]
        pg_clipfrac = masked_mean(pg_clipfrac, loss_masks, dim=1)  # [batch_size]
        vf_clipfrac = masked_mean(vf_clipfrac, loss_masks, dim=1)  # [batch_size]

        rewards = masked_sum(rewards, loss_masks, dim=1)  # [batch_size]
        kl = masked_sum(kl, loss_masks, dim=1)  # [batch_size]
        kl_score = masked_sum(kl_score, loss_masks, dim=1)  # [batch_size]

        # Combined Loss
        total_losses = -pg_losses + config.value_loss_coef * value_losses  # [batch_size]
        total_loss = total_losses.mean()  # needs to be a scalar

        # All stats have shape [batch_size]
        stats = {
            'loss/policy': pg_losses.detach().to(device=self.device, dtype=self.compute_dtype),
            'loss/value': value_losses.detach().to(device=self.device, dtype=self.compute_dtype),
            'loss/total': total_losses.detach().to(device=self.device, dtype=self.compute_dtype),
            'policy/entropy': entropies.detach().to(device=self.device, dtype=self.compute_dtype),
            'policy/approxkl': approxkl.detach().to(device=self.device, dtype=self.compute_dtype),
            'policy/clipfrac': pg_clipfrac.detach().to(device=self.device, dtype=self.compute_dtype),
            'value/error': value_error.detach().to(device=self.device, dtype=self.compute_dtype),
            'value/clipfrac': vf_clipfrac.detach().to(device=self.device, dtype=self.compute_dtype),
            'objective/kl': kl.detach().to(device=self.device, dtype=self.compute_dtype),
            'objective/kl_score': kl_score.detach().to(device=self.device, dtype=self.compute_dtype),
            'objective/rewards': rewards.detach().to(device=self.device, dtype=self.compute_dtype),
            'objective/returns': returns.detach().to(device=self.device, dtype=self.compute_dtype),
        }

        del batch, loss_masks, rewards, returns, advantages

        return total_loss, stats

    def _compute_returns_and_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes returns and advantages using Generalized Advantage Estimation (GAE) for a batch of sequences.

        Args:
            rewards (torch.Tensor): Tensor of shape (batch_size, max_seq_len) containing rewards.
            values (torch.Tensor): Tensor of shape (batch_size, max_seq_len) containing value estimates.
            masks (torch.Tensor): Boolean tensor of shape (batch_size, max_seq_len) indicating valid positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors containing returns and advantages.
        """
        assert rewards.shape == values.shape == masks.shape
        assert rewards.dim() == values.dim() == masks.dim() == 2
        batch_size, max_seq_len = rewards.shape
        device = values.device

        if not masks.dtype == torch.bool:
            masks = masks.bool()

        # Initialize tensors for returns and advantages
        returns = torch.zeros_like(rewards, device=device)
        advantages = torch.zeros_like(rewards, device=device)

        for i in range(batch_size):
            # Compute returns and advantages for a single episode
            _returns, _advantages = self._compute_masked_returns_and_advantages_v2(
                rewards=rewards[i], values=values[i], masks=masks[i]
            )

            # Assign the computed returns and advantages back to the full tensors
            returns[i] = _returns
            advantages[i] = _advantages

        return returns, advantages

    def _compute_masked_returns_and_advantages_v2(
        self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes returns and advantages considering only assistant turns.

        Args:
            rewards (torch.Tensor): Tensor of rewards for each turn, shape [seq_len].
            values (torch.Tensor): Tensor of value predictions for each turn, shape [seq_len].
            masks (torch.Tensor): Binary mask (0 for user, 1 for assistant), shape [seq_len].

        Returns:
            tuple: (returns, advantages) - tensors of the original shape, with values
                for assistant turns and zeros for user turns.
        """
        assert rewards.shape == values.shape == masks.shape
        assert rewards.dim() == values.dim() == masks.dim() == 1

        # Extract rewards and values for assistant turns
        assistant_returns, assistant_advantages = self._compute_returns_and_advantages_for_single_episode(
            rewards=rewards[masks],
            values=values[masks],
        )

        # Map back to the original sequence corresponding to assistant's positions
        returns = torch.zeros_like(rewards, dtype=self.compute_dtype, device=self.device)
        advantages = torch.zeros_like(rewards, dtype=self.compute_dtype, device=self.device)
        returns[masks] = assistant_returns
        advantages[masks] = assistant_advantages

        return returns, advantages

    def _compute_returns_and_advantages_for_single_episode(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes returns and advantages using Generalized Advantage Estimation (GAE) for a single episode.
        We assume the inputs does not contain pad-tokens or user-turns.

        Args:
            rewards (torch.Tensor): Tensor of shape (seq_len,) containing rewards.
            values (torch.Tensor): Tensor of shape (seq_len,) containing value estimates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors containing returns and advantages.
        """
        assert values.dim() == rewards.dim() == 1
        assert values.size(0) == rewards.size(0)

        last_gae = 0
        advantages_reversed = []
        response_length = rewards.size(0)

        gamma = self.train_cfg.gamma  # self.get_dynamic_discount(len(rewards))
        gae_lambda = self.train_cfg.gae_lambda

        for t in reversed(range(response_length)):
            next_values = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + gamma * next_values - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages_reversed.append(last_gae)
        advantages_list = advantages_reversed[::-1]  # Reverse first
        advantages = torch.tensor(advantages_list, dtype=self.compute_dtype, device=self.device)
        returns = advantages + values
        return returns, advantages

    @torch.no_grad()
    def _prepare_ppo_transitions(self, episodes: List[ProcessedEpisode]) -> List[PPOSample]:
        """Do forward pass using the policy model and reference model to get logits and values"""
        self.policy_engine.eval()
        self.ref_policy_engine.eval()
        self.ref_policy_engine = self.ref_policy_engine.to(self.device)
        self._clear_gpu_memory()

        data_loader = self._create_data_loader(episodes, batch_size=self.batch_size_per_gpu, shuffle=False, for_prepare=True)

        all_transitions: List[PPOSample] = []

        batch: PPOSample = None
        for batch, lengths in tqdm(data_loader, desc='Processing PPO transitions'):
            states_tm1 = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            temperatures = batch.temperatures.to(self.device)
            states_tm1, attn_mask = self._prepare_model_inputs(states_tm1)

            model_outputs = self.policy_engine.forward(
                input_ids=states_tm1,
                attention_mask=attn_mask,
                return_dict=True,
                use_cache=False,
            )

            values = model_outputs.values.cpu()
            # apply temperature scaling similar to how it's done during sample collection
            pi_logits = self._apply_temperature_scale_to_logits(model_outputs.logits, temperatures)
            pi_logprobs = compute_logprobs_from_logits(pi_logits, actions).cpu()
            del model_outputs, pi_logits

            ref_outputs = self.ref_policy_engine.forward(
                input_ids=states_tm1,
                attention_mask=attn_mask,
                return_dict=True,
                use_cache=False,
            )
            states_tm1 = states_tm1.cpu()
            del attn_mask

            ref_logits = self._apply_temperature_scale_to_logits(ref_outputs.logits, temperatures)
            ref_logprobs = compute_logprobs_from_logits(ref_logits, actions).cpu()
            del ref_logits, ref_outputs

            # Compute returns and advantages
            for i, ep_len in enumerate(lengths):
                transition = PPOSample(
                    states=states_tm1[i, :ep_len].cpu(),
                    values=values[i, :ep_len].cpu(),
                    actions=actions[i, :ep_len].cpu(),
                    rewards=batch.rewards[i, :ep_len].cpu(),
                    temperatures=batch.temperatures[i, :ep_len].cpu(),
                    pi_logprobs=pi_logprobs[i, :ep_len].cpu(),
                    ref_logprobs=ref_logprobs[i, :ep_len].cpu(),
                    loss_masks=batch.loss_masks[i, :ep_len].cpu(),
                )
                all_transitions.append(transition)

        del data_loader

        self._offload_reference_model()
        return all_transitions

    @staticmethod
    def _apply_temperature_scale_to_logits(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        Safely scale logits by temperature, handling zero temperatures.

        Args:
            logits: tensor of shape [batch_size, sequence, vocab_size]
            temperature: tensor of shape [batch_size, sequence]

        Returns:
            Scaled logits with same shape as input
        """
        assert logits.dim() == 3
        assert temperatures.dim() == 2
        assert logits.size(0) == temperatures.size(0)
        assert logits.size(1) == temperatures.size(1)
        eps = 1e-8
        temp = temperatures.unsqueeze(-1)  # [batch_size, sequence, 1]
        safe_temp = torch.clamp(temp, min=eps)

        # Compute scaled logits
        scaled_logits = logits.div(safe_temp)

        # Create zero temperature mask without full expansion
        zero_temp_mask = temp < eps  # Shape: [batch_size, sequence, 1]

        # Use masked_fill_ which is more memory efficient than where
        # This operates in-place and only on the affected elements
        scaled_logits = scaled_logits.masked_fill_(zero_temp_mask, 0.0)

        # Add back the original logits where temperature was zero
        return scaled_logits.add_(logits.mul(zero_temp_mask))

    def _offload_reference_model(self, to_cpu: bool = True):
        self.ref_policy_engine = self.ref_policy_engine.to('cpu' if to_cpu else self.device)
        self._clear_gpu_memory()

    # def _offload_policy_model(self, to_cpu: bool = True):
    #     self.policy_model.to('cpu' if to_cpu else self.device)
    #     self._clear_gpu_memory()

    def _log_iteration_stats(self, iter_stats: Dict[str, Any]):
        """Log iteration stats"""
        iter_stats.update(self._get_common_stats())
        logger.info(f"Learner stats: {iter_stats}")
        if self.tracker:
            self.tracker.log_learner_iteration_stats(iter_stats)

    @staticmethod
    def _prep_collate_fn(batch: List[ProcessedEpisode], pad_id: int, dtype: torch.dtype) -> Tuple[PPOSample, List[int]]:
        """
        Custom collate function to pad sequences and create attention masks.
        Returns padded s_tm1, a_t tensors and the original lengths.
        """

        batch_size = len(batch)
        max_seq_len = max([len(item.token_ids) for item in batch])

        token_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        rewards = torch.full((batch_size, max_seq_len), 0.0, dtype=dtype)
        loss_masks = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)
        temperatures = torch.full((batch_size, max_seq_len), 0.0, dtype=dtype)

        for i, item in enumerate(batch):
            seq_len = len(item.token_ids)
            token_ids[i, :seq_len] = torch.from_numpy(item.token_ids).long()
            rewards[i, :seq_len] = torch.from_numpy(item.rewards).to(dtype)
            loss_masks[i, :seq_len] = torch.from_numpy(item.loss_masks).bool()
            temperatures[i, :seq_len] = torch.from_numpy(item.temperatures).to(dtype)

        # one step shift to get state and action
        states_tm1 = token_ids[..., :-1].clone()
        actions = token_ids[..., 1:].clone()
        rewards = rewards[..., 1:]
        temperatures = temperatures[..., 1:]
        loss_masks = loss_masks[..., 1:]

        # Calculate original lengths, use -1 to incorporate offset from state to action
        lengths = [len(item.token_ids) - 1 for item in batch]

        return (
            PPOSample(
                states=states_tm1,
                actions=actions,
                pi_logprobs=None,  # will be computed later
                ref_logprobs=None,
                rewards=rewards,
                temperatures=temperatures,
                loss_masks=loss_masks,
            ),
            lengths,
        )

    @staticmethod
    def _train_collate_fn(batch: List[PPOSample], pad_id: int, dtype: torch.dtype) -> PPOSample:
        """
        Custom collate function to pad sequences and create attention masks.
        Returns padded tensors for PPO training data and the original lengths.
        """
        # Extract sequences
        batch_size = len(batch)
        max_seq_len = max([len(item.states) for item in batch])

        states = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        actions = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        pi_logprobs = torch.full((batch_size, max_seq_len), 1e-8, dtype=dtype)  # avoid NaNs
        ref_logprobs = torch.full((batch_size, max_seq_len), 1e-8, dtype=dtype)
        values = torch.full((batch_size, max_seq_len), 0.0, dtype=dtype)
        rewards = torch.full((batch_size, max_seq_len), 0.0, dtype=dtype)
        temperatures = torch.full((batch_size, max_seq_len), 0.0, dtype=dtype)
        loss_masks = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)

        for i, item in enumerate(batch):
            seq_len = len(item.states)
            states[i, :seq_len] = item.states
            actions[i, :seq_len] = item.actions
            pi_logprobs[i, :seq_len] = item.pi_logprobs
            ref_logprobs[i, :seq_len] = item.ref_logprobs
            values[i, :seq_len] = item.values
            rewards[i, :seq_len] = item.rewards
            temperatures[i, :seq_len] = item.temperatures
            loss_masks[i, :seq_len] = item.loss_masks

        return PPOSample(
            states=states,
            actions=actions,
            pi_logprobs=pi_logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            rewards=rewards,
            temperatures=temperatures,
            loss_masks=loss_masks,
        )
