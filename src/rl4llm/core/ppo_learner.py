"""PPO leaner class for model training using deepspeed engine."""

import logging
import math
import os
from collections import defaultdict
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BitsAndBytesConfig

from rl4llm.core.base_ds_class import BaseDeepSpeedClass
from rl4llm.core.episode_processor import EpisodeProcessor
from rl4llm.core.helper import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    masked_mean,
    masked_normalize,
    masked_sum,
)
from rl4llm.models import CustomQwen2Model
from rl4llm.types import Episode, PPOConfig, PPOSample, ProcessedEpisode
from rl4llm.utils import TrainingTracker


class PPOLearner(BaseDeepSpeedClass):
    """Implements the RL PPO leaner for training the large language model using deepspeed."""

    def __init__(
        self,
        config: Dict[str, Any],
        local_rank: int,
        tracker: Optional[TrainingTracker] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, local_rank, tracker=tracker, logger=logger)

        self.batch_size_per_gpu: int = self.config['deepspeed']['train_micro_batch_size_per_gpu']
        self.batch_size: int = self._calculate_batch_size()
        self.ckpt_dir: str = self.tracker.output_paths['checkpoints'] if self.tracker else "/tmp"
        self.train_cfg: PPOConfig = PPOConfig(**self.config['training_config'])
        self.policy_engine: deepspeed.DeepSpeedEngine = self._init_policy_engine()
        self.reference_engine: deepspeed.DeepSpeedEngine = self._init_reference_engine()
        self.sample_processor: EpisodeProcessor = EpisodeProcessor(tokenizer=self.tokenizer)

        self.update_count = 0
        self.iteration_count = 0
        self.episode_count = 0

        dist.barrier()  # Ensure all processes are synchronized

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

        if self.train_cfg.checkpoint_enabled and self.update_count > 50:
            self.save_policy_model(tag="final")

    def train(self, episodes: List[Episode]) -> None:
        """Run PPO training epochs."""
        processed_episodes = self.sample_processor.process_episodes(episodes)
        self.policy_engine = self.policy_engine.to(self.device)  # Move engines to device
        self.reference_engine = self.reference_engine.to(self.device)
        torch.cuda.empty_cache()
        transitions = self._prepare_ppo_transitions(processed_episodes)
        dist.barrier()
        self.reference_engine = self.reference_engine.to('cpu')  # Offload ref engine after preparing transitions
        torch.cuda.empty_cache()
        self.policy_engine.train()  # Set policy model to train mode

        sampler = DistributedSampler(transitions, shuffle=True, seed=self.seed)
        data_loader = DataLoader(  # Create DataLoader
            transitions,
            batch_size=self.batch_size_per_gpu,
            sampler=sampler,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self._train_collate_fn,
            drop_last=True,
        )
        total_steps = math.ceil(self.train_cfg.num_epochs * len(episodes) / self.batch_size)
        pbar = tqdm(desc='Training steps', unit='batch', total=total_steps, disable=not self._is_rank0())
        accumulated_iter_stats = defaultdict(list)

        # with torch.autograd.set_detect_anomaly(True):
        for epoch in range(self.train_cfg.num_epochs):
            accumulated_batch_stats = defaultdict(list)
            for mini_batch in data_loader:
                metrics = self._process_minibatch(mini_batch)  # Process mini-batch
                for name, values in metrics.items():
                    accumulated_batch_stats[name].extend(values)
                    accumulated_iter_stats[name].extend(values)

                if self.policy_engine.is_gradient_accumulation_boundary():
                    self.update_count += 1
                    pbar.update(1)
                    batch_stats = self._aggregate_stats(accumulated_batch_stats)
                    elapsed_time = pbar.format_dict.get('elapsed', 0)
                    batch_stats['step_time'] = round(elapsed_time / max(pbar.n, 1), 4)
                    self._log_batch_stats(batch_stats)
                    accumulated_batch_stats.clear()

                    if self.train_cfg.checkpoint_enabled and self.update_count % self.train_cfg.checkpoint_interval == 0:
                        self.save_policy_model()

        elapsed_time = pbar.format_dict.get('elapsed', 0)
        pbar.close()
        self.iteration_count += 1
        self.episode_count += len(episodes)
        self.logger.info('Aggregating iteration stats...')
        iter_stats = self._aggregate_stats(accumulated_iter_stats)
        iter_stats.update(
            {
                'elapsed/time': round(elapsed_time, 4),
                'elapsed/step_time': round(elapsed_time / max(pbar.total, 1) if pbar.total else 0, 4),
                'elapsed/updates': self.update_count,
                'elapsed/episodes': self.episode_count,
            }
        )
        self.logger.info('Logging iteration stats...')
        self._log_iteration_stats(iter_stats)  # Log iter stats
        dist.barrier()

    def get_lasted_policy_weights(self) -> Dict[str, torch.Tensor]:
        """Retrieves consolidated 16-bit model state dict."""
        return self._get_model_state_dict(self.policy_engine)

    def offload_for_inference(self):
        """Offload policy and reference models to CPU for inference."""
        self.policy_engine = self.policy_engine.to('cpu')
        self.reference_engine = self.reference_engine.to('cpu')
        torch.cuda.empty_cache()
        dist.barrier()

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

    def _init_reference_engine(self) -> deepspeed.DeepSpeedEngine:
        """Initializes the DeepSpeed reference inference engine."""
        self.logger.info(f"Initializing reference model from {self.pretrained_model_name_or_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path, torch_dtype=self.dtype, use_cache=False
        )
        for param in ref_model.parameters():
            param.requires_grad = False

        stage = 3 if self._is_zero3_enabled() else 0
        ref_ds_config = {
            "steps_per_print": 1000,
            'train_micro_batch_size_per_gpu': self.batch_size_per_gpu,
            'train_batch_size': self.batch_size if self.batch_size > 0 else 'auto',
            "zero_optimization": {
                "stage": stage,
                "stage3_param_persistence_threshold": "auto",
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
            "f16": {
                "enabled": self.dtype == torch.float16,
            },
            "bf16": {
                "enabled": self.dtype == torch.bfloat16,
            },
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }

        reference_engine: deepspeed.DeepSpeedEngine = None
        reference_engine, *_ = deepspeed.initialize(
            model=ref_model,
            model_parameters=None,
            config=ref_ds_config,
            args={"local_rank": self.local_rank},
            dist_init_required=True,
        )

        return reference_engine.to('cpu')  # Default on CPU

    def _process_minibatch(self, mini_batch: PPOSample) -> Dict[str, np.array]:
        """Process a mini-batch and compute loss and metrics."""
        states, attn_mask = self._prepare_model_inputs(mini_batch.states)
        outputs = self.policy_engine.forward(
            input_ids=states,
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
        self, pred_pi_logits: torch.Tensor, pred_values: torch.Tensor, batch: PPOSample
    ) -> Tuple[torch.Tensor, Dict[str, np.array]]:
        """Compute PPO loss and metrics."""
        config = self.train_cfg
        device = pred_pi_logits.device

        actions = batch.actions.to(device)
        behavior_logprobs = batch.pi_logprobs.to(device)
        ref_logprobs = batch.ref_logprobs.to(device)
        old_values = batch.values.to(device)
        rewards = batch.rewards.to(device)
        loss_masks = batch.loss_masks.bool().to(device)

        rewards *= loss_masks
        old_values *= loss_masks  # Apply loss masks

        pi_logprobs = compute_logprobs_from_logits(pred_pi_logits, actions)

        # Compute KL using new method mentioned in GRPO paper
        # http://joschu.net/blog/kl-approx.html
        kl = (torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1) * loss_masks

        returns, advantages = self._compute_masked_returns_and_gae_advantages(
            rewards=rewards, values=old_values, masks=loss_masks
        )

        # # Compute KL and KL-based reward score
        # mixed_rewards = (rewards - config.kl_coef * kl) * loss_masks
        # if self.train_cfg.normalize_rewards:
        #     mixed_rewards = masked_normalize(mixed_rewards, loss_masks)

        # returns, advantages = self._compute_masked_returns_and_gae_advantages(
        #     rewards=mixed_rewards, values=old_values, masks=loss_masks
        # )

        if self.train_cfg.normalize_advantages:
            advantages = masked_normalize(advantages, loss_masks)

        # PPO clipped surrogate PG loss
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(1 - config.policy_clip_eps, 1 + config.policy_clip_eps)
        pg_losses = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())
        clipped = ratio.gt(1 + config.policy_clip_eps) | ratio.lt(1 - config.policy_clip_eps)
        pg_clipfrac = torch.as_tensor(clipped, dtype=ratio.dtype)
        approxkl = (pi_logprobs - behavior_logprobs).detach()

        # Value loss
        vpred_clipped = torch.clamp(pred_values, old_values - config.value_clip_eps, old_values + config.value_clip_eps)
        vf_losses1 = torch.square(pred_values - returns)
        vf_losses2 = torch.square(vpred_clipped - returns)
        value_losses = 0.5 * torch.max(vf_losses1, vf_losses2).float()
        vf_clipfrac = torch.gt(vf_losses2, vf_losses1)
        value_error = (pred_values - returns).pow(2).detach()

        # Apply mask to get all data with shape: [batch_size]
        pg_losses = masked_mean(pg_losses, loss_masks, dim=1)
        value_losses = masked_mean(value_losses, loss_masks, dim=1)
        approxkl = masked_mean(approxkl, loss_masks, dim=1)
        value_error = masked_mean(value_error, loss_masks, dim=1)
        pg_clipfrac = masked_mean(pg_clipfrac, loss_masks, dim=1)
        vf_clipfrac = masked_mean(vf_clipfrac, loss_masks, dim=1)
        returns = masked_mean(returns, loss_masks, dim=1)
        summed_kl = masked_sum(kl, loss_masks, dim=1)
        rewards = masked_sum(rewards, loss_masks, dim=1)

        # Combined loss
        kl_penalties = config.kl_loss_coef * masked_mean(kl, loss_masks, dim=1)
        value_losses = config.value_loss_coef * value_losses
        total_losses = -pg_losses + value_losses + kl_penalties
        loss = total_losses.mean()

        # Stats dictionary
        stats = {
            'loss/policy': pg_losses.detach().to(device=self.device, dtype=self.dtype),
            'loss/value': value_losses.detach().to(device=self.device, dtype=self.dtype),
            'loss/kl_penalty': kl_penalties.detach().to(device=self.device, dtype=self.dtype),
            'loss/total': total_losses.detach().to(device=self.device, dtype=self.dtype),
            'policy/approxkl': approxkl.detach().to(device=self.device, dtype=self.dtype),
            'policy/clipfrac': pg_clipfrac.detach().to(device=self.device, dtype=self.dtype),
            'value/error': value_error.detach().to(device=self.device, dtype=self.dtype),
            'value/clipfrac': vf_clipfrac.detach().to(device=self.device, dtype=self.dtype),
            'objective/kl': summed_kl.detach().to(device=self.device, dtype=self.dtype),
            'objective/rewards': rewards.detach().to(device=self.device, dtype=self.dtype),
            'objective/returns': returns.detach().to(device=self.device, dtype=self.dtype),
        }
        return loss, stats

    def _compute_masked_returns_and_gae_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes returns and advantages using GAE."""
        batch_size, max_seq_len = rewards.shape
        device = values.device
        returns = torch.zeros_like(rewards, device=device)
        advantages = torch.zeros_like(rewards, device=device)

        for i in range(batch_size):
            assistant_returns, assistant_advantages = self._compute_masked_returns_and_gae_advantages_for_single_episode(
                rewards=rewards[i], values=values[i], masks=masks[i]
            )
            returns[i] = assistant_returns
            advantages[i] = assistant_advantages
        return returns, advantages

    def _compute_masked_returns_and_gae_advantages_for_single_episode(
        self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes returns and advantages considering only assistant turns."""

        assistant_rewards = rewards[masks]
        assistant_values = values[masks]

        seq_len = len(assistant_rewards)

        gamma = (
            self._compute_dynamic_discount(
                seq_len,
                max_length=self.train_cfg.max_expected_length,
                min_discount=self.train_cfg.min_gamma,
                max_disount=self.train_cfg.max_gamma,
            )
            if self.train_cfg.dynamic_discount
            else self.train_cfg.gamma
        )
        gae_lambda = self.train_cfg.gae_lambda
        last_gae = 0
        advantages_reversed = []
        response_length = assistant_rewards.size(0)

        for t in reversed(range(response_length)):
            next_values = assistant_values[t + 1] if t < response_length - 1 else 0.0
            delta = assistant_rewards[t] + gamma * next_values - assistant_values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages_reversed.append(last_gae)

        # Reverse and create tensors
        assistant_advantages = torch.tensor(advantages_reversed[::-1], dtype=self.dtype, device=self.device)
        assistant_returns = assistant_advantages + assistant_values

        # set returns and advantages for asssitant turns
        returns = torch.zeros_like(rewards, dtype=self.dtype, device=self.device)
        advantages = torch.zeros_like(rewards, dtype=self.dtype, device=self.device)
        returns[masks] = assistant_returns
        advantages[masks] = assistant_advantages
        return returns, advantages

    # def _compute_masked_mc_returns(self, rewards: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    #     """Computes monte carlo returns."""
    #     batch_size, max_seq_len = rewards.shape
    #     device = rewards.device
    #     returns = torch.zeros_like(rewards, device=device)

    #     for i in range(batch_size):
    #         assistant_returns = self._compute_masked_mc_returns_for_single_episode(rewards=rewards[i], masks=masks[i])
    #         returns[i] = assistant_returns
    #     return returns

    # def _compute_masked_mc_returns_for_single_episode(self, rewards: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    #     """Computes monte carlo returns considering only assistant turns."""

    #     gamma = self.train_cfg.gamma
    #     assistant_rewards = rewards[masks]
    #     assistant_returns = torch.zeros_like(assistant_rewards, dtype=self.dtype, device=self.device)
    #     running_return = 0
    #     # Reverse compute returns (like standard GAE-lambda but without baseline)
    #     for t in reversed(range(len(assistant_rewards))):
    #         running_return = assistant_rewards[t] + gamma * running_return
    #         assistant_returns[t] = running_return

    #     returns = torch.zeros_like(rewards, dtype=self.dtype, device=self.device)
    #     returns[masks] = assistant_returns
    #     return returns

    @torch.no_grad()
    def _prepare_ppo_transitions(self, episodes: List[ProcessedEpisode]) -> List[PPOSample]:
        """Prepare PPO transitions by forward pass through policy and reference models."""
        data_loader = DataLoader(  # Create DataLoader
            episodes,
            batch_size=self.batch_size_per_gpu,
            shuffle=False,
            collate_fn=self._preprocess_collate_fn,
        )
        all_transitions: List[PPOSample] = []

        batch: PPOSample = None
        for batch, lengths in tqdm(data_loader, desc='Processing PPO transitions', disable=not self._is_rank0()):
            states = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            states, attn_mask = self._prepare_model_inputs(states)

            pi_outputs = self.policy_engine.forward(
                input_ids=states,
                attention_mask=attn_mask,
                return_dict=True,
                use_cache=False,
                return_values=True,  # compute values
            )

            values = pi_outputs.values.cpu()
            pi_logits = pi_outputs.logits
            pi_logprobs = compute_logprobs_from_logits(pi_logits, actions).cpu()
            del pi_outputs, pi_logits

            ref_outputs = self.reference_engine.forward(
                input_ids=states, attention_mask=attn_mask, return_dict=True, use_cache=False
            )

            ref_logits = ref_outputs.logits
            ref_logprobs = compute_logprobs_from_logits(ref_logits, actions).cpu()
            del ref_logits, ref_outputs

            for i, ep_len in enumerate(lengths):  # Create PPOSample transitions
                transition = PPOSample(
                    states=states[i, :ep_len].cpu(),
                    values=values[i, :ep_len].cpu(),
                    actions=actions[i, :ep_len].cpu(),
                    rewards=batch.rewards[i, :ep_len].cpu(),
                    pi_logprobs=pi_logprobs[i, :ep_len].cpu(),
                    ref_logprobs=ref_logprobs[i, :ep_len].cpu(),
                    loss_masks=batch.loss_masks[i, :ep_len].cpu(),
                )
                all_transitions.append(transition)

        del data_loader
        return all_transitions

    def _preprocess_collate_fn(self, batch: List[ProcessedEpisode]) -> Tuple[PPOSample, List[int]]:
        """Collate function for preprocessing episodes."""
        batch_size = len(batch)
        pad_id = self.pad_token_id
        dtype = self.dtype
        max_seq_len = max([len(item.token_ids) for item in batch])

        token_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)  # Initialize tensors
        rewards = torch.zeros((batch_size, max_seq_len), dtype=dtype)
        loss_masks = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

        for i, item in enumerate(batch):  # Fill tensors from batch data
            seq_len = len(item.token_ids)
            token_ids[i, :seq_len] = torch.from_numpy(item.token_ids).long()
            rewards[i, :seq_len] = torch.from_numpy(item.rewards).to(dtype)
            loss_masks[i, :seq_len] = torch.from_numpy(item.loss_masks).bool()

        states = token_ids[..., :-1].clone()
        actions = token_ids[..., 1:].clone()  # Shift for states and actions
        rewards = rewards[..., 1:]
        loss_masks = loss_masks[..., 1:]
        lengths = [len(item.token_ids) - 1 for item in batch]

        return (
            PPOSample(
                states=states,
                actions=actions,
                pi_logprobs=None,
                ref_logprobs=None,
                rewards=rewards,
                loss_masks=loss_masks,
            ),
            lengths,
        )

    def _train_collate_fn(self, batch: List[PPOSample]) -> PPOSample:
        """Collate function for training PPO samples."""
        batch_size = len(batch)

        pad_id = self.pad_token_id
        dtype = self.dtype
        max_seq_len = max([len(item.states) for item in batch])

        states = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)  # Initialize tensors
        actions = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        pi_logprobs = torch.full((batch_size, max_seq_len), 1e-8, dtype=dtype)
        ref_logprobs = torch.full((batch_size, max_seq_len), 1e-8, dtype=dtype)
        values = torch.zeros((batch_size, max_seq_len), dtype=dtype)
        rewards = torch.zeros((batch_size, max_seq_len), dtype=dtype)
        loss_masks = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

        for i, item in enumerate(batch):  # Fill tensors from batch data
            seq_len = len(item.states)
            states[i, :seq_len] = item.states
            actions[i, :seq_len] = item.actions
            pi_logprobs[i, :seq_len] = item.pi_logprobs
            ref_logprobs[i, :seq_len] = item.ref_logprobs
            values[i, :seq_len] = item.values
            rewards[i, :seq_len] = item.rewards
            loss_masks[i, :seq_len] = item.loss_masks

        return PPOSample(
            states=states,
            actions=actions,
            pi_logprobs=pi_logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            rewards=rewards,
            loss_masks=loss_masks,
        )
