"""PPO leaner class for model training using deepspeed engine."""

import glob
import logging
import math
import os
import random
import shutil
import time
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BitsAndBytesConfig

from rl4llm.core.base_agent import BaseAgent
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
from rl4llm.utils import (
    TrainingTracker,
    load_yaml_config_file,
    save_to_json_file,
    save_yaml_config_file,
    set_seed,
)


class Learner(BaseAgent):
    """Implements the RL PPO leaner for training the large language model using deepspeed."""

    def __init__(
        self,
        config: Dict[str, Any],
        local_rank: int,
        tracker: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, local_rank, tracker=tracker, logger=logger)
        self.policy_engine: deepspeed.DeepSpeedEngine = self._init_policy_engine()
        self.reference_engine: deepspeed.InferenceEngine = self._init_reference_engine()
        self.sample_processor: EpisodeProcessor = EpisodeProcessor(tokenizer=self.tokenizer)
        self.batch_size_per_gpu: int = self.config['deepspeed']['train_micro_batch_size_per_gpu']
        self.batch_size: int = self._calculate_batch_size()
        self.ckpt_dir: str = self.tracker.output_paths['checkpoints'] if self.tracker else "/tmp"
        self.train_cfg: PPOConfig = PPOConfig(**self.config['training_config'])
        self.update_count = 0
        self.iteration_count = 0
        self.episode_count = 0

    def _load_policy_model(self) -> PreTrainedModel:
        """Loads the causal LM for policy and reference models."""
        model_config = self.config['model']
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": self.dtype,
            "use_cache": False,
            "attn_implementation": model_config.get('attn_implementation', 'flash_attention_2'),
        }
        if model_config['load_in_4bit']:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = nf4_config

        model = CustomQwen2Model.from_pretrained(**model_kwargs)

        # Setup activation checkpointing
        if model_config.get('activation_checkpoint', False):
            self.logger.info('Setup activation checkpoint...')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Initialize weights for value head if needed
        if model_config.get('initialize_value_weights', False):
            if hasattr(model, "value_output") and model.value_output:
                self.logger.info('Initialize value head weights...')
                num_layers = model.config.num_hidden_layers
                for module in model.value_output.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)

        return model

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

    def _init_reference_engine(self) -> deepspeed.InferenceEngine:
        """Initializes the DeepSpeed reference inference engine."""
        ref_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype, use_cache=False)
        for param in ref_model.parameters():
            param.requires_grad = False
        reference_engine = self._create_deepspeed_inference_engine(ref_model)
        return reference_engine.to('cpu')  # Default on CPU

    def get_policy_grad_norm(self) -> float:
        """Compute the norm of the policy model's gradients."""
        with torch.no_grad():
            total = 0.0
            for param in self.policy_engine.module.parameters():
                should_gather = (
                    hasattr(param, 'ds_id')
                    and param.ds_status == deepspeed.runtime.zero.partition_parameters.ZeroParamStatus.NOT_AVAILABLE
                )
                with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                    total += float(param.float().norm())
        return total

    def save_policy_model(
        self,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save the policy model to the output directory."""

        if is_best:
            # Save final checkpoint in a special folder
            save_path = os.path.join(self.ckpt_dir, 'best')
        elif is_final:
            # Save final checkpoint in a special folder
            save_path = os.path.join(self.ckpt_dir, 'final')
        else:
            # Create a new checkpoint folder with step
            save_path = self.ckpt_dir

        self.save_checkpoint(self.policy_engine, save_path)

    def on_exit(self):
        """Cleanup on exit."""
        if self.tracker is not None:
            self.tracker.flush()
            self.tracker.close()

        if self.train_cfg.checkpoint_enabled:
            self.save_policy_model(is_final=True)

    def train(self, episodes: List[Episode]) -> None:
        """Run PPO training epochs."""
        episodes = self.sample_processor.process_episodes(episodes)
        self.policy_engine = self.policy_engine.to(self.device)  # Move engines to device
        self.reference_engine = self.reference_engine.to(self.device)
        torch.cuda.empty_cache()
        transitions = self._prepare_ppo_transitions(episodes)
        self.reference_engine = self.reference_engine.to('cpu')  # Offload ref engine after preparing transitions
        torch.cuda.empty_cache()
        self.policy_engine.train()  # Set policy model to train mode
        data_loader = DataLoader(  # Create DataLoader
            transitions,
            batch_size=self.batch_size_per_gpu,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=partial(self._train_collate_fn, pad_id=self.pad_token_id, dtype=self.dtype),
        )
        total_steps = math.ceil(self.train_cfg.num_epochs * len(episodes) / self.batch_size)
        pbar = tqdm(desc='Training steps', unit='batch', total=total_steps)
        accumulated_iter_stats = defaultdict(list)

        for epoch in range(self.train_cfg.num_epochs):
            accumulated_batch_stats = defaultdict(list)
            for mini_batch in data_loader:
                metrics = self._process_minibatch(mini_batch)  # Process mini-batch
                for name, values in metrics.items():  # Accumulate stats
                    accumulated_batch_stats[name].extend(values)
                    accumulated_iter_stats[name].extend(values)

                if self.policy_engine.is_gradient_accumulation_boundary():  # Gradient accumulation boundary
                    self.update_count += 1
                    pbar.update(1)
                    batch_stats = self.aggregate_stats(accumulated_batch_stats)  # Aggregate batch stats
                    elapsed_time = pbar.format_dict.get('elapsed', 0)
                    batch_stats['step_time'] = round(elapsed_time / max(pbar.n, 1), 4)
                    self._log_batch_stats(batch_stats)  # Log batch stats
                    accumulated_batch_stats.clear()

        elapsed_time = pbar.format_dict.get('elapsed', 0)  # Finalize iteration
        pbar.close()
        self.iteration_count += 1
        self.episode_count += len(accumulated_iter_stats.get('loss/policy', []))
        iter_stats = self.aggregate_stats(accumulated_iter_stats)  # Aggregate iter stats
        iter_stats.update(
            {
                'elapsed/time': round(elapsed_time, 4),
                'elapsed/step_time': round(elapsed_time / max(pbar.total, 1) if pbar.total else 0, 4),
                'elapsed/updates': self.update_count,
                'elapsed/episodes': self.episode_count,
            }
        )
        self._log_iteration_stats(iter_stats)  # Log iter stats

        if self.train_cfg.checkpoint_enabled and self.iteration_count % self.train_cfg.checkpoint_interval == 0:
            self.save_policy_model()

    def is_zero3_enabled(self) -> bool:
        """Check if ZeRO-3 is enabled."""
        ds_config = self.config['deepspeed']
        return (
            'zero_optimization' in ds_config
            and 'stage' in ds_config['zero_optimization']
            and ds_config['zero_optimization']['stage'] == 3
        )

    def get_lasted_policy_weights(self) -> Dict[str, torch.Tensor]:
        """Retrieves consolidated 16-bit model state dict."""
        return self.get_model_state_dict(self.policy_engine)

    def offload_for_inference(self):
        """Offload policy and reference models to CPU for inference."""
        self.policy_engine = self.policy_engine.to('cpu')
        self.reference_engine = self.reference_engine.to('cpu')
        torch.cuda.empty_cache()

    def _get_params_groups(self, policy_model: torch.nn.Module, optimizer_config: Dict[str, Any]) -> List[Dict]:
        """Construct parameter groups for optimizer."""
        opt_params = optimizer_config['params']
        lr = float(opt_params['lr'])
        weight_decay = float(opt_params['weight_decay'])

        policy_params = []
        value_params = []
        nodecay_params = []
        for name, param in policy_model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['norm', 'tok_embeddings']):
                    nodecay_params.append(param)
                elif 'value_output' in name or "value_head" in name:
                    value_params.append(param)
                else:
                    policy_params.append(param)

        return [
            {'params': nodecay_params, 'lr': lr, 'weight_decay': 0.0, 'name': 'policy_nodecay'},
            {'params': policy_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'policy'},
            {'params': value_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'value'},
        ]

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

    def _prepare_model_inputs(self, input_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare model inputs and attention mask."""
        attention_mask = (input_tokens != self.pad_token_id).bool()
        return input_tokens.to(self.device), attention_mask.to(self.device)

    def _process_minibatch(self, mini_batch: PPOSample) -> Dict[str, np.array]:
        """Process a mini-batch and compute loss and metrics."""
        states, attn_mask = self._prepare_model_inputs(mini_batch.states)
        outputs = self.policy_engine.forward(input_ids=states, attention_mask=attn_mask, return_dict=True, use_cache=False)
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

        entropies = compute_entropy_from_logits(pred_pi_logits)  # Compute entropy and logprobs
        pi_logprobs = compute_logprobs_from_logits(pred_pi_logits, actions)

        kl = (pi_logprobs - ref_logprobs) * loss_masks  # Compute KL and KL-based reward score
        kl_score = -config.kl_coef * kl
        mixed_rewards = (rewards + kl_score) * loss_masks

        if self.train_cfg.normalize_rewards:  # Normalize rewards if configured
            mixed_rewards = masked_normalize(mixed_rewards, loss_masks)

        returns, advantages = self._compute_returns_and_advantages(rewards=mixed_rewards, values=old_values, masks=loss_masks)

        if self.train_cfg.normalize_advantages:  # Normalize advantages if configured
            advantages = masked_normalize(advantages, loss_masks)

        ratio = torch.exp(pi_logprobs - behavior_logprobs)  # PPO policy loss
        surr1 = ratio * advantages.detach()
        surr2 = ratio.clamp(1 - config.policy_clip_eps, 1 + config.policy_clip_eps) * advantages.detach()
        pg_losses = -torch.min(surr1, surr2)
        clipped = ratio.gt(1 + config.policy_clip_eps) | ratio.lt(1 - config.policy_clip_eps)
        pg_clipfrac = torch.as_tensor(clipped, dtype=ratio.dtype)
        approxkl = (pi_logprobs - behavior_logprobs).detach()

        vpred_clipped = torch.clamp(
            pred_values, old_values - config.value_clip_eps, old_values + config.value_clip_eps
        )  # Value loss
        vf_losses1 = torch.square(pred_values - returns)
        vf_losses2 = torch.square(vpred_clipped - returns)
        value_losses = 0.5 * torch.max(vf_losses1, vf_losses2).float()
        vf_clipfrac = torch.gt(vf_losses2, vf_losses1)
        value_error = (pred_values - returns).pow(2).detach()

        pg_losses = masked_mean(pg_losses, loss_masks, dim=1)  # Apply mask and mean
        value_losses = masked_mean(value_losses, loss_masks, dim=1)
        entropies = masked_mean(entropies, loss_masks, dim=1)
        value_error = masked_mean(value_error, loss_masks, dim=1)
        approxkl = masked_mean(approxkl, loss_masks, dim=1)
        pg_clipfrac = masked_mean(pg_clipfrac, loss_masks, dim=1)
        vf_clipfrac = masked_mean(vf_clipfrac, loss_masks, dim=1)
        returns = masked_mean(returns, loss_masks, dim=1)

        rewards = masked_sum(rewards, loss_masks, dim=1)  # Sum rewards, kl, kl_score
        kl = masked_sum(kl, loss_masks, dim=1)
        kl_score = masked_sum(kl_score, loss_masks, dim=1)

        total_losses = pg_losses + config.value_loss_coef * value_losses  # Combined loss
        loss = total_losses.mean()

        stats = {  # Stats dictionary
            'loss/policy': pg_losses.detach().to(device=self.device, dtype=self.dtype),
            'loss/value': value_losses.detach().to(device=self.device, dtype=self.dtype),
            'loss/total': total_losses.detach().to(device=self.device, dtype=self.dtype),
            'policy/entropy': entropies.detach().to(device=self.device, dtype=self.dtype),
            'policy/approxkl': approxkl.detach().to(device=self.device, dtype=self.dtype),
            'policy/clipfrac': pg_clipfrac.detach().to(device=self.device, dtype=self.dtype),
            'value/error': value_error.detach().to(device=self.device, dtype=self.dtype),
            'value/clipfrac': vf_clipfrac.detach().to(device=self.device, dtype=self.dtype),
            'objective/kl': kl.detach().to(device=self.device, dtype=self.dtype),
            'objective/kl_score': kl_score.detach().to(device=self.device, dtype=self.dtype),
            'objective/rewards': rewards.detach().to(device=self.device, dtype=self.dtype),
            'objective/returns': returns.detach().to(device=self.device, dtype=self.dtype),
        }
        return loss, stats

    def _compute_returns_and_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes returns and advantages using GAE."""
        batch_size, max_seq_len = rewards.shape
        device = values.device
        returns = torch.zeros_like(rewards, device=device)
        advantages = torch.zeros_like(rewards, device=device)

        for i in range(batch_size):
            assistant_returns, assistant_advantages = self._compute_returns_and_advantages(
                rewards=rewards[i], values=values[i], masks=masks[i]
            )
            returns[i] = assistant_returns
            advantages[i] = assistant_advantages
        return returns, advantages

    def _compute_returns_and_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes returns and advantages considering only assistant turns."""
        assistant_returns, assistant_advantages = self._compute_gae_for_episode(rewards=rewards[masks], values=values[masks])
        returns = torch.zeros_like(rewards, dtype=self.dtype, device=self.device)
        advantages = torch.zeros_like(rewards, dtype=self.dtype, device=self.device)
        returns[masks] = assistant_returns
        advantages[masks] = assistant_advantages
        return returns, advantages

    def _compute_gae_for_episode(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes GAE for a single episode."""
        gamma = self.train_cfg.gamma
        gae_lambda = self.train_cfg.gae_lambda
        last_gae = 0
        advantages_reversed = []
        response_length = rewards.size(0)

        for t in reversed(range(response_length)):
            next_values = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + gamma * next_values - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages_reversed.append(last_gae)

        advantages = torch.tensor(advantages_reversed[::-1], dtype=self.dtype, device=self.device)  # Reverse and create tensors
        returns = advantages + values
        return returns, advantages

    @torch.no_grad()
    def _prepare_ppo_transitions(self, episodes: List[ProcessedEpisode]) -> List[PPOSample]:
        """Prepare PPO transitions by forward pass through policy and reference models."""
        data_loader = DataLoader(  # Create DataLoader
            episodes,
            batch_size=self.batch_size_per_gpu,
            shuffle=False,
            collate_fn=partial(self._preprocess_collate_fn, pad_id=self.pad_token_id, dtype=self.dtype),
        )
        all_transitions: List[PPOSample] = []

        for batch, lengths in tqdm(data_loader, desc='Processing PPO transitions'):
            states = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            temperatures = batch.temperatures.to(self.device)
            states, attn_mask = self._prepare_model_inputs(states)

            model_outputs = self.policy_engine(
                input_ids=states, attention_mask=attn_mask, return_dict=True, use_cache=False
            )  # Policy forward pass
            temperatures = temperatures.unsqueeze(-1)  # Apply temperature scaling
            pi_logits = model_outputs.logits / (temperatures + 1e-8)
            pi_logprobs = compute_logprobs_from_logits(pi_logits, actions).cpu()
            values = model_outputs.values.cpu()

            del model_outputs, pi_logits

            ref_outputs = self.reference_engine(
                input_ids=states, attention_mask=attn_mask, return_dict=True, use_cache=False
            )  # Reference forward pass

            ref_logits = ref_outputs.logits / (temperatures + 1e-8)
            ref_logprobs = compute_logprobs_from_logits(ref_logits, actions).cpu()

            del ref_outputs, ref_logits

            for i, ep_len in enumerate(lengths):  # Create PPOSample transitions
                transition = PPOSample(
                    states=states[i, :ep_len].cpu(),
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
        return all_transitions

    @staticmethod
    def _preprocess_collate_fn(batch: List[ProcessedEpisode], pad_id: int, dtype: torch.dtype) -> Tuple[PPOSample, List[int]]:
        """Collate function for preprocessing episodes."""
        batch_size = len(batch)
        max_seq_len = max([len(item.token_ids) for item in batch])

        token_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)  # Initialize tensors
        rewards = torch.zeros((batch_size, max_seq_len), dtype=dtype)
        loss_masks = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        temperatures = torch.zeros((batch_size, max_seq_len), dtype=dtype)

        for i, item in enumerate(batch):  # Fill tensors from batch data
            seq_len = len(item.token_ids)
            token_ids[i, :seq_len] = torch.from_numpy(item.token_ids).long()
            rewards[i, :seq_len] = torch.from_numpy(item.rewards).to(dtype)
            loss_masks[i, :seq_len] = torch.from_numpy(item.loss_masks).bool()
            temperatures[i, :seq_len] = torch.from_numpy(item.temperatures).to(dtype)

        states = token_ids[..., :-1].clone()
        actions = token_ids[..., 1:].clone()  # Shift for states and actions
        rewards = rewards[..., 1:]
        temperatures = temperatures[..., 1:]
        loss_masks = loss_masks[..., 1:]
        lengths = [len(item.token_ids) - 1 for item in batch]

        return (
            PPOSample(
                states=states,
                actions=actions,
                pi_logprobs=None,
                ref_logprobs=None,
                rewards=rewards,
                temperatures=temperatures,
                loss_masks=loss_masks,
            ),
            lengths,
        )

    @staticmethod
    def _train_collate_fn(batch: List[PPOSample], pad_id: int, dtype: torch.dtype) -> PPOSample:
        """Collate function for training PPO samples."""
        batch_size = len(batch)
        max_seq_len = max([len(item.states) for item in batch])

        states = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)  # Initialize tensors
        actions = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        pi_logprobs = torch.full((batch_size, max_seq_len), 1e-8, dtype=dtype)
        ref_logprobs = torch.full((batch_size, max_seq_len), 1e-8, dtype=dtype)
        values = torch.zeros((batch_size, max_seq_len), dtype=dtype)
        rewards = torch.zeros((batch_size, max_seq_len), dtype=dtype)
        temperatures = torch.zeros((batch_size, max_seq_len), dtype=dtype)
        loss_masks = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

        for i, item in enumerate(batch):  # Fill tensors from batch data
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
