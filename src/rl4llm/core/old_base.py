import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

from rl4llm.models import CustomQwen2Model
from rl4llm.utils import (
    TrainingTracker,
    cleanup_old_checkpoints,
    load_yaml_config_file,
    save_to_json_file,
    save_yaml_config_file,
    set_seed,
)


class BaseDeepSpeedClass:
    """Base class for RL agents (Actor and Learner)."""

    def __init__(
        self,
        config: Dict[str, Any],
        local_rank: int,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        tracker: Optional[TrainingTracker] = None,
        logger: Optional[logging.Logger] = None,
    ):
        assert dist.is_initialized(), 'Distributed environment must be initialized before creating RL Agent.'

        self.config = config
        self.local_rank = local_rank
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{local_rank}")
        self.dtype = dtype
        self.tracker = tracker
        self.logger = logger if logger else logging.getLogger(__name__)  # Use default logger if none provided
        self.model_name = self.config['model']['pretrained_model']
        self.pretrained_model_name_or_path = (
            self.config['model']['load_checkpoint']
            if 'load_checkpoint' in self.config['model'] and self.config['model']['load_checkpoint']
            else self.model_name
        )
        self.max_seq_len = self.config['model'].get('max_seq_len', 8192)
        self.seed = self.config['job'].get('seed', 123) + self.local_rank
        self.tokenizer: PreTrainedTokenizer = self._init_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = [self.tokenizer.eos_token, self.tokenizer.pad_token]

        self.policy_engine: deepspeed.DeepSpeedEngine = None

        set_seed(self.seed)

    def _is_rank0(self) -> bool:
        """Check if the current process is rank 0."""
        return self.local_rank == 0

    def _init_tokenizer(self) -> PreTrainedTokenizer:
        """Initialize tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if 'llama' in self.model_name.lower():
            tokenizer.pad_token = '<|reserved_special_token_0|>'
            tokenizer.pad_token_id = 128002
        elif tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _load_policy_model(self) -> PreTrainedModel:
        """Loads the causal LM for policy and reference models."""
        self.logger.info(f"Initializing pretrained model from {self.pretrained_model_name_or_path}")
        model_config = self.config['model']
        model_kwargs = {
            'pretrained_model_name_or_path': self.pretrained_model_name_or_path,
            'torch_dtype': self.dtype,
            'use_cache': False,
            'attn_implementation': model_config.get('attn_implementation', 'flash_attention_2'),
        }
        if model_config['load_in_4bit']:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs['quantization_config'] = nf4_config

        model = CustomQwen2Model.from_pretrained(**model_kwargs)

        # Setup activation checkpointing
        if model_config.get('activation_checkpoint', False):
            self.logger.info('Setup activation checkpoint...')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        # Initialize weights for value head if needed
        if model_config.get('initialize_value_weights', False):
            if hasattr(model, 'value_head') and model.value_head:
                self.logger.info('Initialize value head weights...')
                for module in model.value_head.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)

        return model

    def _create_deepspeed_inference_engine(
        self,
        model: PreTrainedModel,
    ) -> deepspeed.InferenceEngine:
        """Creates DeepSpeed inference engine."""
        if self.logger:
            self.logger.info('Creating inference engine...')
        tp_size = dist.get_world_size() if self._is_zero3_enabled() else 1
        ds_infer_config = {
            'tensor_parallel': {'tp_size': tp_size},
            'dtype': torch.half,
            'replace_with_kernel_inject': True,
            # "use_triton": True,
            'max_out_tokens': self.max_seq_len,
        }

        inference_engine: deepspeed.InferenceEngine = None
        inference_engine = deepspeed.init_inference(
            model=model,
            config=ds_infer_config,
            # base_dir="/dev/shm",
            checkpoint=None,
        )

        return inference_engine

    def _create_deepspeed_training_engine(
        self,
        model: PreTrainedModel,
    ) -> deepspeed.DeepSpeedEngine:
        """Creates DeepSpeed training engine."""
        if self.logger:
            self.logger.info('Creating training engine...')

        ds_config = self.config['deepspeed']

        # deepspeed's hybrid engine is a joke it's so slow with zero-3
        # if enable_hybrid_engine:
        #     if ds_config['zero_optimization']['stage'] != 3:
        #         raise ValueError("Hybrid engine only works with ZeRO-3.")

        #     self.logger.info("Enabling hybrid engine...")
        #     ds_config["hybrid_engine"] = {
        #         "enabled": enable_hybrid_engine,
        #         "max_out_tokens": self.max_seq_len,
        #         "inference_tp_size": dist.get_world_size(),
        #         "release_inference_cache": True,
        #         "pin_parameters": True,
        #         "tp_gather_partition_size": 8,
        #     }

        model_parameters = self._get_params_groups(model, ds_config['optimizer'])

        engine: deepspeed.DeepSpeedEngine = None
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            config=ds_config,
            args={'local_rank': self.local_rank},
            dist_init_required=True,
        )

        return engine

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
                if 'value_head' in name:
                    value_params.append(param)
                elif any(nd in name for nd in ['bias', 'layer_norm.weight', 'layernorm.weight', 'norm.weight']):
                    nodecay_params.append(param)
                else:
                    policy_params.append(param)

        return [
            {'params': nodecay_params, 'lr': lr, 'weight_decay': 0.0, 'name': 'policy_nodecay'},
            {'params': policy_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'policy'},
            {'params': value_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'value'},
        ]

    def _aggregate_stats(
        self, accumulated_stats: Dict[str, List[torch.Tensor]], var_keys: Optional[List[str]] = None
    ) -> Dict[str, float]:
        agg_stats = {}

        for key, values in accumulated_stats.items():
            if not values:
                continue

            # Convert list of tensors to a single tensor
            if isinstance(values, list):
                values_tensor = torch.stack(values)
            else:
                values_tensor = values

            # Compute sum and count locally
            local_sum = torch.sum(values_tensor)
            local_count = torch.tensor(len(values_tensor), device=self.device, dtype=torch.float)

            # Gather sums and counts across all ranks
            global_sums = self._gather_scalar_tensor(local_sum)
            global_counts = self._gather_scalar_tensor(local_count)

            # Calculate global mean
            total_sum = torch.sum(global_sums)
            total_count = torch.sum(global_counts)
            agg_stats[key] = (total_sum / total_count).item()

            # Optional: Compute variance if needed (requires full data)
            if var_keys and key in var_keys:
                # Note: Variance calculation requires gathering all data
                # Only include this if absolutely necessary
                all_values = self._safe_gather_tensor(values_tensor)  # Use safe gather
                agg_stats[f"{key}_var"] = torch.var(all_values).item()

        if 'value/error' in agg_stats and 'objective/returns_var' in agg_stats:
            agg_stats['value/var_explained'] = 1 - agg_stats['value/error'] / agg_stats['objective/returns_var']
        if 'objective/kl_score' in agg_stats and 'objective/rewards' in agg_stats:
            agg_stats['objective/total_reward'] = agg_stats['objective/kl_score'] + agg_stats['objective/rewards']
        return agg_stats

    def _gather_scalar_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather scalar tensors (sum/count) across all ranks."""
        if not dist.is_initialized():
            return tensor.unsqueeze(0)  # Return as 1-element tensor

        tensor = tensor.contiguous().to(self.device)
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    def _safe_gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely gather tensors of variable lengths using padding."""
        if not dist.is_initialized():
            return tensor

        tensor = tensor.contiguous().to(self.device)

        # Get sizes from all ranks
        local_size = torch.tensor(tensor.numel(), device=self.device, dtype=torch.long)
        sizes = [torch.empty_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)

        # Pad tensor to max size
        if tensor.numel() < max_size:
            pad_size = max_size - tensor.numel()
            padded_tensor = torch.cat([tensor, torch.full((pad_size,), float('nan'), device=tensor.device)])
        else:
            padded_tensor = tensor

        # Gather padded tensors
        padded_tensors = [torch.empty_like(padded_tensor) for _ in range(self.world_size)]
        dist.all_gather(padded_tensors, padded_tensor)

        # Remove padding and combine
        gathered = []
        for t, s in zip(padded_tensors, sizes):
            gathered.append(t[: s.item()])
        return torch.cat(gathered)

    def _gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gathers a tensor from all ranks."""
        if not dist.is_initialized():
            return tensor  # Return local tensor if not using distributed training

        output_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors)

    def _is_zero3_enabled(self) -> bool:
        """Check if ZeRO-3 is enabled."""
        ds_config = self.config['deepspeed']
        return (
            'zero_optimization' in ds_config
            and 'stage' in ds_config['zero_optimization']
            and ds_config['zero_optimization']['stage'] == 3
        )

    def _prepare_model_inputs(self, input_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare model inputs and attention mask."""
        attention_mask = (input_tokens != self.pad_token_id).bool()
        return input_tokens.to(self.device), attention_mask.to(self.device)

    def _get_grad_norm(self, engine: deepspeed.DeepSpeedEngine) -> float:
        """Compute the norm of the model's gradients."""
        with torch.no_grad():
            total = 0.0
            for param in engine.module.parameters():
                should_gather = hasattr(param, 'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                    total += float(param.float().norm())

        return total

    def _get_model_state_dict(self, engine: deepspeed.DeepSpeedEngine) -> Dict[str, torch.Tensor]:
        """Retrieves the model state_dict from the engine."""
        if self._is_zero3_model(engine):  # check for zero3
            full_state_dict = engine._zero3_consolidated_16bit_state_dict()
        else:
            full_state_dict = {k: v.cpu() for k, v in engine.module.state_dict().items()}

        dist.barrier()
        return full_state_dict

    # def _create_checkpoint(
    #     self,
    #     engine: deepspeed.DeepSpeedEngine,
    #     save_dir: str,
    #     tag: Optional[str] = None,
    #     keep_last_n: Optional[int] = 3,
    # ) -> None:
    #     """Saves checkpoint using DeepSpeed engine."""

    #     self.logger.info(f"Saving checkpoint to {save_dir!r}")

    #     # Use DeepSpeed's save checkpoint if enabled
    #     engine.save_checkpoint(save_dir=save_dir, tag=tag)

    #     # _ = engine.save_16bit_model(save_dir)

    #     if keep_last_n > 0 and self.local_rank == 0:
    #         # Cleanup old checkpoints, keeping only N most recent
    #         cleanup_old_checkpoints(save_dir, keep_last_n)

    def _save_hf_model(
        self,
        engine: deepspeed.DeepSpeedEngine,
        save_base_dir: str,
        step_count: int,
        tag: Optional[str] = None,
        keep_last_n: Optional[int] = 0,
    ) -> None:
        """Saves the model's weights to the specified directory using HF 'save_pretrained'."""

        assert save_base_dir, 'Save directory must be specified.'

        is_zero3 = self._is_zero3_model(engine)
        model: PreTrainedModel = engine.module

        # Prepare the checkpoint directory name
        if tag:
            ckpt_dir = os.path.join(save_base_dir, f"checkpoint_step_{step_count}_{tag}")
        else:
            ckpt_dir = os.path.join(save_base_dir, f"checkpoint_step_{step_count}")

        # Create directory on main process only
        if self.local_rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        dist.barrier()

        if is_zero3:
            with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
                if self.local_rank == 0:
                    model.save_pretrained(ckpt_dir, save_peft_format=False)
                    self.logger.info(f"Model saved to {ckpt_dir}")
        else:
            if self.local_rank == 0:
                model.save_pretrained(ckpt_dir, save_peft_format=False)
                self.logger.info(f"Model saved to {ckpt_dir}")

        dist.barrier()

        # Cleanup old checkpoints only on main process
        if keep_last_n > 0 and self.local_rank == 0:
            cleanup_old_checkpoints(save_base_dir, keep_last_n)

    def _is_zero3_model(self, engine: deepspeed.DeepSpeedEngine) -> bool:
        """Check if the model is ZeRO-3 partitioned."""
        return engine.zero_optimization_partition_weights()

    def _compute_dynamic_discount(
        self, episode_length: int, max_length: int = 10000, min_discount: float = 0.999, max_disount: float = 0.9999
    ) -> float:
        """Compute dynamic discount factor."""
        assert episode_length > 0, 'Episode length must be greater than 0.'
        assert max_length > 1000, 'Max length must be greater than 1000.'
        assert 0.0 < min_discount < 1.0, 'Min discount must be in the range (0, 1).'
        assert 0.0 < max_disount < 1.0, 'Max discount must be in the range (0, 1).'
        assert min_discount < max_disount, 'Min discount must be less than max discount.'
        scaled_length = min(episode_length / max_length, 1.0)
        gamma = min_discount + (max_disount - min_discount) * scaled_length
        return gamma

    def _get_lr_by_group_name(self, name: str) -> float:
        """Get learning rate for a parameter group by name."""
        if self._is_rank0():
            for group in self.policy_engine.optimizer.param_groups:
                if group['name'] == name:
                    return group['lr']
            return self.policy_engine.optimizer.param_groups[0]['lr']
        return 0.0

    def _get_common_stats(self) -> Dict:
        """Get common statistics like learning rates."""
        if self._is_rank0():
            return {
                'policy/learning_rate': self._get_lr_by_group_name('policy'),
                'value/learning_rate': self._get_lr_by_group_name('value'),
            }
        return {}

    def _log_batch_stats(self, batch_stats: Dict[str, Any]):
        """Log batch statistics."""
        if self._is_rank0() and self.tracker:
            batch_stats.update(self._get_common_stats())
            self.tracker.log_learner_step_stats(batch_stats)
        dist.barrier()

    def _log_iteration_stats(self, iter_stats: Dict[str, Any]):
        """Log iteration statistics."""
        if self._is_rank0() and self.tracker:
            iter_stats.update(self._get_common_stats())
            self.logger.info(f"Learner stats: {iter_stats}")
            self.tracker.log_learner_iteration_stats(iter_stats)
        dist.barrier()
