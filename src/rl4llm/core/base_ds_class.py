import logging
import os
import random
import time
import math
from contextlib import nullcontext
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig

from rl4llm.models import CustomQwen2Model
from rl4llm.envs import VectorEnvWrapper
from rl4llm.types import DecodingConfig, EnvAction, EnvState, Episode, TokenUsage
from rl4llm.utils import (
    TrainingTracker,
    load_yaml_config_file,
    save_to_json_file,
    save_yaml_config_file,
    set_seed,
    cleanup_old_checkpoints,
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
        assert dist.is_initialized(), "Distributed environment must be initialized before creating RL Agent."

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
        self.seed = self.config['job'].get('seed', 123)
        self.tokenizer: PreTrainedTokenizer = self._init_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = [self.tokenizer.eos_token, self.tokenizer.pad_token]

        set_seed(self.seed)

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
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
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
            if hasattr(model, "value_head") and model.value_head:
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
            self.logger.info("Creating inference engine...")
        tp_size = dist.get_world_size()
        ds_infer_config = {
            "tensor_parallel": {"tp_size": tp_size},
            "dtype": self.dtype,
            "replace_with_kernel_inject": True,
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
        model_parameters: Optional[List[Dict]] = None,
    ) -> deepspeed.DeepSpeedEngine:
        """Creates DeepSpeed training engine."""
        if self.logger:
            self.logger.info("Creating training engine...")

        if not model_parameters:
            model_parameters = model.parameters()

        engine: deepspeed.DeepSpeedEngine = None
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            config=self.config['deepspeed'],
            args={"local_rank": self.local_rank},
            dist_init_required=True,
        )

        # if load_ckpt_dir:
        #     _, checkpoint_state_dict = engine.load_checkpoint(load_ckpt_dir, load_ckpt_tag, load_module_only=True)

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
                if "value_head" in name:
                    value_params.append(param)
                elif any(nd in name for nd in ['norm', 'tok_embeddings']):
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
            if not values:  # Handle cases where no values were collected
                continue

            # Convert list of tensors to a single tensor
            if isinstance(values, list):
                values_tensor = torch.stack(values)
            else:
                values_tensor = values

            # Gather tensors from all ranks
            all_values = self._gather_tensor(values_tensor)

            # Compute mean across all ranks
            agg_stats[key] = torch.mean(all_values).item()

            # Compute variance if needed
            # var_keys = ['objective/kl_score', 'objective/returns'] if for_ppo else []
            if var_keys and key in var_keys:
                agg_stats[f"{key}_var"] = torch.var(all_values).item()

        if 'value/error' in agg_stats and 'objective/returns_var' in agg_stats:
            agg_stats['value/var_explained'] = 1 - agg_stats['value/error'] / agg_stats['objective/returns_var']
        if 'objective/kl_score' in agg_stats and 'objective/rewards' in agg_stats:
            agg_stats['objective/total_reward'] = agg_stats['objective/kl_score'] + agg_stats['objective/rewards']
        return agg_stats

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

    @staticmethod
    def _get_grad_norm(engine: deepspeed.DeepSpeedEngine) -> float:
        """Compute the norm of the model's gradients."""
        with torch.no_grad():
            total = 0.0
            for param in engine.module.parameters():
                should_gather = hasattr(param, 'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                    total += float(param.float().norm())

        return total

    @staticmethod
    def _get_model_state_dict(engine: deepspeed.DeepSpeedEngine) -> Dict[str, torch.Tensor]:
        """Retrieves the model state_dict from the engine."""

        # engine.eval()  # Ensure model is in eval mode when extracting weights

        if engine.zero_optimization_partition_weights():  # check for zero3
            full_state_dict = engine._zero3_consolidated_16bit_state_dict()
        else:
            full_state_dict = {k: v.cpu() for k, v in engine.module.state_dict().items()}

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

        assert save_base_dir, "Save directory must be specified."

        is_zero3 = self._is_zero3_model(engine)

        # Retrieve the model from the DeepSpeed engine
        model: PreTrainedModel = engine.module

        if is_zero3:
            # Gather all parameters across processes
            ctx = deepspeed.zero.GatheredParameters(model.parameters())
        else:
            ctx = nullcontext()

        with ctx:
            # Only save on the main process
            if self.local_rank == 0:
                if tag:
                    ckpt_dir = os.path.join(save_base_dir, f"checkpoint_step_{step_count}_{tag}")
                else:
                    ckpt_dir = os.path.join(save_base_dir, f"checkpoint_step_{step_count}")

                # Ensure the save directory exists
                os.makedirs(ckpt_dir, exist_ok=True)

                # Save model in Hugging Face format
                model.save_pretrained(ckpt_dir, save_peft_format=False)
                self.logger.info(f"Model saved to {ckpt_dir}")

                # Remove old checkpoints
                if keep_last_n > 0 and self.local_rank == 0:
                    cleanup_old_checkpoints(save_base_dir, keep_last_n)

    def _is_zero3_model(self, engine: deepspeed.DeepSpeedEngine) -> bool:
        """Check if the model is ZeRO-3 partitioned."""
        return engine.zero_optimization_partition_weights()
