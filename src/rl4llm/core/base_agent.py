import logging
import os
import random
import time
import math
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


# def get_params_groups(policy_model: torch.nn.Module, optimizer_config: Dict[str, Any]) -> List[Dict]:

#     opt_params = optimizer_config['params']

#     # Check required optimizer parameters
#     required_keys = ['betas', 'eps', 'weight_decay']
#     missing_keys = [key for key in required_keys if key not in opt_params]
#     if missing_keys:
#         raise KeyError(f"Missing required keys in optimizer_config['params']: {missing_keys}")

#     weight_decay = float(opt_params['weight_decay'])

#     # Separate parameters by weight decay and model type
#     lr = float(opt_params['lr'])
#     policy_params = []
#     value_params = []
#     nodecay_params = []
#     for name, param in policy_model.named_parameters():
#         if param.requires_grad:
#             if any(nd in name for nd in ['norm', 'tok_embeddings']):
#                 nodecay_params.append(param)
#             elif 'value_output' in name or "value_head" in name:
#                 value_params.append(param)
#             else:
#                 policy_params.append(param)

#     # order is important here
#     param_groups = [
#         {'params': nodecay_params, 'lr': lr, 'weight_decay': 0.0, 'name': 'policy_nodecay'},
#         {'params': policy_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'policy'},
#         {'params': value_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'value'},
#     ]

#     return param_groups


class BaseAgent:
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
        self.seed = self.config['job'].get('seed', 123)
        self.tokenizer: PreTrainedTokenizer = self._init_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = [self.tokenizer.eos_token]

        set_seed(self.seed)
        # self.model: PreTrainedModel = self._load_model()  # Placeholder, to be defined in subclass
        # self.engine: Any = None  # Placeholder for DeepSpeed engine

    # def _load_model(self) -> PreTrainedModel:
    #     """Loads the base language model. To be implemented by subclasses."""
    #     raise NotImplementedError

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

    def _create_deepspeed_inference_engine(self, model) -> deepspeed.InferenceEngine:
        """Creates DeepSpeed inference engine."""
        if self.logger:
            self.logger.info("Creating inference engine...")
        tp_size = dist.get_world_size()
        ds_infer_config = {
            "tensor_parallel": {"tp_size": tp_size},
            "dtype": self.dtype,
            "replace_with_kernel_inject": True,
        }

        inference_engine = deepspeed.init_inference(
            model=model,
            config=ds_infer_config,
            base_dir="/dev/shm",
            checkpoint=None,
        )
        return inference_engine

    def _create_deepspeed_training_engine(
        self, model: PreTrainedModel, model_parameters: Optional[List[Dict]]
    ) -> deepspeed.DeepSpeedEngine:
        """Creates DeepSpeed training engine."""
        if self.logger:
            self.logger.info("Creating training engine...")

        if not model_parameters:
            model_parameters = model.parameters()

        policy_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            config=self.config['deepspeed'],
            args={"local_rank": self.local_rank},
            dist_init_required=True,
        )
        return policy_engine

    def aggregate_stats(
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
            all_values = self.gather_tensor(values_tensor)

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

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gathers a tensor from all ranks."""
        if not dist.is_initialized():
            return tensor  # Return local tensor if not using distributed training

        output_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors)

    @staticmethod
    def get_grad_norm(engine: deepspeed.DeepSpeedEngine) -> float:
        """Compute the norm of the model's gradients."""
        with torch.no_grad():
            total = 0.0
            for param in engine.module.parameters():
                should_gather = hasattr(param, 'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                    total += float(param.float().norm())

        return total

    def get_model_state_dict(self, ds_engine: deepspeed.DeepSpeedEngine) -> Dict[str, torch.Tensor]:
        """Retrieves the model state_dict from the engine. To be implemented by subclasses."""

        ds_engine.eval()  # Ensure model is in eval mode when extracting weights

        if ds_engine.zero_optimization_partition_weights():  # check for zero3
            full_state_dict = ds_engine._zero3_consolidated_16bit_state_dict()
        else:
            full_state_dict = ds_engine.module.state_dict()

        return full_state_dict

    def save_checkpoint(
        self,
        ds_engine: deepspeed.DeepSpeedEngine,
        save_path: str,
        keep_last_n: int = 3,
    ):
        """Saves checkpoint using DeepSpeed engine. To be implemented by subclasses."""

        self.logger.info(f"Saving checkpoint to {save_path!r}")

        # Use DeepSpeed's save checkpoint if enabled
        ds_engine.save_checkpoint(save_path)

        if keep_last_n > 0 and self.local_rank == 0:
            # Cleanup old checkpoints, keeping only N most recent
            cleanup_old_checkpoints(save_path, keep_last_n)
