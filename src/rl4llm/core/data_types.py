from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator


class GRPOConfig(BaseModel):
    """GRPO Training Configuration"""

    """For RL sample generation"""
    system_prompt: Optional[str] = Field(None, description='System prompt for generation')
    max_new_tokens: Optional[int] = Field(4096, ge=50, description='Maximum number of new tokens to generate')
    temperature: Optional[float] = Field(0.9, gt=0.0, le=1.0, description='Sampling temperature for generation')
    top_k: Optional[int] = Field(100, ge=0, le=50000, description='Sampling top-k for generation')
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description='Sampling top-p for generation')
    group_size: int = Field(8, ge=4, le=256, description='Number of group outcomes for single question')
    min_completion_length: Optional[int] = Field(
        100, ge=10, le=1000, description='Minimum completion token length for compute reward'
    )
    xml_format: Optional[bool] = Field(False, description='Check R1 style XML format for compute reward')

    # enhancements to encourage exploration
    group_temperature: Optional[bool] = Field(False, description='Use group temperatures to sample tokens during generation')
    explore_init_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Initial exploration epsilon')
    explore_min_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Minimum exploration epsilon after decay')
    explore_decay_steps: Optional[int] = Field(0, ge=0, le=1000000, description='Exploration epsilon decay steps')
    explore_start_ratio: Optional[float] = Field(0, ge=0, le=1.0, description='Ratio of random start steps to do exploration')
    explore_top_k: Optional[int] = Field(50, ge=10, le=200, description='Unified top-k for both exploration')
    explore_top_k_beta: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description='Square root of probabilities during exploration'
    )

    """For RL GRPO training"""
    max_steps: int = Field(10000, ge=1, description='How long to run the training')
    rollout_size: int = Field(1024, ge=1, le=5120, description='Number of samples to collect before update policy')
    num_updates: int = Field(1, ge=1, le=4, description='GRPO update epochs for a collection of samples')
    batch_size: int = Field(4, ge=1, le=1024, description='Mini-batch size for training')
    gradient_accumulate_steps: int = Field(1, ge=1, description='Gradient accumulation steps')
    clip_eps: float = Field(0.2, ge=0.0, le=1.0, description='PPO policy loss clip epsilon')
    gamma: float = Field(1.0, ge=0.0, le=1.0, description='Default discount factor for compute returns')
    normalize_group_rewards: bool = Field(True, description='Normalized rewards for the group outcomes')
    normalize_advantages: bool = Field(False, description='Normalized advantages before compute PG loss')
    kl_loss_coef: float = Field(0.01, ge=0.0, le=1.0, description='KL penalty loss coefficient')

    # enhancement of dynamic discount based on sequence length
    dynamic_discount: bool = Field(False, description='Use dynamic discount based on sequence length')
    min_gamma: float = Field(0.999, ge=0.0, le=1.0, description='Min value of dynamic discount for compute returns')
    max_gamma: float = Field(0.9999, ge=0.0, le=1.0, description='Max value of dynamic discount for compute returns')
    max_completion_length: int = Field(
        8192, ge=1024, le=51200, description='Maximum to scale the dynamic discount compute returns'
    )

    sync_reference_interval: int = Field(
        0, ge=10, le=1000, description='Interval to update reference model using latest policy'
    )
    checkpoint_interval: int = Field(0, ge=0, le=100, description='Interval to save policy model checkpoint')
    eval_interval: int = Field(100, ge=1, description='Interval to evaluate policy model')
    eval_batch_size: int = Field(8, ge=1, le=1024, description='Mini-batch size for evaluation')

    class Config:
        arbitrary_types_allowed = True


class GRPOSample(BaseModel):
    """GPPO transition for training"""

    states: torch.LongTensor = Field(..., description='A long tensor for token sequences from t=0, 1, ..., T-1')
    actions: torch.LongTensor = Field(..., description='A long tensor for token sequences from t=1, 2, ..., T-1, T')
    loss_mask: torch.BoolTensor = Field(
        ...,
        description='A boolean tensor (0s user tokens, 1s assistant tokens) corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    pi_logprobs: torch.Tensor = Field(
        ..., description='A float tensor for action logprobs corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    ref_logprobs: torch.Tensor = Field(
        ...,
        description='A float tensor for action logprobs from reference model corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    advantages: torch.Tensor = Field(
        ..., description='A float tensor for GAE advantages estimate corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    reward: torch.Tensor = Field(..., description='A scalar reward (not normalized) corresponding to terminal time step t=T')

    @model_validator(mode='after')
    def check_tensor_shapes(cls, values):
        tensors = [
            values.states,
            values.actions,
            values.loss_mask,
            values.pi_logprobs,
            values.ref_logprobs,
            values.advantages,
        ]

        # Ensure all tensors are of the same shape
        tensor_shapes = [tensor.shape if isinstance(tensor, torch.Tensor) else None for tensor in tensors]

        if len(set(tensor_shapes)) > 1:
            raise ValueError(f"Tensors have mismatched shapes: {tensor_shapes}")

        return values

    class Config:
        arbitrary_types_allowed = True
