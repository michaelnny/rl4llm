from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator


class GRPOConfig(BaseModel):
    """GRPO Training Configuration"""

    """For RL sample generation"""
    system_prompt: Optional[str] = Field(None, description='System prompt for generation')
    max_prompt_length: Optional[int] = Field(
        1024, ge=256, le=10240, description='Skip sample with prompt length greater than this to avoid peak memory spikes'
    )
    max_new_tokens: Optional[int] = Field(4096, ge=50, description='Maximum number of new tokens to generate')
    temperature: Optional[float] = Field(0.9, gt=0.0, le=1.0, description='Sampling temperature for generation')
    min_temperature: Optional[float] = Field(
        0.6, gt=0.0, le=1.0, description='Minimum sampling temperature for group temperature'
    )
    max_temperature: Optional[float] = Field(
        1.2, gt=0.0, le=2.0, description='Maximum sampling temperature for group temperature'
    )
    repetition_penalty: Optional[float] = Field(1.0, gt=0.0, le=2.0, description='Repetition penalty for generation')
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description='Sampling top-p for generation')
    top_k: Optional[int] = Field(50, ge=-1, le=1000, description='Sampling top-k for generation')
    group_size: int = Field(8, ge=4, le=256, description='Number of group outcomes for single question')
    xml_format: Optional[bool] = Field(False, description='Check R1 style XML format for compute reward')

    # enhancements to encourage exploration
    group_temperature: Optional[bool] = Field(False, description='Use group temperatures to sample tokens during generation')
    explore_init_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Initial exploration epsilon')
    explore_min_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Minimum exploration epsilon after decay')
    explore_decay_steps: Optional[int] = Field(0, ge=0, le=1000000, description='Exploration epsilon decay steps')
    explore_start_steps: Optional[int] = Field(0, ge=0, le=30, description='Random start steps to do exploration')
    explore_top_k: Optional[int] = Field(50, ge=10, le=500, description='Unified top-k for both exploration')
    explore_replace_prob: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description='Probabilities to replace end think token with "Wait" during exploration'
    )
    explore_max_replacements: Optional[int] = Field(
        3, ge=1, le=10, description='Maximum number of token replacements to the same sequence during exploration'
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
    clip_grad_norm: Optional[float] = Field(0.0, ge=0.0, le=10.0, description='Clip L2 gradient norm')

    # enhancement of dynamic discount based on sequence length
    dynamic_discount: bool = Field(False, description='Use dynamic discount based on sequence length')
    min_gamma: float = Field(0.999, ge=0.0, le=1.0, description='Min value of dynamic discount for compute returns')
    max_gamma: float = Field(0.9999, ge=0.0, le=1.0, description='Max value of dynamic discount for compute returns')
    max_completion_length: int = Field(
        1000, ge=500, le=51200, description='Maximum to scale the dynamic discount compute returns'
    )

    sync_reference_interval: int = Field(0, ge=0, le=1000, description='Interval to update reference model using latest policy')
    checkpoint_interval: int = Field(0, ge=0, le=1000, description='Interval to save policy model checkpoint')
    eval_interval: int = Field(100, ge=0, description='Interval to evaluate policy model')
    eval_batch_size: int = Field(8, ge=0, le=1024, description='Mini-batch size for evaluation')

    @model_validator(mode='after')
    def check_temperatures(cls, values):
        min_temp = values.min_temperature
        max_temp = values.max_temperature
        if min_temp is not None and max_temp is not None and min_temp >= max_temp:
            raise ValueError(f"min_temperature ({min_temp}) must be less than max_temperature ({max_temp})")
        return values

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
    # reward: torch.Tensor = Field(..., description='A scalar reward (not normalized) corresponding to terminal time step t=T')

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


class SampleLog(BaseModel):
    """Pydantic model for sample logging data."""

    question: str
    task_type: str
    ground_truth: str
    completion: str
    accuracy_reward: Optional[float] = 0.0
    format_reward: Optional[float] = 0.0
    total_reward: Optional[float] = 0.0
    completion_length: Optional[int] = 0
    step: Optional[int] = 0
