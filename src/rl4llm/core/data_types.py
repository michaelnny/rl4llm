from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator


class BasicTrainConfig(BaseModel):
    """Basic Training Config"""

    seed: int = Field(167, ge=1, description='Runtime seed')
    checkpoint_interval: int = Field(0, ge=0, le=100, description='Interval to save policy model checkpoint')
    artifacts_path: str = Field(None, description='Path to save artifacts like checkpoints, tensorboard logs')

    class Config:
        arbitrary_types_allowed = True


class GRPOConfig(BasicTrainConfig):
    """GRPO Training Configuration"""

    """For RL sample generation"""
    system_prompt: Optional[str] = Field(None, description='System prompt for generation')
    max_new_tokens: Optional[int] = Field(4096, ge=100, description='Maximum number of new tokens to generate')
    temperature: Optional[float] = Field(0.9, gt=0.0, le=1.0, description='Sampling temperature for generation')
    top_k: Optional[int] = Field(0, ge=0, le=50000, description='Sampling top-k for generation')
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description='Sampling top-p for generation')
    do_sample: Optional[bool] = Field(True, description='Enable sampling for generation')
    group_size: int = Field(8, ge=4, le=256, description='Number of group outcomes for single question')

    # our enhancements to GRPO to encourage exploration
    group_temperature: Optional[bool] = Field(False, description='Use group temperatures to sample tokens during generation')
    explore_init_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Initial exploration epsilon')
    explore_min_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Minimum exploration epsilon after decay')
    explore_decay_steps: Optional[int] = Field(0, ge=0, le=1000000, description='Exploration epsilon decay steps')
    random_start_steps: Optional[int] = Field(
        10, ge=0, le=128, description='Number of random start steps to randomly sample tokens'
    )
    random_start_top_k: Optional[int] = Field(20, ge=10, le=200, description='Number of top-k to sample during random start')

    """For RL GRPO training"""
    max_iterations: int = Field(10000, ge=1, description='How long to run the training')
    rollout_size: int = Field(1024, ge=1, le=5120, description='Number of samples to collect before update policy')
    num_updates: int = Field(1, ge=1, le=4, description='GRPO update epochs for a collection of samples')
    batch_size: int = Field(1, ge=1, le=1024, description='Mini-batch size')
    gradient_accumulate_steps: int = Field(1, ge=1, le=32, description='Gradient accumulation steps')
    clip_eps: float = Field(0.2, ge=0.0, le=1.0, description='PPO policy loss clip epsilon')
    gamma: float = Field(1.0, ge=0.0, le=1.0, description='Fallback default discount factor for compute returns')
    zero_based_reward: bool = Field(False, description='Use 0 for correct answer, -1 for incorrect answer')
    normalize_group_rewards: bool = Field(True, description='Normalized rewards for the group outcomes')
    normalize_advantages: bool = Field(False, description='Normalized advantages before compute PG loss')
    kl_loss_coef: float = Field(0.01, ge=0.0, le=1.0, description='KL penalty loss coefficient')
    sync_reference_interval: int = Field(
        0, ge=10, le=1000, description='Interval to update reference model using latest policy'
    )


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
    reward: Optional[torch.Tensor] = Field(..., description='A scalar reward corresponding to terminal time step t=T')

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
