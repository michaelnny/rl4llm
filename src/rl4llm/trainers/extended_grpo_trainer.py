"""Implements Extended GRPO trainer"""

import math
from typing import List, Optional

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from rl4llm.core.base_trainer import BaseRLConfig
from rl4llm.envs import EpisodeData
from rl4llm.trainers.grpo_trainer import GRPOTrainer, TransitionData


class ExtendedGRPOConfig(BaseRLConfig):
    """GRPO config instance for RL LLM"""

    filter_low_reward_std: Optional[bool] = Field(
        True,
        description='Skip using group samples with low reward std for training',
    )

    # enhancements to encourage exploration
    group_temperature: Optional[bool] = Field(
        False,
        description='Use group temperatures to sample tokens during generation',
    )
    min_temperature: Optional[float] = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description='Minimum sampling temperature for group temperature',
    )
    max_temperature: Optional[float] = Field(
        1.2,
        gt=0.0,
        le=2.0,
        description='Maximum sampling temperature for group temperature',
    )
    explore_init_epsilon: Optional[float] = Field(
        0.0, ge=0.0, le=1.0, description='Initial exploration epsilon'
    )
    explore_min_epsilon: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description='Minimum exploration epsilon after decay',
    )
    explore_decay_steps: Optional[int] = Field(
        0, ge=0, le=1000000, description='Exploration epsilon decay steps'
    )
    explore_steps: Optional[int] = Field(
        0, ge=0, le=30, description='Random start steps to do exploration'
    )
    explore_top_k: Optional[int] = Field(
        0, ge=0, le=500, description='Explore start top-k'
    )
    explore_decay: Optional[float] = Field(
        0.8, gt=0, le=1, description='Rate to decay explore top-k'
    )
    continue_max_retry: Optional[int] = Field(
        0,
        ge=0,
        le=10,
        description='Maximum number of continue generation by adding the special token and continue generation',
    )
    continue_prob: Optional[float] = Field(
        0,
        ge=0,
        le=1.0,
        description='Probability to continue generation',
    )


class ExtendedGRPOTrainer(GRPOTrainer):
    """Extended GRPO trainer for LLM"""

    def initialize_trainer(self):
        """Initialize GRPO specific settings"""

        # avoid adding group of samples with almost identical outcomes
        _dummy_rewards = torch.tensor(
            [0] * self.config.group_size, dtype=torch.float32
        )
        _idx = math.ceil(self.config.group_size * 0.05)
        _dummy_rewards[:_idx] = 1.0
        self.group_reward_std_threshold = torch.std(
            _dummy_rewards, unbiased=False
        )

        # Controls exploration
        self.explore_epsilon = 0.0

    @torch.inference_mode()
    def generate_experience(self) -> List[EpisodeData]:
        """Generates samples using the current policy."""

        if self.is_inference_engine_enabled():
            train_sampling_params = {
                'max_new_tokens': self.config.max_completion_tokens,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'repetition_penalty': self.config.repetition_penalty,
            }
        else:
            train_sampling_params = {
                'max_new_tokens': self.config.max_completion_tokens,
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'repetition_penalty': self.config.repetition_penalty,
                'num_return_sequences': 1,  # we handle the group size inside the LocalLLMEnv
                'do_sample': True,
            }

        # we always use batch size of 1 during training roll out
        local_rollout_size = (
            self.config.train_rollout_size // self.dist_ops.world_size
        )
        collected_episodes: List[List[EpisodeData]] = []
        local_count = 0

        with self.unwrapped_model_for_generation() as policy_model:
            while local_count < local_rollout_size:
                # control explore logit
                custom_kwargs = {
                    'explore_probability': self._get_exploration_epsilon(),
                }
                outputs = self.train_env.rollout(
                    policy_model, train_sampling_params, **custom_kwargs
                )
                self.logger.log_scalar(
                    'other/exploration_epsilon', self.explore_epsilon
                )
                if self._check_group_episodes(outputs):
                    # IMPORTANT: do not flatten the episodes yet
                    # as we need to normalize the rewards on group level
                    collected_episodes.extend([outputs])
                    local_count += len(outputs)

                    self.log_batch_episodes(
                        self._train_phase, outputs, self.global_step
                    )

        return collected_episodes

    def _check_group_episodes(self, episodes: List[EpisodeData]) -> bool:
        """Checks if the group of episode is valid for training"""
        if not episodes:
            return False
        if len(episodes) < 4:
            return False

        if self.config.filter_low_reward_std:
            # Discard samples with rewards of low std, as they leads to zero advantages -> zero gradients
            rewards = self.transform_batch_rewards(episodes).cpu()
            if (
                torch.std(rewards, unbiased=False)
                <= self.group_reward_std_threshold
            ):
                self.logger.debug(
                    f"Skipping group samples with rewards of low std, minimum group reward std: {self.group_reward_std_threshold:.4f}"
                )
                self.logger.log_scalar(
                    'other/skipped_sample_count', len(rewards)
                )
                return False

        return True

    def _get_exploration_epsilon(self) -> float:
        """Computes exploration epsilon based on the current iteration step count."""
        if self.config.explore_decay_steps == 0:
            self.explore_epsilon = 0.0
        elif self.global_step >= self.config.explore_decay_steps:
            self.explore_epsilon = self.config.explore_min_epsilon
        else:
            # Cosine decay schedule
            progress = self.global_step / self.config.explore_decay_steps
            cosine_decay = (
                0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
            )
            self.explore_epsilon = (
                self.config.explore_min_epsilon
                + (
                    self.config.explore_init_epsilon
                    - self.config.explore_min_epsilon
                )
                * cosine_decay
            )

        return self.explore_epsilon
