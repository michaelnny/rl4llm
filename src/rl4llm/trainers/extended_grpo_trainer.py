"""Implements Extended GRPO trainer"""

import math
from typing import List, Optional

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from rl4llm.core.base_env import EpisodeData
from rl4llm.trainers.grpo_trainer import GRPOConfig, GRPOTrainer, TransitionData


class ExtendedGRPOConfig(GRPOConfig):
    """GRPO config instance for RL LLM"""

    filter_low_reward_std: Optional[bool] = Field(
        True,
        description='Skip using group samples with low reward std for training',
    )

    gamma_min: Optional[float] = Field(
        0.998,
        ge=0.0,
        le=1.0,
        description='Minimum discount',
    )
    gamma_max: Optional[float] = Field(
        0.9998,
        ge=0.0,
        le=1.0,
        description='Maximum discount',
    )
    clip_eps_min: Optional[float] = Field(
        0.998,
        ge=0.0,
        le=1.0,
        description='Minimum clip epsilon for PPO PG',
    )
    clip_eps_max: Optional[float] = Field(
        0.9998,
        ge=0.0,
        le=1.0,
        description='Maximum clip epsilon for PPO PG',
    )

    # enhancements to encourage exploration
    min_temperature: Optional[float] = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description='Minimum sampling temperature for group generation',
    )
    max_temperature: Optional[float] = Field(
        1.2,
        gt=0.0,
        le=2.0,
        description='Maximum sampling temperature for group generation',
    )
    min_top_p: Optional[float] = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description='Minimum sampling top p for group generation',
    )
    max_top_p: Optional[float] = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description='Maximum sampling top p for group generation',
    )
    explore_eps_max: Optional[float] = Field(
        0.0, ge=0.0, le=1.0, description='Initial exploration epsilon'
    )
    explore_eps_min: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description='Minimum exploration epsilon after decay',
    )
    explore_percentage: Optional[float] = Field(
        0, ge=0, le=1.0, description='Exploration percentage'
    )
    random_start_steps: Optional[int] = Field(
        0, ge=0, le=30, description='Random start steps to do exploration'
    )
    random_start_top_k: Optional[int] = Field(
        0, ge=0, le=500, description='Explore start top-k'
    )
    replace_top_k: Optional[int] = Field(
        10,
        ge=1,
        le=20,
        description='Check for special source token during token replacement',
    )
    replace_max_count: Optional[int] = Field(
        0,
        ge=0,
        le=10,
        description='Maximum number of continue generation by adding the special token and continue generation',
    )
    replace_prob: Optional[float] = Field(
        0,
        ge=0,
        le=1.0,
        description='Probability to continue generation',
    )

    @model_validator(mode='after')
    def check_discounts_and_clips(cls, values):
        if values.gamma_min >= values.gamma_max:
            raise ValueError('gamma_min must be lesser than gamma_max')
        if values.clip_eps_min >= values.clip_eps_max:
            raise ValueError('clip_eps_min must be lesser than clip_eps_max')
        return values


class ExtendedGRPOTrainer(GRPOTrainer):
    """Extended GRPO trainer for LLM"""

    def initialize_trainer(self):
        """Initialize GRPO specific settings"""

        # better type hint
        self.config: ExtendedGRPOConfig = self.config

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
        self.explore_epsilon = self.config.explore_eps_max
        self.clip_eps = self.config.clip_eps_max

    def post_step(self):
        """Handles epsilon decays"""
        super().post_step()

        self._decay_explore_epsilon()
        self._decay_clip_epsilon()

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
                'num_return_sequences': 1,  # we handle the group size inside the HfMDPEnv
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
                    'explore_epsilon': self.explore_epsilon,
                }
                outputs = self.train_env.rollout(
                    policy_model, train_sampling_params, **custom_kwargs
                )
                self.logger.log_scalar(
                    'other/explore_epsilon', self.explore_epsilon
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
            terminal_rewards = torch.concat(
                [torch.tensor([ep.terminal_reward]) for ep in episodes]
            ).to(self.torch_dtype)
            if (
                torch.std(terminal_rewards, unbiased=False)
                <= self.group_reward_std_threshold
            ):
                self.logger.debug(
                    f"Skipping group samples with rewards of low std, minimum group reward std: {self.group_reward_std_threshold:.4f}"
                )
                self.logger.log_scalar(
                    'other/skipped_sample_count', len(terminal_rewards)
                )
                return False

        return True

    def _decay_explore_epsilon(self):
        """
        Computes exploration epsilon using a cosine decay schedule based on the current iteration step.
        """
        if self.config.explore_percentage == 0:
            self.explore_epsilon = 0.0
            return

        max_random_start_steps = (
            self.config.max_steps * self.config.explore_percentage
        )
        if self.global_step >= max_random_start_steps:
            self.explore_epsilon = self.config.explore_eps_min
            return

        # Calculate progress for cosine decay
        progress = self.global_step / max_random_start_steps
        cosine_decay = (
            0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
        )

        # Compute epsilon with cosine decay
        self.explore_epsilon = (
            self.config.explore_eps_min
            + (self.config.explore_eps_max - self.config.explore_eps_min)
            * cosine_decay
        )

    def _decay_clip_epsilon(self):
        """Compute PG clip epsilon based on the current iteration step count."""
        if self.global_step >= self.config.max_steps:
            self.clip_eps = self.config.clip_eps_min
        else:
            # Linear decay schedule
            progress = self.global_step / self.config.max_steps
            self.clip_eps = (
                self.config.clip_eps_max
                - (self.config.clip_eps_max - self.config.clip_eps_min)
                * progress
            )

    def _compute_episode_returns(
        self, rewards: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes returns for the episode sequence."""

        episode_length = loss_mask.sum().item()
        gamma = self.get_dynamic_discount(
            episode_length, self.config.gamma_min, self.config.gamma_max
        )

        return self.masked_monte_carlo_returns(rewards, loss_mask, gamma)

    @staticmethod
    def get_dynamic_discount(
        episode_length: int,
        gamma_min=0.998,
        gamma_max=0.9998,
        max_expected_length=8192,
    ) -> float:
        """Computes dynamic discount based on the episode length and min/max discount."""
        normalized_length = min(episode_length / max_expected_length, 1.0)
        gamma = gamma_max - (gamma_max - gamma_min) * normalized_length
        return gamma
