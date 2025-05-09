"""Implements DAPO using GRPO as base trainer"""

from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from rl4llm.core.base_env import EpisodeData
from rl4llm.core.base_trainer import BaseRLConfig
from rl4llm.trainers.grpo_trainer import GRPOTrainer, TransitionData


class DAPOConfig(BaseRLConfig):
    """DAPO config instance for RL LLM"""

    clip_eps_high: float = Field(
        0.28,
        ge=0.0,
        le=1.0,
        description='PPO policy loss clip epsilon higher bound',
    )
    clip_eps_low: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description='PPO policy loss clip epsilon lower bound',
    )
    length_max: int = Field(
        10240,
        ge=10,
        le=20480,
        description='Maximum completion length for length penalty reward',
    )
    length_cache: int = Field(
        1024,
        ge=10,
        le=10240,
        description='Maximum completion length cache buffer for length penalty reward',
    )
    num_updates: int = Field(
        1,
        ge=1,
        le=5,
        description='PPO policy update epochs for a collection of samples',
    )

    @model_validator(mode='after')
    def check_lengths(cls, model_instance):
        if model_instance.length_cache >= model_instance.length_max:
            raise ValueError(
                f"Cache length should be lesser than  maximum length: "
                f"length_max={model_instance.length_max}, length_cache={model_instance.length_cache}"
            )
        return model_instance


class DAPOTrainer(GRPOTrainer):
    """DAPO using GRPO base trainer for LLM

    DAPO paper:
    https://arxiv.org/abs/2503.14476

    """

    def compute_loss(
        self, pi_logits: torch.Tensor, experience_batch: TransitionData
    ) -> torch.Tensor:
        """Compute GRPO loss for a single training batch

        Args:
            pi_logits (torch.Tensor): Raw logits of actions computed using
                current policy, shape [batch_size, seq_len]
            experience_batch (TransitionData): A batch of samples collected
                during generation
        Returns:
            torch.Tensor: The total loss tensor
        """
        behavior_logprobs = experience_batch.pi_logprobs.to(self.device)
        actions = experience_batch.actions.to(self.device)
        advantages = experience_batch.advantages.to(self.device)
        loss_mask = experience_batch.loss_mask.to(self.device)

        advantages = advantages.float()
        behavior_logprobs = behavior_logprobs.float()
        pi_logits = pi_logits.float()

        assert pi_logits.dtype == behavior_logprobs.dtype == advantages.dtype

        if self.config.normalize_advantages:
            advantages = self.dist_masked_whiten(advantages, loss_mask, dim=1)

        # PPO clipped surrogate PG loss
        pi_logprobs = self.compute_logprobs_from_logits(pi_logits, actions)
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(
            1 - self.config.clip_eps_low, 1 + self.config.clip_eps_high
        )
        pg_losses1 = ratio * advantages.detach()
        pg_losses2 = clipped_ratio * advantages.detach()
        pg_losses = -torch.min(pg_losses1, pg_losses2)

        with torch.no_grad():
            approxkl = (
                0.5
                * self.masked_mean(
                    torch.square(pi_logprobs - behavior_logprobs),
                    loss_mask,
                    dim=1,
                ).mean()
            )
            clipfrac = self.masked_mean(
                torch.lt(pg_losses2, pg_losses1), loss_mask, dim=1
            ).mean()

        # Token-level PG loss
        pg_loss = self.masked_sum(pg_losses, loss_mask) / loss_mask.sum()

        # Compute entropy for the policy
        entropies = self.compute_entropy_from_logits(
            logits=pi_logits, loss_mask=loss_mask
        )
        entropy = entropies.mean()
        entropy_loss = -(self.config.entropy_loss_coef * entropy)

        self.logger.log_scalar('train/pg_loss', pg_loss.detach().item())
        self.logger.log_scalar(
            'train/entropy_loss', entropy_loss.detach().item()
        )
        self.logger.log_scalar('policy/entropy', entropy.detach().item())
        self.logger.log_scalar('policy/approxkl', approxkl.detach().item())
        self.logger.log_scalar('policy/clipfrac', clipfrac.detach().item())

        loss = pg_loss + entropy_loss
        # Compute KL divergence if coefficient is positive
        if self.config.kl_loss_coef > 0:
            # We add the kl as auxiliary loss instead of mixing pre-token KL to the rewards
            ref_logprobs = experience_batch.ref_logprobs.to(self.device)
            # Compute the KL divergence between the model and the reference model
            per_token_kl = (
                torch.exp(ref_logprobs - pi_logprobs)
                - (ref_logprobs - pi_logprobs)
                - 1
            )

            kl = self.masked_mean(per_token_kl, loss_mask, dim=1).mean()
            kl_loss = self.config.kl_loss_coef * kl

            loss += kl_loss
            self.logger.log_scalar('train/kl_loss', kl_loss.detach().item())
            self.logger.log_scalar('objective/kl', kl.detach().item())

        return loss

    def _check_group_episodes(self, episodes: List[EpisodeData]) -> bool:
        """Checks if the group of episode is valid for training"""
        if not episodes:
            return False
        if len(episodes) < self.config.group_size:
            return False

        # Discard samples with identical rewards, as they leads to zero advantages -> zero gradients
        terminal_rewards_list = []
        for ep in episodes:
            # Assuming ep.terminal_reward is a Python scalar (int/float).
            reward_tensor = torch.tensor(
                [ep.terminal_reward], dtype=self.torch_dtype, device=self.device
            )
            terminal_rewards_list.append(reward_tensor)

        terminal_rewards = torch.concat(terminal_rewards_list)

        all_rewards_identical = (
            (terminal_rewards == terminal_rewards[0]).all().item()
        )
        if all_rewards_identical:
            identical_value_str = (
                str(terminal_rewards[0].item())
                if terminal_rewards.numel() > 0
                else 'N/A'
            )
            self.logger.debug(
                f"Skipping group samples with identical rewards: {identical_value_str}"
            )
            # tips: using metric name ends with '_count' will automatically sums over
            self.logger.log_scalar(
                'other/skipped_samples_count', terminal_rewards.numel()
            )
            return False

        return True
