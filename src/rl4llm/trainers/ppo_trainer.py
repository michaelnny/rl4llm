"""Implements PPO trainer"""

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union

import torch
from deepspeed import DeepSpeedEngine
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.core.base_trainer import (
    BaseRLConfig,
    BaseRLTrainer,
    RewardTransform,
)
from rl4llm.envs import EpisodeData, LocalLLMEnv


class PPOConfig(BaseRLConfig):
    """PPO config instance for RL LLM"""

    policy_num_updates: int = Field(
        1,
        ge=1,
        le=5,
        description='PPO policy update epochs for a collection of samples',
    )
    value_num_updates: int = Field(
        4,
        ge=1,
        le=5,
        description='PPO value update epochs for a collection of samples',
    )
    gae_lambda: float = Field(0.95, gt=0.0, le=1.0, description='GAE lambda')
    clip_eps: float = Field(
        0.2, ge=0.0, le=1.0, description='PPO policy loss clip epsilon'
    )
    value_clip_eps: float = Field(
        0.2, ge=0.0, le=1.0, description='PPO value loss clip epsilon'
    )


class TransitionData(BaseModel):
    """PPO transition for training"""

    states: torch.Tensor = Field(
        ...,
        description='A long tensor for token sequences from t=0, 1, ..., T-1',
    )
    actions: torch.Tensor = Field(
        ...,
        description='A long tensor for token sequences from t=1, 2, ..., T-1, T',
    )
    values: torch.Tensor = Field(
        ...,
        description='A tensor for value estimated for sequences from t=1, 2, ..., T-1, T',
    )
    loss_mask: torch.Tensor = Field(
        ...,
        description='A boolean tensor (0s user tokens, 1s assistant tokens) corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    pi_logprobs: torch.Tensor = Field(
        ...,
        description='A float tensor for action logprobs corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    ref_logprobs: torch.Tensor = Field(
        ...,
        description='A float tensor for action logprobs from reference model corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    returns: torch.Tensor = Field(
        ...,
        description='A float tensor for returns estimate corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    advantages: torch.Tensor = Field(
        ...,
        description='A float tensor for GAE advantages estimate corresponding to token sequences from t=1, 2, ..., T-1, T',
    )

    @model_validator(mode='after')
    def check_tensor_shapes(cls, values):
        tensors = [
            values.states,
            values.actions,
            values.values,
            values.loss_mask,
            values.pi_logprobs,
            values.ref_logprobs,
            values.returns,
            values.advantages,
        ]

        # Ensure all tensors are of the same shape
        tensor_shapes = [
            tensor.shape if isinstance(tensor, torch.Tensor) else None
            for tensor in tensors
        ]

        if len(set(tensor_shapes)) > 1:
            raise ValueError(f"Tensors have mismatched shapes: {tensor_shapes}")

        return values

    class Config:
        arbitrary_types_allowed = True


class PPOTrainer(BaseRLTrainer):
    """PPO trainer for LLM"""

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy_engine: DeepSpeedEngine,
        value_engine: DeepSpeedEngine,
        log_config: Dict[str, Any],
        train_env: LocalLLMEnv,
        eval_env: Optional[LocalLLMEnv] = None,
        ref_model: Optional[Union[PreTrainedModel, DeepSpeedEngine]] = None,
        inference_client: Optional[InferenceClient] = None,
        reward_transform_fn: Optional[RewardTransform] = None,
        seed: Optional[int] = 175,
    ):
        """Initialize the PPO trainer instance"""

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            policy_engine=policy_engine,
            log_config=log_config,
            train_env=train_env,
            eval_env=eval_env,
            ref_model=ref_model,
            value_engine=value_engine,
            inference_client=inference_client,
            reward_transform_fn=reward_transform_fn,
            seed=seed,
        )

        if config.train_rollout_size % self.dist_ops.world_size != 0:
            raise ValueError(
                'Train rollout size must be divisible by world size'
            )
        if config.eval_rollout_size % self.dist_ops.world_size != 0:
            raise ValueError(
                'Evaluation rollout size must be divisible by world size'
            )

        self.config: PPOConfig = config  # for better type hinting

    def initialize_trainer(self):
        """Initialize PPO specific settings"""
        pass

    def save_checkpoint(self, tag: str) -> None:
        """Save trained model in HF format"""
        subpath = f"epoch_{tag}"
        policy_save_path = os.path.join(
            self.checkpoint_dir, f"policy_{subpath}"
        )
        value_save_path = os.path.join(self.checkpoint_dir, f"value_{subpath}")
        self.logger.info(
            f"Saving policy model HF checkpoint to {policy_save_path}..."
        )
        self.save_weights_hf_pretrained(self.policy_engine, policy_save_path)
        self.logger.info(
            f"Saving value model HF checkpoint to {policy_save_path}..."
        )
        self.save_weights_hf_pretrained(self.value_engine, value_save_path)
        self.dist_ops.barrier()
        self.logger.info('Checkpoint saved.')

    def build_train_loader(self, experience: List[EpisodeData]) -> DataLoader:
        """Creates a train loader using the collected experiences.

        Args:
            experience (List[EpisodeData]): local rollout episodes
        Returns:
            DataLoader: A dataloader ready for training.
        """

        if not experience:
            raise ValueError('No samples for training')

        samples = []

        for start_idx in range(0, len(experience), self.config.group_size):
            end_idx = start_idx + self.config.group_size
            subsets = experience[start_idx:end_idx]
            processed = self._convert_batch_episodes_to_transitions(subsets)
            if processed:
                samples.extend(processed)

        if not samples:
            raise ValueError('No samples for training')

        local_rollout_size = (
            self.config.train_rollout_size // self.dist_ops.world_size
        )

        if len(samples) > local_rollout_size:
            samples = samples[:local_rollout_size]

        data_loader = DataLoader(
            samples,
            batch_size=self.config.train_micro_batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self._train_collate_fn,
            drop_last=True,
        )

        return data_loader

    def compute_policy_loss(
        self,
        pi_logits: torch.Tensor,
        experience_batch: TransitionData,
    ) -> torch.Tensor:
        """Compute policy loss for a single training batch

        Args:
            pi_logits (torch.Tensor): Raw logits of actions computed using
                current policy model, shape [batch_size, seq_len]
            experience_batch (TransitionData): A batch of samples collected
                during generation
        Returns:
            torch.Tensor: The total loss tensor
        """
        behavior_logprobs = experience_batch.pi_logprobs.to(self.device)
        actions = experience_batch.actions.to(self.device)
        advantages = experience_batch.advantages.to(self.device)
        loss_mask = experience_batch.loss_mask.to(self.device)

        if self.config.normalize_advantages:
            advantages = self.dist_masked_whiten(advantages, loss_mask, dim=1)

        # PPO clipped surrogate PG loss
        pi_logprobs = self.compute_logprobs_from_logits(pi_logits, actions)
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(
            1 - self.config.clip_eps, 1 + self.config.clip_eps
        )
        pg_losses1 = ratio * advantages.detach()
        pg_losses2 = clipped_ratio * advantages.detach()
        pg_losses = -torch.min(pg_losses1, pg_losses2)

        with torch.no_grad():
            approxkl = (
                0.5
                * self.dist_masked_mean(
                    torch.square(pi_logprobs - behavior_logprobs),
                    loss_mask,
                    dim=1,
                ).mean()
            )
            clipfrac = self.dist_masked_mean(
                torch.lt(pg_losses2, pg_losses1), loss_mask, dim=1
            ).mean()

        # First average over the sequence length, then average over the batch
        pg_loss = self.dist_masked_mean(pg_losses, loss_mask, dim=1).mean()

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
            # We add the kl  as an auxiliary loss instead of mixing pre-token KL to the rewards
            ref_logprobs = experience_batch.ref_logprobs.to(self.device)
            # Compute the KL divergence between the model and the reference model
            per_token_kl = (
                torch.exp(ref_logprobs - pi_logprobs)
                - (ref_logprobs - pi_logprobs)
                - 1
            )

            kl = self.dist_masked_mean(per_token_kl, loss_mask, dim=1).mean()
            kl_loss = self.config.kl_loss_coef * kl

            loss += kl_loss
            self.logger.log_scalar('train/kl_loss', kl_loss.detach().item())
            self.logger.log_scalar('objective/kl', kl.detach().item())

        return loss

    def compute_value_loss(
        self,
        pred_values: torch.Tensor,
        experience_batch: TransitionData,
    ) -> torch.Tensor:
        """Compute value loss for a single training batch

        Args:
            pred_values (torch.Tensor): Predicted state values computed using
                current value model, shape [batch_size, seq_len]
            experience_batch (TransitionData): A batch of samples collected
                during generation

        Returns:
            torch.Tensor: The total loss tensor
        """

        # values = experience_batch.values.to(self.device)
        returns = experience_batch.returns.to(self.device)
        loss_mask = experience_batch.loss_mask.to(self.device)

        # # Compute clipped value loss as in standard RLHF
        # vpred_clipped = torch.clamp(
        #     pred_values,
        #     values - self.config.value_clip_eps,
        #     values + self.config.value_clip_eps,
        # )
        # vf_losses1 = torch.square(pred_values - returns)
        # vf_losses2 = torch.square(vpred_clipped - returns)
        # losses = 0.5 * torch.max(vf_losses1, vf_losses2)
        # loss = self.dist_masked_mean(losses, loss_mask, dim=1).mean()
        # with torch.no_grad():
        #     clipfrac = self.dist_masked_mean(
        #         torch.gt(vf_losses2, vf_losses1), loss_mask, dim=1
        #     ).mean()
        # self.logger.log_scalar('value/clipfrac', clipfrac.detach().item())

        # Value loss using MC returns
        losses = 0.5 * torch.square(returns - pred_values)
        loss = self.dist_masked_mean(losses, loss_mask, dim=1).mean()

        with torch.no_grad():
            pred_error = self.dist_masked_mean(
                torch.square(pred_values.detach() - returns.detach()),
                loss_mask,
                dim=1,
            ).mean()
            returns_var = self.masked_var(returns, loss_mask, dim=1).mean()
            var_explained = (1 - pred_error / (returns_var + 1e-8)).item()

        self.logger.log_scalar('train/vf_loss', loss.detach().item())
        self.logger.log_scalar('value/error', pred_error.detach().item())
        self.logger.log_scalar('value/returns_var', returns_var.detach().item())
        self.logger.log_scalar('value/var_explained', var_explained)

        return loss

    def train_step(self, train_dataloader: DataLoader):
        """Performs the policy and value models update using collected rollout."""

        self._configure_model(self.value_engine, 'cpu', 'offload')
        self.clean_up()
        self._train_policy_step(train_dataloader)

        self._configure_model(self.policy_engine, 'cpu', 'offload')
        self.clean_up()
        self._configure_model(self.value_engine, self.device, 'reload')
        self._train_value_step(train_dataloader)

    @torch.inference_mode()
    def evaluate_step(self):
        """Run the policy on evaluation dataset"""

        if self.eval_env is None:
            return

        local_rollout_size = (
            self.config.eval_rollout_size
            // self.config.eval_batch_size
            // self.dist_ops.world_size
        )

        # Use greedy sampling
        if self.is_inference_engine_enabled():
            eval_sampling_params = {
                'max_new_tokens': self.config.max_completion_tokens,
                'temperature': 0.0,
            }
        else:
            eval_sampling_params = {
                'max_new_tokens': self.config.max_completion_tokens,
                'temperature': None,
                'top_p': None,
                'top_k': None,
                'repetition_penalty': None,
                'do_sample': False,
            }

        with self.unwrapped_model_for_generation() as policy_model:
            for _ in range(local_rollout_size):
                outputs = self.eval_env.rollout(
                    policy_model, eval_sampling_params
                )
                self.log_batch_episodes(
                    self._eval_phase, outputs, self.global_step
                )

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
                outputs = self.train_env.rollout(
                    policy_model, train_sampling_params
                )
                if outputs:
                    collected_episodes.extend(outputs)
                    local_count += len(outputs)
                    self.log_batch_episodes(
                        self._train_phase, outputs, self.global_step
                    )

        return collected_episodes

    @torch.no_grad()
    def _convert_batch_episodes_to_transitions(
        self,
        episodes: List[EpisodeData],
    ) -> List[TransitionData]:
        """Converts the raw env rollout episodes to RL training transition samples.

        Args:
            episodes (List[TransitionData]): A list of episodes from the env rollout.
        Returns:
            List[TransitionData]: A list of training sample for training
        """

        if not episodes:
            return []

        # Training specific pre-processing
        # This is the terminal-step reward from outcome function or reward model
        rewards = self.transform_batch_rewards(episodes).cpu()

        if self.config.normalize_rewards:
            normed_rewards = self.whiten(rewards, shift_mean=False)
        else:
            normed_rewards = rewards

        # Prepare batched sequences for model forward pass
        sequences = [
            torch.concat([ep.prompt_tokens, ep.completion_tokens]).long()
            for ep in episodes
        ]
        sequence_lengths = [
            len(seq) for seq in sequences
        ]  # Total length (prompt + completion)

        # States: tokens 0 to N-1; Actions: tokens 1 to N
        state_sequences = [seq[:-1] for seq in sequences]
        action_sequences = [seq[1:] for seq in sequences]
        batch_states = pad_sequence(
            state_sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.device)
        batch_actions = pad_sequence(
            action_sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.device)

        batch_attention_mask = (
            batch_states != self.tokenizer.pad_token_id
        ).bool()

        # Policy Model
        batch_pi_logits = self.policy_engine.forward(
            input_ids=batch_states, attention_mask=batch_attention_mask
        ).logits
        # Ensure batch_actions is LongTensor for gather
        batch_pi_logprobs = self.compute_logprobs_from_logits(
            batch_pi_logits,
            batch_actions,
        )
        self.clean_up()

        # Reference Model (if applicable)
        if (
            self.config.kl_loss_coef > 0
            and hasattr(self, 'reference_model')
            and self.reference_model
        ):
            batch_ref_logits = self.reference_model.forward(
                input_ids=batch_states, attention_mask=batch_attention_mask
            ).logits
            batch_ref_logprobs = self.compute_logprobs_from_logits(
                batch_ref_logits, batch_actions
            )
            self.clean_up()
        else:
            # Safer placeholder: Use policy logprobs -> KL=0, or zeros
            # Using policy logprobs ensures KL is zero if ref model not used
            batch_ref_logprobs = batch_pi_logprobs.clone()

        # State value
        batch_values: torch.Tensor = self.value_engine.forward(
            input_ids=batch_states, attention_mask=batch_attention_mask
        ).values  # [batch_size, seq_len]

        del batch_attention_mask

        # Move results back to CPU for per-episode processing and storage
        batch_states = batch_states.cpu()
        batch_actions = batch_actions.cpu()
        batch_values = batch_values.cpu()
        batch_pi_logprobs = batch_pi_logprobs.cpu()
        batch_ref_logprobs = batch_ref_logprobs.cpu()

        transitions = []

        for i, ep in enumerate(episodes):
            seq_len = sequence_lengths[i]  # Total length (prompt + completion)
            prompt_len = ep.prompt_length
            completion_len = ep.completion_length

            # Ensure sequence length calculation matches
            if seq_len != prompt_len + completion_len:
                self.logger.error(
                    f"Episode {i}: Mismatch seq_len ({seq_len}) vs prompt ({prompt_len}) + completion ({completion_len})"
                )
                continue  # Skip this problematic episode

            states = state_sequences[i]
            actions = action_sequences[i]
            values = batch_values[i, : len(actions)]
            pi_logprobs = batch_pi_logprobs[i, : len(actions)]
            ref_logprobs = batch_ref_logprobs[i, : len(actions)]

            # Do not include the prompt tokens in the loss
            # for example, if we have a sequence token ids: [1, 2, 3, 4, 5, 6, 7]
            # where [1, 2, 3, 4] are the prompt tokens
            # and [5, 6, 7] are the completion tokens
            # the, the loss mask will be [0, 0, 0, 1, 1, 1]

            loss_mask = torch.zeros_like(actions, dtype=torch.bool)
            loss_mask[prompt_len - 1 :] = True

            assert loss_mask.sum().item() == ep.completion_length

            # Rewards are all zero for non-terminal step, and use the normalized reward for terminal step
            seq_rewards = torch.zeros_like(actions, dtype=self.torch_dtype)
            seq_rewards[-1] = rewards[i]
            seq_rewards_for_adv = seq_rewards.clone()
            seq_rewards_for_adv[-1] = normed_rewards[i]

            # Compute GAE advantages for policy update
            _, advantages = self.masked_returns_and_gae_advantages(
                seq_rewards_for_adv,
                values,
                loss_mask,
                self.config.gamma,
                self.config.gae_lambda,
            )

            # Don't mix advantage estimations for value function
            # like GRPO, but we only use the MC returns to train value model
            returns = self.masked_monte_carlo_returns(
                seq_rewards,
                loss_mask,
                self.config.gamma,
            )
            # advantages = returns # <--- this is exactly what GRPO does

            assert (
                states.shape
                == actions.shape
                == pi_logprobs.shape
                == ref_logprobs.shape
                == advantages.shape
                == returns.shape
                == loss_mask.shape
            )

            transitions.append(
                TransitionData(
                    states=states,
                    actions=actions,
                    values=values,
                    loss_mask=loss_mask,
                    advantages=advantages,
                    returns=returns,
                    pi_logprobs=pi_logprobs,
                    ref_logprobs=ref_logprobs,
                )
            )

        return transitions

    def _train_policy_step(self, train_dataloader: DataLoader):
        """Performs the policy model update using collected rollout."""
        for _ in range(self.config.policy_num_updates):
            for i, micro_batch in enumerate(train_dataloader):
                input_ids = micro_batch.states.to(self.device)
                attention_mask = (
                    input_ids != self.tokenizer.pad_token_id
                ).bool()
                pi_logits = self.policy_engine.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                loss = self.compute_policy_loss(pi_logits, micro_batch)

                del (micro_batch, input_ids, attention_mask, pi_logits)
                self.clean_up()

                self.policy_engine.backward(loss)
                self.policy_engine.step()

                if self.policy_engine.is_gradient_accumulation_boundary():
                    self.policy_update_count += 1
                    self.logger.log_scalar(
                        'train/policy_update', self.policy_update_count
                    )
                    self.logger.log_scalar(
                        'train/policy_learning_rate',
                        self.policy_engine.get_lr()[0],
                    )

    def _train_value_step(self, train_dataloader: DataLoader):
        """Performs the value model update using collected rollout."""
        for _ in range(self.config.value_num_updates):
            for i, micro_batch in enumerate(train_dataloader):
                input_ids = micro_batch.states.to(self.device)
                attention_mask = (
                    input_ids != self.tokenizer.pad_token_id
                ).bool()
                pred_values = self.value_engine.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                ).values

                loss = self.compute_value_loss(pred_values, micro_batch)

                del (micro_batch, input_ids, attention_mask, pred_values)
                self.clean_up()

                self.value_engine.backward(loss)
                self.value_engine.step()

                if self.value_engine.is_gradient_accumulation_boundary():
                    self.value_update_count += 1
                    self.logger.log_scalar(
                        'train/value_update', self.value_update_count
                    )
                    self.logger.log_scalar(
                        'train/value_learning_rate',
                        self.value_engine.get_lr()[0],
                    )

    def _train_collate_fn(self, batch: List[TransitionData]) -> TransitionData:
        """Collate function for DataLoader during training"""
        eos_token_id = self.tokenizer.eos_token_id
        torch_dtype = self.torch_dtype

        # Pad states and actions (long tensors)
        batch_states = pad_sequence(
            [item.states for item in batch],
            batch_first=True,
            padding_value=eos_token_id,
        ).long()
        batch_actions = pad_sequence(
            [item.actions for item in batch],
            batch_first=True,
            padding_value=eos_token_id,
        ).long()

        # Pad loss_mask (boolean tensor)
        batch_loss_mask = pad_sequence(
            [item.loss_mask for item in batch],
            batch_first=True,
            padding_value=False,
        ).bool()

        # Pad values, return, advantages, pi_logprobs, and ref_logprobs (float tensors)
        batch_values = (
            pad_sequence(
                [item.values for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )
        batch_returns = (
            pad_sequence(
                [item.returns for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )
        batch_advantages = (
            pad_sequence(
                [item.advantages for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )
        batch_pi_logprobs = (
            pad_sequence(
                [item.pi_logprobs for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )
        batch_ref_logprobs = (
            pad_sequence(
                [item.ref_logprobs for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )

        return TransitionData(
            states=batch_states,
            actions=batch_actions,
            values=batch_values,
            loss_mask=batch_loss_mask,
            pi_logprobs=batch_pi_logprobs,
            ref_logprobs=batch_ref_logprobs,
            returns=batch_returns,
            advantages=batch_advantages,
        )
