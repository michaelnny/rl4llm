"""Implements GRPO trainer"""

from typing import List, Optional, Union

import torch
from deepspeed import DeepSpeedEngine
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.core.base_trainer import RewardTransform, RLConfig, RLTrainer
from rl4llm.core.distributed import DistributedManager
from rl4llm.envs import EpisodeData, LocalLLMEnv
from rl4llm.logging import LoggingManager


class GRPOConfig(RLConfig):
    """GRPO config instance for RL LLM"""

    group_reward_zero_mean: bool = Field(
        False,
        description='Normalized group reward to have a zero mean without standard deviation scaling',
    )


class TransitionData(BaseModel):
    """GPPO transition for training"""

    states: torch.Tensor = Field(
        ...,
        description='A long tensor for token sequences from t=0, 1, ..., T-1',
    )
    actions: torch.Tensor = Field(
        ...,
        description='A long tensor for token sequences from t=1, 2, ..., T-1, T',
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
    advantages: torch.Tensor = Field(
        ...,
        description='A float tensor for GAE advantages estimate corresponding to token sequences from t=1, 2, ..., T-1, T',
    )

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
        tensor_shapes = [
            tensor.shape if isinstance(tensor, torch.Tensor) else None
            for tensor in tensors
        ]

        if len(set(tensor_shapes)) > 1:
            raise ValueError(f"Tensors have mismatched shapes: {tensor_shapes}")

        return values

    class Config:
        arbitrary_types_allowed = True


class GRPOTrainer(RLTrainer):
    """GRPO trainer for LLM"""

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy_engine: DeepSpeedEngine,
        dist_manager: DistributedManager,
        logger: LoggingManager,
        artifacts_path: str,
        train_env: LocalLLMEnv,
        eval_env: Optional[LocalLLMEnv] = None,
        ref_model: Optional[Union[PreTrainedModel, DeepSpeedEngine]] = None,
        inference_client: Optional[InferenceClient] = None,
        reward_transform_fn: Optional[RewardTransform] = None,
        seed: Optional[int] = 175,
    ):

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            policy_engine=policy_engine,
            dist_manager=dist_manager,
            logger=logger,
            artifacts_path=artifacts_path,
            train_env=train_env,
            eval_env=eval_env,
            ref_model=ref_model,
            inference_client=inference_client,
            reward_transform_fn=reward_transform_fn,
            seed=seed,
        )

        self.config: GRPOConfig = config  # for better type hinting

    def initialize_trainer(self):
        """Initialize GRPO specific settings"""
        pass

    def build_train_loader(
        self, experience: List[List[EpisodeData]]
    ) -> DataLoader:
        """Creates a train loader using the collected experiences.

        Args:
            experience (List[List[EpisodeData]]): local rollout episodes in group
        Returns:
            DataLoader: A dataloader ready for training.
        """
        flatted_samples = []
        for group_eps in experience:
            # we need to process the samples from same question outcome
            # in a single group for GRPO
            samples = self._convert_group_episodes_to_transitions(group_eps)
            if samples:
                flatted_samples.extend(samples)

        if not flatted_samples:
            raise ValueError('No samples for training')

        local_rollout_size = (
            self.config.train_rollout_size // self.dist_manager.world_size
        )

        if len(flatted_samples) > local_rollout_size:
            flatted_samples = flatted_samples[:local_rollout_size]

        data_loader = DataLoader(
            flatted_samples,
            batch_size=self.config.train_micro_batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self._train_collate_fn,
            drop_last=True,
        )

        return data_loader

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

        if self.config.normalize_advantages:
            advantages = self.masked_whiten(advantages, loss_mask)

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
            approxkl = 0.5 * self.masked_mean(
                torch.square(pi_logprobs - behavior_logprobs), loss_mask
            )
            clipfrac = self.masked_mean(
                torch.lt(pg_losses2, pg_losses1), loss_mask
            )

        # First average over the sequence length, then average over the batch
        pg_loss = self.masked_mean(pg_losses, loss_mask, dim=1).mean()

        # Compute entropy for the policy
        # Convert log probabilities to probabilities first
        probs = torch.exp(pi_logprobs)
        entropies = -torch.sum(probs * pi_logprobs * loss_mask, dim=-1)
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
            ref_logprobs = experience_batch.ref_logprobs.to(self.device)
            # Compute the KL divergence between the model and the reference model
            per_token_kl = (
                torch.exp(ref_logprobs - pi_logprobs)
                - (ref_logprobs - pi_logprobs)
                - 1
            )

            # # Clamp for stability
            # per_token_log_ratio = torch.clamp(ref_logprobs - pi_logprobs, min=-20, max=20)
            # per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1.0

            kl = self.masked_mean(per_token_kl, loss_mask, dim=1).mean()
            kl_loss = self.config.kl_loss_coef * kl

            loss += kl_loss
            self.logger.log_scalar('train/kl_loss', kl_loss.detach().item())
            self.logger.log_scalar('objective/kl', kl.detach().item())

        return loss

    def train_step(self, train_dataloader: DataLoader):
        """Performs the policy update using collected rollout."""

        for _ in range(self.config.num_updates):
            for i, micro_batch in enumerate(train_dataloader):
                input_ids = micro_batch.states.to(self.device)
                attention_mask = (
                    input_ids != self.tokenizer.pad_token_id
                ).bool()
                pi_logits = self.policy_engine.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                loss = self.compute_loss(pi_logits, micro_batch)

                del micro_batch, input_ids, attention_mask, pi_logits
                self.clean_up()

                self.policy_engine.backward(loss)
                self.policy_engine.step()

                if self.policy_engine.is_gradient_accumulation_boundary():
                    self.policy_update_count += 1
                    self.logger.log_scalar(
                        'train/policy_update', self.policy_update_count
                    )
                    self.logger.log_scalar(
                        'train/learning_rate',
                        self.policy_engine.get_lr()[0],
                    )

    @torch.inference_mode()
    def evaluate_step(self):
        """Run the policy on evaluation dataset"""

        if self.eval_env is None:
            return

        local_rollout_size = (
            self.config.eval_rollout_size
            // self.config.eval_batch_size
            // self.dist_manager.world_size
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
            self.config.train_rollout_size // self.dist_manager.world_size
        )
        collected_episodes: List[List[EpisodeData]] = []
        local_count = 0

        with self.unwrapped_model_for_generation() as policy_model:
            while local_count < local_rollout_size:

                outputs = self.train_env.rollout(
                    policy_model, train_sampling_params
                )

                if outputs:
                    # IMPORTANT: do not flatten the episodes yet
                    # as we need to normalize the rewards on group level
                    collected_episodes.extend([outputs])
                    local_count += len(outputs)

                    self.log_batch_episodes(
                        self._train_phase, outputs, self.global_step
                    )

        return collected_episodes

    @torch.no_grad()
    def _convert_group_episodes_to_transitions(
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
        if len(episodes) < 4:
            raise ValueError('Expect group episodes to be greater than 4')

        # Training specific pre-processing
        rewards = self.transform_batch_rewards(episodes).cpu()

        normalized_rewards = self._normalize_group_rewards(
            rewards, self.config.group_reward_zero_mean
        )

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

        del batch_attention_mask

        # Move results back to CPU for per-episode processing and storage
        batch_states = batch_states.cpu()
        batch_actions = batch_actions.cpu()
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
            seq_rewards[-1] = normalized_rewards[i]

            returns = self.compute_masked_monte_carlo_returns(
                seq_rewards, loss_mask, self.config.gamma
            )

            assert (
                states.shape
                == actions.shape
                == pi_logprobs.shape
                == ref_logprobs.shape
                == returns.shape
                == loss_mask.shape
            )

            transitions.append(
                TransitionData(
                    states=states,
                    actions=actions,
                    loss_mask=loss_mask,
                    advantages=returns,
                    pi_logprobs=pi_logprobs,
                    ref_logprobs=ref_logprobs,
                )
            )

        return transitions

    def _normalize_group_rewards(
        self,
        rewards: torch.Tensor,
        zero_mean_only: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize group rewards by subtracting the mean and dividing by the standard deviation.

        Args:
            rewards (torch.Tensor): List of rewards for the group.
            eps (float): Small value to prevent division by zero.
        Returns:
            torch.Tensor: Normalized rewards.
        """
        assert eps > 0.0, 'Epsilon must be positive'
        assert rewards.dim() == 1, 'Rewards must be 1-dimensional'
        if len(rewards) < 4:
            raise ValueError('Number of group rewards must be greater than 4')

        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)
        if zero_mean_only:
            return rewards - mean_reward

        return (rewards - mean_reward) / (std_reward + eps)

    def _train_collate_fn(self, batch: List[TransitionData]) -> TransitionData:
        """Collate function for DataLoader during training"""
        pad_token_id = self.tokenizer.pad_token_id
        torch_dtype = self.torch_dtype

        # Pad states and actions (long tensors)
        batch_states = pad_sequence(
            [item.states for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ).long()
        batch_actions = pad_sequence(
            [item.actions for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ).long()

        # Pad loss_mask (boolean tensor)
        batch_loss_mask = pad_sequence(
            [item.loss_mask for item in batch],
            batch_first=True,
            padding_value=False,
        ).bool()

        # Pad advantages, pi_logprobs, and ref_logprobs (float tensors)
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
            loss_mask=batch_loss_mask,
            pi_logprobs=batch_pi_logprobs,
            ref_logprobs=batch_ref_logprobs,
            advantages=batch_advantages,
        )
