import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import torch
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.core.base_trainer import RLConfig, RLTrainer
from rl4llm.core.distributed import DistributedManager
from rl4llm.envs import EpisodeData, LLMEnv
from rl4llm.generations import ExploreLLMGenerator
from rl4llm.logging import LoggingManager


class GRPOConfig(RLConfig):
    """GRPO config instance for RL LLM"""

    xml_format: Optional[bool] = Field(
        False, description='Check R1 style XML format for compute reward'
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
        00, ge=0, le=500, description='Unified top-k for both exploration'
    )
    replace_max_per_seq: Optional[int] = Field(
        0,
        ge=0,
        le=10,
        description='Maximum number of token replacements to the same sequence during exploration',
    )
    replace_threshold: Optional[float] = Field(
        0,
        ge=0,
        le=1.0,
        description='Source token probability scores to consider for replacement',
    )
    replace_prob: Optional[float] = Field(
        0,
        ge=0,
        le=1.0,
        description='Probability to replace source token with target token if conditions are meet',
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

    # @model_validator(mode='after')
    # def check_tensor_shapes(cls, values):
    #     tensors = [
    #         values.states,
    #         values.actions,
    #         values.loss_mask,
    #         values.pi_logprobs,
    #         values.ref_logprobs,
    #         values.advantages,
    #     ]

    #     # Ensure all tensors are of the same shape
    #     tensor_shapes = [
    #         tensor.shape if isinstance(tensor, torch.Tensor) else None
    #         for tensor in tensors
    #     ]

    #     if len(set(tensor_shapes)) > 1:
    #         raise ValueError(f"Tensors have mismatched shapes: {tensor_shapes}")

    #     return values

    class Config:
        arbitrary_types_allowed = True


class GRPOTrainer(RLTrainer):
    """"""

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy_engine: DeepSpeedEngine,
        dist_manager: DistributedManager,
        logger: LoggingManager,
        artifacts_path: str,
        train_env: LLMEnv,
        eval_env: Optional[LLMEnv] = None,
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
            seed=seed,
        )

        self.config: GRPOConfig = config

    def _initialize_trainer(self):
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

        # For custom exploring start where we skip do exploration for the <think> token
        self.think_token_len = (
            len(self.tokenizer.encode('<think>'))
            if self.config.xml_format
            else 0
        )

    #     self.explore_generator = self._create_explore_generator(
    #         self.policy_model
    #     )

    # def _create_explore_generator(
    #     self, policy_model: PreTrainedModel
    # ) -> ExploreLLMGenerator:
    #     """Create a custom generator wrapped around the policy model, which supports group temperature and stochastic sampling"""
    #     # Try replacing the end token with "Wait" for some samples
    #     source_tokens = []
    #     # # Determine which tokens should be replaced based on format
    #     if self.config.xml_format:
    #         source_tokens.append(self.tokenizer.encode('</think>')[0])
    #         source_tokens.append(self.tokenizer.encode(' </think>')[0])
    #         source_tokens.append(self.tokenizer.encode(':</think>')[0])
    #         source_tokens.append(self.tokenizer.encode('.</think>')[0])
    #     else:
    #         source_tokens.append(self.tokenizer.eos_token_id)

    #     self.special_tokens = ['Wait']
    #     target_tokens = [
    #         self.tokenizer.encode(f' {kwd}')[0] for kwd in self.special_tokens
    #     ]

    #     # we should only make the replacement for reasoning tokens
    #     prevent_patterns = [
    #         self.tokenizer.encode('</think>'),
    #         self.tokenizer.encode(' </think>'),
    #         self.tokenizer.encode('<answer>'),
    #     ]

    # return ExploreLLMGenerator(
    #     model=policy_model,
    #     tokenizer=self.tokenizer,
    #     device=self.device,
    #     source_tokens=source_tokens,
    #     target_tokens=target_tokens,
    #     prevent_patterns=prevent_patterns,
    # )

    def _get_exploration_epsilon(self) -> float:
        """Computes exploration epsilon based on the current iteration step count."""
        if self.config.explore_decay_steps == 0:
            self.explore_epsilon = 0.0
        elif self.iteration_count >= self.config.explore_decay_steps:
            self.explore_epsilon = self.config.explore_min_epsilon
        else:
            # Cosine decay schedule
            progress = self.iteration_count / self.config.explore_decay_steps
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

    @torch.inference_mode()
    def _generate_group_samples(self) -> List[TransitionData]:
        """Generate responses for a batch of questions

        Returns:
            List[TransitionData]: List of samples for all groups in the batch
        """

        gen_kwargs = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'max_new_tokens': self.config.max_completion_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'repetition_penalty': self.config.repetition_penalty,
            'num_return_sequences': self.config.group_size,
            'do_sample': True,
            'use_cache': True,
            'output_scores': False,
            'output_logits': False,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
            'explore_probability': self._get_exploration_epsilon(),  # control explore env custom logit
        }

        # enable_exploring = (
        #     (self.config.explore_steps > 0)
        #     and (explore_prob > 0)
        #     and (random.random() < explore_prob)
        # )

        # if enable_exploring:
        #     llm_generator = self.explore_generator

        # # apply group temperature
        # if self.config.group_temperature:
        #     # Spread temperature values according to self.config.group_size, where 0.0 means greedy sampling
        #     # this idea is similar how we do it in distributed RL training in classical RL
        #     # where we have multiple agents running in parallel, some agents are more exploratory than others
        #     temperature = torch.linspace(
        #         self.config.min_temperature,
        #         self.config.max_temperature,
        #         steps=self.config.group_size,
        #         dtype=self.torch_dtype,
        #         device=self.device,
        #     )
        #     # Round to 2 decimal places
        #     gen_kwargs['temperature'] = torch.round(temperature, decimals=2)

        # # apply exploring start
        # gen_kwargs['explore_steps'] = self.config.explore_steps
        # gen_kwargs['explore_skip_n'] = self.think_token_len
        # gen_kwargs['explore_top_k'] = self.config.explore_top_k

        # # apply token swap: like "</think>" -> "Wait"
        # if self.config.replace_max_per_seq > 0:
        #     gen_kwargs['explore_replace_prob'] = explore_prob
        #     gen_kwargs['replace_max_per_seq'] = (
        #         self.config.replace_max_per_seq
        #     )

        outputs = self.train_env.rollout(self.policy_model, gen_kwargs)

        self.log_batch_episodes(self._train_phase, outputs, self.global_step)

        return self._convert_group_episodes_to_transitions(outputs)

    @torch.inference_mode()
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
        if len(episodes) < 2:
            raise ValueError('Expect group episodes to be greater than 2')

        rewards = torch.tensor(
            [ep.reward_dict['accuracy_reward'] for ep in episodes],
            dtype=self.torch_dtype,
        ).cpu()

        # discard samples as they leads to zero advantages -> zero gradients
        if (
            torch.std(rewards, unbiased=False)
            <= self.group_reward_std_threshold
        ):
            self.logger.warning(
                f"Skipping samples with identical rewards, \
                    minimum group reward std: {self.group_reward_std_threshold}"
            )
            self.logger.log_scalar('others/skipped_sample_count', len(rewards))
            return []

        # Training specific processing
        normalized_rewards = (
            self.normalize_group_rewards(rewards)
            if self.config.normalize_rewards
            else rewards
        )

        # Prepare Batched Sequences for Model Input
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

            # IMPORTANT with slicing upper bound is exclusive
            states = state_sequences[i]
            actions = action_sequences[i]
            pi_logprobs = batch_pi_logprobs[i, : len(actions)]
            ref_logprobs = batch_ref_logprobs[i, : len(actions)]

            assert states[-1] != self.tokenizer.pad_token_id
            assert states[-1] != self.tokenizer.eos_token_id

            # Do not include the prompt tokens in the loss
            # for example, if we have a sequence token ids: [1, 2, 3, 4, 5, 6, 7]
            # where [1, 2, 3, 4] are the prompt tokens
            # and [5, 6, 7] are the completion tokens
            # the, the loss mask will be [0, 0, 0, 1, 1, 1]

            loss_mask = torch.zeros_like(actions, dtype=torch.bool)
            loss_mask[prompt_len - 1 :] = True

            assert loss_mask.sum().item() == ep.completion_length

            returns = normalized_rewards[i] * loss_mask.cpu()
            assert torch.nonzero(returns).sum() > 0

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

    def normalize_group_rewards(
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
        if len(rewards) <= 1:
            return rewards

        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)
        if zero_mean_only:
            return rewards - mean_reward

        return (rewards - mean_reward) / (std_reward + eps)

    @torch.inference_mode()
    def generate_experience(self) -> List[TransitionData]:
        """Generates samples using the current policy."""

        assert not self.policy_model.training
        collected_samples: List[TransitionData] = []

        # we always use batch size of 1 during training roll out
        local_rollout_size = (
            self.config.train_rollout_size // self.dist_manager.world_size
        )

        while len(collected_samples) < local_rollout_size:
            samples = self._generate_group_samples()
            if samples:
                collected_samples.extend(samples)

        if len(collected_samples) > local_rollout_size:
            collected_samples = collected_samples[:local_rollout_size]

        return collected_samples

    def build_train_batch(self, experience: List[TransitionData]) -> DataLoader:
        data_loader = DataLoader(
            experience,
            batch_size=self.config.train_micro_batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self._train_collate_fn,
            drop_last=True,
        )

        return data_loader

    def compute_loss(
        self, pi_logits: torch.Tensor, experience_batch: TransitionData
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single training batch

        Args:
            pi_logits (torch.Tensor): Raw logits of actions computed using
                current policy, shape [batch_size, seq_len]
            experience_batch (TransitionData): A batch of samples collected
                during generation

        Returns:
            Tuple[torch.Tensor, Dict]: Tuple containing the total loss tensor
                and a dictionary of metrics
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
        entropy_loss = self.config.entropy_loss_coef * entropy

        # Initialize metrics with common values
        metrics = {
            'train/pg_loss': pg_loss.detach().item(),
            'train/entropy_loss': entropy_loss.detach().item(),
            'policy/entropy': entropy.detach().item(),
            'policy/approxkl': approxkl.detach().item(),
            'policy/clipfrac': clipfrac.detach().item(),
        }

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

            loss = pg_loss + kl_loss + entropy_loss
            metrics.update(
                {
                    'train/kl_loss': kl_loss.detach().item(),
                    'objective/kl': kl.detach().item(),
                }
            )
        else:
            loss = pg_loss + entropy_loss

        return loss, metrics

    def train_step(self, train_dataloader: DataLoader):
        """Performs the policy update phase."""

        for _ in range(self.config.num_updates):
            for i, micro_batch in enumerate(train_dataloader):
                input_ids = micro_batch.states.to(self.device)
                attention_mask = (
                    input_ids != self.tokenizer.pad_token_id
                ).bool()
                pi_logits = self.policy_engine.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                loss, metrics = self.compute_loss(pi_logits, micro_batch)

                del micro_batch, input_ids, attention_mask, pi_logits
                self.clean_up()

                self.policy_engine.backward(loss)
                self.policy_engine.step()

                for k, v in metrics.items():
                    self.logger.log_scalar(k, v)

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

        assert not self.policy_model.training

        local_rollout_size = (
            self.config.eval_rollout_size
            // self.config.eval_batch_size
            // self.dist_manager.world_size
        )

        # Use greedy sampling
        eval_gen_kwargs = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'max_new_tokens': self.config.max_completion_tokens,
            'temperature': None,
            'top_p': None,
            'top_k': None,
            'repetition_penalty': None,
            'num_return_sequences': 1,
            'do_sample': False,
            'use_cache': True,
            'output_scores': False,
            'output_logits': False,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
        }

        for _ in range(local_rollout_size):
            outputs = self.eval_env.rollout(self.policy_model, eval_gen_kwargs)
            self.log_batch_episodes(self._eval_phase, outputs, self.global_step)

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
