import math
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
from rl4llm.envs import LLMEnv, EpisodeData
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.logging import LoggingManager


class GRPOConfig(RLConfig):
    """GRPO config instance for RL LLM"""

    ...


class TransitionData(BaseModel):
    """GPPO transition for training"""

    states: torch.Tensor = Field(
        ...,
        description="A long tensor for token sequences from t=0, 1, ..., T-1",
    )
    actions: torch.Tensor = Field(
        ...,
        description="A long tensor for token sequences from t=1, 2, ..., T-1, T",
    )
    loss_mask: torch.Tensor = Field(
        ...,
        description="A boolean tensor (0s user tokens, 1s assistant tokens) corresponding to token sequences from t=1, 2, ..., T-1, T",
    )
    pi_logprobs: torch.Tensor = Field(
        ...,
        description="A float tensor for action logprobs corresponding to token sequences from t=1, 2, ..., T-1, T",
    )
    ref_logprobs: torch.Tensor = Field(
        ...,
        description="A float tensor for action logprobs from reference model corresponding to token sequences from t=1, 2, ..., T-1, T",
    )
    advantages: torch.Tensor = Field(
        ...,
        description="A float tensor for GAE advantages estimate corresponding to token sequences from t=1, 2, ..., T-1, T",
    )

    @model_validator(mode="after")
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
        # train_dataset: Union[List[Dict] | Dataset],
        # eval_dataset: Optional[Union[List[Dict] | Dataset]] = None,
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
            # train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            seed=seed,
        )

        self.math_grader = math_problem_grader

    def _initialize_trainer(self):
        """Initialize GRPO specific settings"""

        # avoid adding group of samples with almost identical outcomes
        _dummy_rewards = torch.tensor([0] * self.config.group_size, dtype=torch.float32)
        _idx = math.ceil(self.config.group_size * 0.05)
        _dummy_rewards[:_idx] = 1.0
        self.group_reward_std_threshold = torch.std(_dummy_rewards, unbiased=False)

    @torch.inference_mode()
    def _generate_group_samples(self) -> List[TransitionData]:
        """Generate responses for a batch of questions

        Returns:
            List[TransitionData]: List of samples for all groups in the batch
        """

        gen_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": self.config.max_completion_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "num_return_sequences": self.config.group_size,
            "do_sample": True,
            "use_cache": True,
            "output_scores": False,
            "output_logits": False,
            "return_dict_in_generate": True,
            "return_legacy_cache": False,
        }

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
            raise ValueError("Expect group episodes to be greater than 2")

        rewards = torch.tensor(
            [ep.reward_dict["accuracy_reward"] for ep in episodes],
            dtype=self.torch_dtype,
        ).cpu()

        # discard samples as they leads to zero advantages -> zero gradients
        if torch.std(rewards, unbiased=False) <= self.group_reward_std_threshold:
            self.logger.warning(
                f"Skipping samples with identical rewards, \
                    minimum group reward std: {self.group_reward_std_threshold}"
            )
            self.logger.log_scalar("generation/skipped_sample", len(rewards))
            return []

        # Training specific processing
        normalized_rewards = (
            self.normalize_group_rewards(rewards)
            if self.config.normalize_rewards
            else rewards
        )

        # Prepare Batched Sequences for Model Input
        sequences = [
            torch.concat([ep.prompt_tokens, ep.completion_tokens]) for ep in episodes
        ]
        sequence_lengths = [
            len(seq) for seq in sequences
        ]  # Total length (prompt + completion)

        # Pad sequences for batch processing
        batch_sequences = (
            pad_sequence(
                sequences,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            .to(self.device)
            .long()
        )

        # Prepare states (inputs) and actions (targets) for the language model
        # States: tokens 0 to N-1; Actions: tokens 1 to N
        batch_states = batch_sequences[:, :-1]
        batch_actions = batch_sequences[:, 1:]
        batch_attention_mask = (batch_states != self.tokenizer.pad_token_id).bool()

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
            and hasattr(self, "reference_model")
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

        # Move results back to CPU for per-episode processing and storage
        batch_pi_logprobs = batch_pi_logprobs.cpu()
        batch_ref_logprobs = batch_ref_logprobs.cpu()
        batch_sequences = batch_sequences.cpu()

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
            states = batch_sequences[i, :seq_len]
            actions = batch_sequences[i, 1 : seq_len + 1]
            pi_logprobs = batch_pi_logprobs[i, :seq_len]
            ref_logprobs = batch_ref_logprobs[i, :seq_len]
            mask = ([0] * ep.prompt_length - 1) + [0] * ep.prompt_length
            loss_mask = torch.tensor(mask, dtype=torch.bool)

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
        assert eps > 0.0, "Epsilon must be positive"
        assert rewards.dim() == 1, "Rewards must be 1-dimensional"
        if len(rewards) <= 1:
            return rewards

        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)
        if zero_mean_only:
            return rewards - mean_reward

        return (rewards - mean_reward) / (std_reward + eps)

    def generate_experience(self) -> List[TransitionData]:
        """Generates samples using the current policy."""

        assert not self.policy_model.training
        collected_samples: List[TransitionData] = []

        local_rollout_size = self.config.rollout_size // self.dist_manager.world_size

        with self.logger.timer("generation"):
            while len(collected_samples) < local_rollout_size:
                samples = self._generate_group_samples()
                if samples:
                    collected_samples.extend(samples)

            if len(collected_samples) > local_rollout_size:
                collected_samples = collected_samples[:local_rollout_size]

        # TODO log more metrics
        # self.logger.log_scalar('generation/episodes_total', self.train_episode_count)
        # self.logger.log_scalar('generation/explore_epsilon', self.explore_epsilon)

        return collected_samples

    def build_train_batch(self, experience: List[TransitionData]) -> DataLoader:
        data_loader = DataLoader(
            experience,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
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
        ref_logprobs = experience_batch.ref_logprobs.to(self.device)
        loss_mask = experience_batch.loss_mask.to(self.device)

        if self.config.normalize_advantages:
            advantages = self.masked_whiten(advantages, loss_mask)

        # PPO clipped surrogate PG loss
        pi_logprobs = self.compute_logprobs_from_logits(pi_logits, actions)
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
        pg_losses1 = ratio * advantages.detach()
        pg_losses2 = clipped_ratio * advantages.detach()
        pg_losses = -torch.min(pg_losses1, pg_losses2)

        approxkl = 0.5 * self.masked_mean(
            torch.square(pi_logprobs - behavior_logprobs), loss_mask
        )
        clipfrac = self.masked_mean(torch.lt(pg_losses2, pg_losses1), loss_mask)

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
            "train/pg_loss": pg_loss.detach().item(),
            "train/entropy_loss": entropy_loss.detach().item(),
            "policy/entropy": entropy.detach().item(),
            "policy/approxkl": approxkl.detach().item(),
            "policy/clipfrac": clipfrac.detach().item(),
        }

        # Compute KL divergence if coefficient is positive
        if self.config.kl_loss_coef > 0:
            # Compute the KL divergence between the model and the reference model
            per_token_kl = (
                torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
            )

            # # Clamp for stability
            # per_token_log_ratio = torch.clamp(ref_logprobs - pi_logprobs, min=-20, max=20)
            # per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1.0

            kl = self.masked_mean(per_token_kl, loss_mask, dim=1).mean()
            kl_loss = self.config.kl_loss_coef * kl

            loss = pg_loss + kl_loss + entropy_loss
            metrics.update(
                {
                    "train/kl_loss": kl_loss.detach().item(),
                    "objective/kl": kl.detach().item(),
                }
            )
        else:
            loss = pg_loss + entropy_loss

        return loss, metrics

    def train_step(self, train_dataloader: DataLoader):
        """Performs the policy update phase."""

        self._prepare_for_training()

        for _ in range(self.config.num_updates):
            for i, micro_batch in enumerate(train_dataloader):
                input_ids = micro_batch.states
                attention_mask = (input_ids != self.tokenizer.pad_token_id).bool()
                pi_logits = self.policy_engine.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                loss, metrics = self.compute_loss(pi_logits, micro_batch)
                self.policy_engine.backward(loss)
                self.policy_engine.step()

                del input_ids, attention_mask, pi_logits
                torch.cuda.empty_cache()

                for k, v in metrics.items():
                    self.logger.log_scalar(k, v)

                if self.policy_engine.is_gradient_accumulation_boundary():
                    self.policy_update_count += 1
                    self.logger.log_scalar(
                        "train/policy_update", self.policy_update_count
                    )
                    self.logger.log_scalar(
                        "train/learning_rate",
                        self.policy_engine.get_lr()[0],
                    )

    def evaluate_step(self):
        pass

    def _eval_collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader during evaluation"""
        pad_token_id = self.tokenizer.pad_token_id

        # Extract input_ids and attention_mask as lists of tensors
        input_ids_list = [sample["input_ids"] for sample in batch]
        attention_mask_list = [sample["attention_mask"] for sample in batch]

        # Dynamically pad to the longest sequence in the batch
        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
            padding_side="left",
        )
        attention_mask = pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )

        # Collect other fields
        questions = [sample["question"] for sample in batch]
        ground_truths = [sample["ground_truth"] for sample in batch]
        task_types = [sample["task_type"] for sample in batch]

        return {
            "input_ids": input_ids.to(
                self.device
            ),  # Shape: [batch_size, max_seq_len_in_batch]
            "attention_mask": attention_mask.to(self.device),
            "questions": questions,
            "ground_truths": ground_truths,
            "task_types": task_types,
        }

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
            states=batch_states.to(self.device),
            actions=batch_actions.to(self.device),
            loss_mask=batch_loss_mask.to(self.device),
            pi_logprobs=batch_pi_logprobs.to(self.device),
            ref_logprobs=batch_ref_logprobs.to(self.device),
            advantages=batch_advantages.to(self.device),
        )
