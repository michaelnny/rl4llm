"""Implements Value Model trainer for RL"""

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


class ValueNetConfig(BaseRLConfig):
    """Value model config instance"""

    """Training specific"""
    train_rollout_size: int = Field(
        10240,
        ge=1,
        le=102400,
        description='Number of samples to collect before update model',
    )
    num_epochs: int = Field(
        4,
        ge=1,
        le=5,
        description='Update epochs for a collection of samples',
    )
    train_micro_batch_size: int = Field(
        4,
        ge=1,
        le=1024,
        description='Micro-batch size pre device/rank for training',
    )
    train_batch_size: int = Field(
        128,
        ge=16,
        le=1024,
        description='Global batch size across devices/ranks for training',
    )
    gamma: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description='Default discount factor for compute returns',
    )


class TransitionData(BaseModel):
    """Value transition data for training"""

    states: torch.Tensor = Field(
        ...,
        description='A long tensor for token sequences from t=0, 1, ..., T-1',
    )
    loss_mask: torch.Tensor = Field(
        ...,
        description='A boolean tensor (0s user tokens, 1s assistant tokens) corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    returns: torch.Tensor = Field(
        ...,
        description='A float tensor for returns estimate corresponding to token sequences from t=1, 2, ..., T-1, T',
    )

    @model_validator(mode='after')
    def check_tensor_shapes(cls, values):
        tensors = [values.states, values.loss_mask, values.returns]

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


class ValueNetTrainer(BaseRLTrainer):
    """Value model trainer for LLM"""

    def __init__(
        self,
        config: ValueNetConfig,
        tokenizer: PreTrainedTokenizer,
        value_engine: DeepSpeedEngine,
        log_config: Dict[str, Any],
        train_env: LocalLLMEnv,
        inference_client: InferenceClient,
        reward_transform_fn: Optional[RewardTransform] = None,
        seed: Optional[int] = 175,
    ):
        """Initialize the RL Value trainer instance"""

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            policy_engine=None,
            log_config=log_config,
            train_env=train_env,
            eval_env=None,
            ref_model=None,
            value_engine=value_engine,
            inference_client=inference_client,
            reward_transform_fn=reward_transform_fn,
            seed=seed,
        )

        if config.train_rollout_size % self.dist_ops.world_size != 0:
            raise ValueError(
                'Train rollout size must be divisible by world size'
            )

        self.config: ValueNetConfig = config  # for better type hinting

    def initialize_trainer(self):
        """Initialize algorithm specific settings"""
        pass

    def save_checkpoint(self, tag: str) -> None:
        """Save trained model in HF format"""
        subpath = f"epoch_{tag}"
        save_path = os.path.join(self.checkpoint_dir, subpath)
        self.logger.info(f"Saving HF checkpoint to {save_path}...")
        self.save_weights_hf_pretrained(self.value_engine, save_path)
        self.dist_ops.barrier()
        self.logger.info('Checkpoint saved.')

    def train(self, job_config: Dict):
        """
        Executes the main training loop.

        Args:
            job_config (Dict): Configuration dictionary containing job parameters such as
                hyperparameters and other metadata useful for logging or reproducibility.
        """
        self.logger.info('Initializing training loop...')
        if job_config and self.dist_ops.is_master:
            self.logger.log_hyperparams(job_config)

        self.dist_ops.synchronize()

        self.logger.info('Running rollout on training environment...')
        with self.logger.timer('generate_experience'):
            self._prepare_for_generation()
            local_experience = self.generate_experience()
            self.clean_up()
        self.dist_ops.barrier()

        self.logger.info('Preprocessing collected experience for training...')
        with self.logger.timer('pre_processing'):
            self._prepare_for_pre_processing()
            train_dataloader = self.build_train_loader(local_experience)
            self.clean_up()
        self.dist_ops.barrier()

        self.logger.info('Starting model training step...')
        with self.logger.timer('train_epochs'):
            self._prepare_for_training()
            self.train_step(train_dataloader)
            self.clean_up()
        self.dist_ops.barrier()

        self.logger.info('Training loop complete. Finalizing...')
        self.on_exit()

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

    def compute_loss(
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
        returns = experience_batch.returns.to(self.device)
        loss_mask = experience_batch.loss_mask.to(self.device)

        # Value loss
        losses = 0.5 * torch.square(returns - pred_values)
        loss = self.masked_mean(losses, loss_mask, dim=1).mean()

        with torch.no_grad():
            pred_error = self.masked_mean(
                torch.square(pred_values.detach() - returns.detach()),
                loss_mask,
                dim=1,
            ).mean()
            returns_var = self.masked_var(returns, loss_mask, dim=1).mean()
            var_explained = (1 - pred_error / (returns_var + 1e-8)).item()

        self.logger.log_scalar('train/loss', loss.detach().item())
        self.logger.log_scalar('value/error', pred_error.detach().item())
        self.logger.log_scalar('value/returns_var', returns_var.detach().item())
        self.logger.log_scalar('value/var_explained', var_explained)

        return loss

    def train_step(self, train_dataloader: DataLoader):
        """Performs the value model update using collected rollout."""

        self._configure_model(self.policy_engine, 'cpu', 'offload')

        for epoch in range(self.config.num_epochs):
            for i, micro_batch in enumerate(train_dataloader):
                input_ids = micro_batch.states.to(self.device)
                attention_mask = (
                    input_ids != self.tokenizer.pad_token_id
                ).bool()
                pred_values = self.value_engine.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                ).values

                # with torch.autograd.detect_anomaly():
                loss = self.compute_loss(pred_values, micro_batch)

                del (micro_batch, input_ids, attention_mask, pred_values)
                self.clean_up()

                self.value_engine.backward(loss)
                self.value_engine.step()

                if self.value_engine.is_gradient_accumulation_boundary():
                    self.value_update_count += 1
                    self.global_step += 1
                    self.logger.log_scalar(
                        'train/value_update', self.value_update_count
                    )
                    self.logger.log_scalar(
                        'train/learning_rate',
                        self.value_engine.get_lr()[0],
                    )

                    self.logger.aggregate_and_log(self.global_step)

            self.save_checkpoint(epoch)

    @torch.inference_mode()
    def evaluate_step(self):
        """ """
        pass

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
        step_count = 0
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
                    step_count += 1
                    # Log progress every 50 valid steps or at completion
                    if (
                        step_count % 50 == 0
                        or local_count >= local_rollout_size
                    ):
                        progress = (local_count / local_rollout_size) * 100
                        self.logger.info(
                            f"Progress: {progress:.2f}% ({local_count}/{local_rollout_size} episodes collected)"
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

        # Move results back to CPU for per-episode processing and storage
        batch_states = batch_states.cpu()

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

            # Works well when not using value function
            returns = self.masked_monte_carlo_returns(
                seq_rewards,
                loss_mask,
                self.config.gamma,
            )

            assert states.shape == returns.shape == loss_mask.shape

            transitions.append(
                TransitionData(
                    states=states,
                    loss_mask=loss_mask,
                    returns=returns,
                )
            )

        return transitions

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

        # Pad loss_mask (boolean tensor)
        batch_loss_mask = pad_sequence(
            [item.loss_mask for item in batch],
            batch_first=True,
            padding_value=0,
        ).bool()

        batch_returns = (
            pad_sequence(
                [item.returns for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )

        return TransitionData(
            states=batch_states,
            loss_mask=batch_loss_mask,
            returns=batch_returns,
        )
