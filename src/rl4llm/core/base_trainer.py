"""
Base RL trainer for LLMs in distributed training setup.

This module defines an abstract `RLTrainer` class that provides the foundational
infrastructure for reinforcement learning algorithms with LLM environments,
including training loop, checkpointing, model sync, and logging.
"""

import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import deepspeed
import torch
from deepspeed import DeepSpeedEngine
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.constants import EVAL_PHASE, LOGGING_PHASES, TRAIN_PHASE
from rl4llm.core.base_env import BaseEnv, EpisodeData
from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.core.deepspeed_mixin import DeepSpeedUtilsMixin
from rl4llm.core.distributed import DistributedManager
from rl4llm.core.training_mixin import TrainingMixin
from rl4llm.logging import LoggingManager


class RLConfig(BaseModel):
    """Basic config for RL fine-tuning for LLM"""

    """For RL sample generation"""
    max_prompt_tokens: Optional[int] = Field(
        1024,
        ge=256,
        le=10240,
        description='Skip sample with prompt length greater than this to avoid peak memory spikes',
    )
    max_completion_tokens: Optional[int] = Field(
        4096, ge=50, description='Maximum number of new tokens to generate'
    )
    temperature: Optional[float] = Field(
        0.9, gt=0.0, le=1.0, description='Sampling temperature for generation'
    )
    repetition_penalty: Optional[float] = Field(
        1.0, gt=0.0, le=2.0, description='Repetition penalty for generation'
    )
    top_p: Optional[float] = Field(
        1.0, ge=0.0, le=1.0, description='Sampling top-p for generation'
    )
    top_k: Optional[int] = Field(
        50, ge=-1, le=1000, description='Sampling top-k for generation'
    )
    group_size: int = Field(
        8,
        ge=4,
        le=256,
        description='Number of group outcomes for single question',
    )

    """For RL PPO training"""
    max_steps: int = Field(
        10000, ge=1, description='How long to run the training'
    )
    train_rollout_size: int = Field(
        1024,
        ge=1,
        le=5120,
        description='Number of samples to collect before update policy',
    )
    num_updates: int = Field(
        4,
        ge=1,
        le=5,
        description='PPO update epochs for a collection of samples',
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
    clip_eps: float = Field(
        0.2, ge=0.0, le=1.0, description='PPO policy loss clip epsilon'
    )
    gamma: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description='Default discount factor for compute returns',
    )
    normalize_rewards: bool = Field(True, description='Normalized rewards')
    normalize_advantages: bool = Field(
        False, description='Normalized advantages before compute PG loss'
    )
    entropy_loss_coef: float = Field(
        0.0, ge=0.0, le=1.0, description='Entropy loss coefficient'
    )
    kl_loss_coef: float = Field(
        0.04, ge=0.0, le=1.0, description='KL penalty loss coefficient'
    )
    # clip_grad_norm: Optional[float] = Field(0.0, ge=0.0, le=10.0, description='Clip L2 gradient norm')

    sync_reference_interval: int = Field(
        0,
        ge=0,
        le=1000,
        description='Interval to update reference model using latest policy',
    )
    checkpoint_interval: int = Field(
        0, ge=0, le=1000, description='Interval to save policy model checkpoint'
    )
    eval_interval: int = Field(
        100, ge=0, description='Interval to evaluate policy model'
    )
    eval_batch_size: int = Field(
        8,
        ge=0,
        le=1024,
        description='Batch size size pre device/rank for evaluation',
    )
    eval_rollout_size: int = Field(
        1024,
        ge=1,
        le=5120,
        description='Number of samples to collect for evaluation',
    )

    @model_validator(mode='after')
    def check_batch_size(cls, values):
        if values.train_batch_size % values.train_micro_batch_size != 0:
            raise ValueError(
                'Global train batch size must be divisible by mini batch size'
            )
        # if values.normalize_advantages and values.train_micro_batch_size < 4:
        #     raise ValueError(
        #         'Mini batch size must be at least 4 when normalize advantages is True'
        #     )
        return values

    class Config:
        arbitrary_types_allowed = True


class RLTrainer(ABC, TrainingMixin, DeepSpeedUtilsMixin):
    """
    Base class for training RL algorithms on LLM environments using DeepSpeed with Zero-1/Zero-2 only.
    """

    _train_phase: str = TRAIN_PHASE
    _eval_phase: str = EVAL_PHASE
    _log_phases: List[str] = LOGGING_PHASES

    def __init__(
        self,
        config: RLConfig,
        tokenizer: PreTrainedTokenizer,
        policy_engine: DeepSpeedEngine,
        dist_manager: DistributedManager,
        logger: LoggingManager,
        artifacts_path: str,
        train_env: BaseEnv,
        eval_env: Optional[BaseEnv] = None,
        inference_client: Optional[InferenceClient] = None,
        seed: Optional[int] = 175,
    ):
        if config.train_rollout_size % dist_manager.world_size != 0:
            raise ValueError(
                'Train rollout size must be divisible by world size'
            )
        if config.eval_rollout_size % dist_manager.world_size != 0:
            raise ValueError(
                'Evaluation rollout size must be divisible by world size'
            )

        self.config = config
        self.tokenizer = tokenizer
        self.policy_engine = policy_engine
        self.dist_manager = dist_manager
        self.logger = logger
        self.train_env = train_env
        self.eval_env = eval_env
        self.artifacts_path = artifacts_path
        self.seed = seed
        self.device = dist_manager.device
        self.torch_dtype = self.get_torch_dtype(policy_engine)

        self.checkpoint_dir = os.path.join(artifacts_path, 'checkpoints')
        if dist_manager.is_master:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        dist_manager.barrier()

        self.reference_model: Optional[PreTrainedModel] = (
            self._create_reference_model() if config.kl_loss_coef > 0 else None
        )

        self.inference_client = inference_client

        self.global_step = 0
        self.policy_update_count = 0
        self.ref_update_count = 0

        self.initialize_trainer()

        if self.is_inference_engine_enabled():
            self.logger.info('Using inference engine client.')

        self.logger.info('RL Trainer initialized.')

    @abstractmethod
    def initialize_trainer(self):
        """Algorithm-specific initialization hook."""
        pass

    @abstractmethod
    def generate_experience(self) -> List[Any]:
        """
        Collects experience (trajectories, rollouts) from the environment.
        """
        pass

    @abstractmethod
    def compute_loss(
        self, experience_batch: Any, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes policy loss and returns metrics.
        """
        pass

    @abstractmethod
    def build_train_loader(self, experience: List[Any]) -> DataLoader:
        """
        Converts collected experience into training samples.
        """
        pass

    @abstractmethod
    @torch.inference_mode()
    def evaluate_step(self) -> None:
        """
        Runs evaluation on current policy.
        """
        pass

    @abstractmethod
    def train_step(self, train_dataloader: DataLoader) -> None:
        """
        Performs a training step using a DataLoader.
        """
        pass

    @contextmanager
    def unwrapped_model_for_generation(
        self,
    ) -> Generator[Union[PreTrainedModel, InferenceClient], None, None]:
        """
        Returns the unwrapped model for generation.
        """
        if self.is_inference_engine_enabled():
            yield self.inference_client
        else:
            with self.with_unwrapped_model(self.policy_engine) as model:
                yield model
        self.clean_up()

    def _prepare_for_generation(self):
        """Switch models to eval mode for rollout."""

        if self.reference_model:
            self.reference_model = self.reference_model.to('cpu')
            self.reference_model.eval()

        if self.can_offload_state(self.policy_engine):
            self.policy_engine.offload_states()

        if self.is_cohost_inference_engine():
            try:
                self.policy_engine = self.policy_engine.to('cpu')
            except Exception as e:
                raise RuntimeError(
                    f"Failed to offload deepspeed engine, error: {str(e)}"
                )
        else:
            self.policy_engine.eval()

        self.clean_up()

    def _prepare_for_post_processing(self):
        """Switch models handle post-processing (like compute logprobs) before training."""
        if self.is_cohost_inference_engine():
            try:
                # Try offload GPU RAM caches
                self.inference_client.release_memory()

                self.policy_engine = self.policy_engine.to(self.device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to offload inference engine, error: {str(e)}"
                )

        if self.can_offload_state(self.policy_engine):
            self.policy_engine.offload_states()

        self.policy_engine.eval()
        if self.reference_model and self.reference_model.device != self.device:
            self.reference_model = self.reference_model.to(self.device)

        self.clean_up()

    def _prepare_for_training(self):
        """Switch models to train mode."""
        if self.is_cohost_inference_engine():
            try:
                # Try offload GPU RAM caches
                self.inference_client.release_memory()

            except Exception as e:
                raise RuntimeError(
                    f"Failed to offload inference engine, error: {str(e)}"
                )

        if self.reference_model:
            self.reference_model = self.reference_model.to('cpu')

        if self.can_offload_state(self.policy_engine):
            self.policy_engine.reload_states()

        self.policy_engine.train()

        self.clean_up()

    # TODO: consider using deepspeed inference engine for the reference model
    def _sync_reference_model(self):
        """Copies current policy weights to reference model."""
        if not self.reference_model:
            return
        self.logger.info('Syncing reference model...')
        with self.unwrapped_model_for_generation() as model:
            self.reference_model.load_state_dict(model.state_dict())
        self.ref_update_count += 1
        self.logger.info('Reference model synced.')
        self.clean_up()

    def _create_reference_model(self) -> Optional[PreTrainedModel]:
        """Creates a frozen copy of the current policy model."""
        self.logger.info('Creating reference model...')
        with self.unwrapped_model_for_generation() as model:
            ref_model = deepcopy(model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
        self.logger.info('Reference model created.')
        return ref_model

    def sync_policy_model(self) -> None:
        """Update policy model weights with the inference engine."""

        if not self.is_inference_engine_enabled():
            self.logger.info(
                'Inference engine not enabled, skipping weight sync.'
            )
            return

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"ckpt_sync_{self.global_step}_", dir=self.checkpoint_dir
            ) as temp_ckpt_path:
                # Save full checkpoint.
                self.logger.info(
                    f"Attempting to save weights to {temp_ckpt_path}"
                )
                self.save_weights_hf_pretrained(
                    self.policy_engine, temp_ckpt_path
                )
                self.logger.info(
                    f"Successfully saved weights to {temp_ckpt_path}"
                )

                # Only the master process interacts with the inference client
                if self.dist_manager.is_master:
                    try:
                        self.logger.info(
                            f"Master process updating inference engine from {temp_ckpt_path}..."
                        )
                        self.inference_client.resume_memory()
                        self.inference_client.update_weights(
                            model_path=temp_ckpt_path
                        )
                        self.logger.info(
                            'Inference engine weights updated successfully by master.'
                        )
                    except Exception as client_e:
                        self.logger.error(
                            f"Failed to update inference engine weights via client: {client_e}",
                            exc_info=True,
                        )
                        raise RuntimeError(
                            f"Failed to update inference engine weights: {client_e}"
                        ) from client_e

        except Exception as e:
            # Catch errors during saving, client update, or temp dir creation/cleanup
            self.logger.error(
                f"Error during policy model sync: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to sync policy model weights: {e}"
            ) from e

        # Barrier to ensure all processes wait until the master (if applicable)
        # has finished its work within the 'with' block or an error occurred.
        self.logger.debug('Reached barrier after sync attempt.')
        self.dist_manager.barrier()
        self.logger.debug('Passed barrier after sync attempt.')

        self.logger.info('Policy model sync process finished.')

    def is_inference_engine_enabled(self) -> bool:
        """Checks if inference engine is enabled."""
        return (
            hasattr(self, 'inference_client')
            and self.inference_client is not None
            and isinstance(self.inference_client, InferenceClient)
        )

    def is_cohost_inference_engine(self) -> bool:
        """Checks should release inference server memory"""
        return (
            self.is_inference_engine_enabled()
            and self.inference_client.is_cohost_mode()
        )

    def log_batch_episodes(
        self,
        phase: str,
        samples: List[EpisodeData],
        step: int,
    ) -> None:
        """
        Logs batch samples and token statistics.
        """
        if phase not in self._log_phases:
            raise ValueError(
                f"Invalid phase: {phase}, expected one of {self._log_phases}"
            )

        metric_key = f"objective/{phase}"
        token_metric_key = f"tokens/{phase}"

        for ep in samples:
            # Save data to external file
            data_to_log = {
                'rank': self.dist_manager.global_rank,
                'prompt_text': ep.prompt_text,
                'prompt_length': ep.prompt_length,
                'completion_length': ep.completion_length,
                'completion_text': ep.completion_text,
                'timestamp': ep.timestamp,
                **ep.reward_dict,
            }
            if ep.raw_data and 'ground_truth' in ep.raw_data:
                data_to_log['ground_truth'] = ep.raw_data['ground_truth']
            self.logger.log_sample(phase, data_to_log, step)

            # Logging metrics
            for k, v in ep.reward_dict.items():
                self.logger.log_scalar(f"{metric_key}/{k}", v)
            self.logger.log_scalar(
                f"{token_metric_key}/completion_length", ep.completion_length
            )
            self.logger.log_scalar(
                f"{token_metric_key}/prompt_length", ep.prompt_length
            )

    def save_checkpoint(self, step: int):
        """Saves a model checkpoint."""
        tag = f"iteration_{step}"
        save_path = os.path.join(self.checkpoint_dir, tag)
        self.logger.info(f"Saving checkpoint to {save_path}...")
        self.policy_engine.save_checkpoint(save_path)
        self.dist_manager.barrier()
        self.logger.info('Checkpoint saved.')

    def train(self, job_config: Dict):
        """Main training loop."""
        self.logger.info('Starting training...')
        if job_config and self.dist_manager.is_master:
            self.logger.log_hyperparams(job_config)

        self.dist_manager.synchronize()

        if self.config.eval_interval > 0 and self.eval_env:
            self.logger.info('Running initial evaluation...')
            with self.logger.timer('evaluate'):
                self._prepare_for_generation()
                self.evaluate_step()
                self.clean_up()
            self.dist_manager.barrier()

        while self.global_step < self.config.max_steps:
            with self.logger.timer('global_step'):
                # Run rollouts on train env
                with self.logger.timer('generation'):
                    self._prepare_for_generation()
                    local_experience = self.generate_experience()
                    self.clean_up()
                self.dist_manager.barrier()

                # Turn episode data into training samples
                with self.logger.timer('post_processing'):
                    self._prepare_for_post_processing()
                    train_dataloader = self.build_train_loader(local_experience)
                    self.clean_up()
                self.dist_manager.barrier()

                # Train the model(s)
                with self.logger.timer('train'):
                    self._prepare_for_training()
                    self.train_step(train_dataloader)
                    self.clean_up()
                self.dist_manager.barrier()

                # Handle post training tasks
                with self.logger.timer('sync_policy_model'):
                    self.sync_policy_model()
                self.dist_manager.barrier()

                if (
                    self.reference_model
                    and self.config.sync_reference_interval > 0
                    and (self.global_step + 1)
                    % self.config.sync_reference_interval
                    == 0
                ):
                    with self.logger.timer('_sync_reference_model'):
                        self._sync_reference_model()
                    self.dist_manager.barrier()

                if (
                    self.config.checkpoint_interval > 0
                    and (self.global_step + 1) % self.config.checkpoint_interval
                    == 0
                ):
                    with self.logger.timer('checkpoint'):
                        self.save_checkpoint(self.global_step + 1)

                if (
                    self.config.eval_interval > 0
                    and self.eval_env
                    and (self.global_step + 1) % self.config.eval_interval == 0
                ):
                    with self.logger.timer('evaluate'):
                        self._prepare_for_generation()
                        self.evaluate_step()
                        self.clean_up()
                    self.dist_manager.barrier()

            # Aggregate metrics and logging at end of each global step
            self.logger.aggregate_and_log(self.global_step)
            del local_experience, train_dataloader
            self.clean_up()

            self.global_step += 1

        self.logger.info('Training finished.')
        self.on_exit()

    def on_exit(self):
        """Final clean-up and checkpoint save."""
        self.save_checkpoint(self.config.max_steps)
        self.logger.close()
