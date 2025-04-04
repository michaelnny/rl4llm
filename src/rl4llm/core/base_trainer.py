import datetime
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
from deepspeed import DeepSpeedEngine
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.constants import EVAL_PHASE, LOGGING_PHASES, TRAIN_PHASE
from rl4llm.core.data_types import RLConfig
from rl4llm.core.distributed import DistributedManager
from rl4llm.core.training_mixin import TrainingMixin
from rl4llm.envs import EpisodeData, LLMEnv
from rl4llm.logging import LoggingManager


class RLTrainer(ABC, TrainingMixin):
    """
    Abstract base class for RL training with LLM in a distributed environment.
    Provides the core infrastructure and defines the interface for algorithm-specific components.
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
        train_env: LLMEnv,
        eval_env: Optional[LLMEnv] = None,
        seed: Optional[int] = 175,
    ):
        if policy_engine.zero_optimization_stage() == 3:
            raise RuntimeError('Zero-3 not supported')

        if config.train_rollout_size % dist_manager.world_size != 0:
            raise ValueError(
                f"Rollout size must be divisible by {dist_manager.world_size}"
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

        self.policy_model: PreTrainedModel = self.policy_engine.module
        self.device = self.dist_manager.device
        self.torch_dtype = self.policy_model.dtype  # Infer from model

        # Setup directories
        self.checkpoint_dir = os.path.join(self.artifacts_path, 'checkpoints')
        if self.dist_manager.is_master:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.dist_manager.barrier()  # Ensure dirs exist before proceeding

        # Reference model (optional, common in PPO-like methods)
        self.reference_model: Optional[PreTrainedModel] = (
            self._create_reference_model()
            if self.config.kl_loss_coef > 0
            else None
        )

        # Internal state
        self.iteration_count = 0
        self.global_step = 0
        self.policy_update_count = 0  # Tracks optimizer steps
        self.ref_update_count = 0

        self._initialize_trainer()
        self.logger.info('RL Trainer initialized.')

    def _initialize_trainer(self):
        pass

    def _create_reference_model(self) -> Optional[PreTrainedModel]:
        """Creates a non-trainable copy of the policy model."""
        self.logger.info('Creating reference model...')
        if self.is_zero3_enabled():
            # TODO maybe create reference model with deepspeed??
            raise RuntimeError('Not supported for Zero-3')

            # # Ensure model is fully available if using Zero-3
            # with deepspeed.zero.GatheredParameters(self.policy_model.parameters()):
            #     ref_model = deepcopy(self.policy_model)
        else:
            # For Zero-1/2, deepcopy might work directly, but state_dict is safer
            ref_model = type(self.policy_model)(
                self.policy_model.config
            )  # Create new instance
            ref_model.load_state_dict(self.policy_model.state_dict())
        ref_model = ref_model.to(self.device).eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        self.logger.info('Reference model created.')
        return ref_model

    def is_zero3_enabled(self) -> bool:
        """Returns true if Zero-3 is enabled"""
        return self.policy_engine.zero_optimization_stage() == 3

    def log_batch_episodes(
        self,
        phase: str,
        samples: List[EpisodeData],
        step: int,
    ) -> None:
        """Log batch samples to Tensorboard and external files"""

        if phase not in self._log_phases:
            raise ValueError(
                f"Invalid phase {phase}, expect to be [{self._log_phases}]"
            )

        metric_key = f"objective/{phase}"
        token_metric_key = f"tokens/{phase}"
        for ep in samples:
            data_to_log = {
                'rank': self.dist_manager.global_rank,
                'prompt_text': ep.prompt_text,
                'prompt_length': ep.prompt_length,
                'completion_length': ep.completion_length,
                'completion_text': ep.completion_text,
                'timestamp': ep.timestamp,
                **ep.reward_dict,
            }

            if ep.raw_data is not None and 'ground_truth' in ep.raw_data:
                data_to_log['ground_truth'] = ep.raw_data['ground_truth']

            self.logger.log_sample(phase, data_to_log, step)
            for k, v in ep.reward_dict.items():
                self.logger.log_scalar(f"{metric_key}/{k}", v)
            # Token usage
            self.logger.log_scalar(
                f"{token_metric_key}/completion_length", ep.completion_length
            )
            self.logger.log_scalar(
                f"{token_metric_key}/prompt_length", ep.prompt_length
            )

    @abstractmethod
    def generate_experience(self) -> List[Any]:
        """
        Generates experience (e.g., rollouts, trajectories) using the current policy.
        This is algorithm-specific (e.g., PPO needs states, actions, logprobs, rewards, values).
        Should handle interaction with the environment/LLM feedback.
        Returns:
            List[Any]: A list of collected experience elements (e.g., trajectories)
                       specific to the current rank.
        """
        pass

    @abstractmethod
    def compute_loss(
        self, experience_batch: Any, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the loss for a batch of experience.
        Args:
            experience_batch: A batch collated from the generated experience.
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: The computed loss tensor and a dictionary
                                                   of metrics for logging.
        """
        pass

    @abstractmethod
    def build_train_batch(self, experience: List[Any]) -> DataLoader:
        """
        Processes the generated experience and creates a DataLoader for training updates.
        Args:
            experience: The list of experience elements generated by `generate_experience`.
        Returns:
            DataLoader: DataLoader yielding batches ready for `compute_loss`.
        """
        pass

    @abstractmethod
    @torch.inference_mode()
    def evaluate_step(self) -> Dict[str, Any]:
        """
        Performs evaluation using the current policy model.
        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        pass

    @abstractmethod
    def train_step(self, train_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Performs the policy update phase.
        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        pass

    def _prepare_for_generation(self):
        """Sets models to evaluation mode for experience generation."""
        self.policy_engine.eval()
        if self.reference_model:
            self.reference_model = self.reference_model.to(self.device)
            self.reference_model.eval()  # Already in eval, but good practice
        self.clean_up()

    def _prepare_for_training(self):
        """Sets models back to training mode."""
        self.policy_engine.train()
        # Move reference model back to GPU if moved earlier and needed
        if self.reference_model and self.reference_model.device != self.device:
            self.reference_model = self.reference_model.to('cpu')
        self.clean_up()

    def save_checkpoint(self, step: int):
        """Saves model checkpoint using DeepSpeed."""
        tag = f"iteration_{step}"
        save_path = os.path.join(self.checkpoint_dir, tag)
        self.logger.info(f"Saving checkpoint to {save_path}...")
        # DeepSpeed handles distributed saving internally
        self.policy_engine.save_checkpoint(save_path)
        self.dist_manager.barrier()
        self.logger.info('Checkpoint saved.')

    def sync_reference_model(self):
        """Updates the reference model weights with the current policy model weights."""
        if not self.reference_model:
            return

        self.logger.info('Syncing reference model...')
        # Ensure parameters are gathered if using Zero-3 before state_dict access
        if self.is_zero3_enabled():
            # TODO recreate the reference model engine
            raise RuntimeError('Not supported for Zero-3')
            # with deepspeed.zero.GatheredParameters(self.policy_model.parameters(), modifier_rank=0):
            #    if self.dist_manager.is_master:
            #        state_dict = self.policy_model.state_dict()
            # state_dict = self.dist_manager.broadcast_object(state_dict, src=0) # If needed

        else:
            # For Zero-1/2, state_dict() should work directly
            state_dict = self.policy_model.state_dict()

            # Load state dict into reference model (all ranks have the state_dict now)
            self.reference_model.load_state_dict(state_dict)
            self.ref_update_count += 1
            self.logger.info('Reference model synced.')

        self.clean_up()
        self.dist_manager.barrier()

    def train(self, run_config: Dict):
        """Main training loop."""
        self.logger.info('Starting training...')
        if run_config and self.dist_manager.is_master:
            self.logger.log_hyperparams(run_config)

        self.dist_manager.synchronize()

        # Initial evaluation before training starts
        if self.config.eval_interval > 0 and self.eval_env:
            self.logger.info('Running initial evaluation...')
            with self.logger.timer('evaluate'):
                self._prepare_for_generation()
                self.evaluate_step()
                self.clean_up()
            self.dist_manager.barrier()

        for t in range(self.config.max_steps):
            with self.logger.timer('global_step'):
                self.global_step = t
                # 1. Generation Phase
                with self.logger.timer('generation'):
                    self._prepare_for_generation()
                    # generate_experience returns rank-local experience
                    local_experience = self.generate_experience()
                    self.clean_up()
                self.dist_manager.barrier()

                # 2. Learning Phase
                with self.logger.timer('train'):
                    self._prepare_for_training()
                    # build_train_batch processes local_experience and returns a DataLoader
                    train_dataloader = self.build_train_batch(local_experience)
                    self.train_step(train_dataloader)
                    self.clean_up()
                self.dist_manager.barrier()

                # 3. Post-Iteration Operations
                # Sync reference model
                if (
                    self.reference_model
                    and self.config.sync_reference_interval > 0
                    and (t + 1) % self.config.sync_reference_interval == 0
                ):
                    with self.logger.timer('sync_reference_model'):
                        self.sync_reference_model()

                # Checkpointing
                if (
                    self.config.checkpoint_interval > 0
                    and (t + 1) % self.config.checkpoint_interval == 0
                ):
                    with self.logger.timer('checkpoint'):
                        self.save_checkpoint(t + 1)

                # Evaluation
                if (
                    self.config.eval_interval > 0
                    and self.eval_env
                    and (t + 1) % self.config.eval_interval == 0
                ):
                    with self.logger.timer('evaluate'):
                        self._prepare_for_generation()
                        self.evaluate_step()
                        self.clean_up()
                    self.dist_manager.barrier()

                # Logging metrics
                self.logger.aggregate_and_log(self.global_step)

                # Clean up memory
                del local_experience
                del train_dataloader
                self.clean_up()

        self.logger.info('Training finished.')
        # Final checkpoint save
        self.save_checkpoint(self.config.max_steps)
        self.on_exit()

    def on_exit(self):
        """Handles exits like kill of process"""
        self.logger.close()
