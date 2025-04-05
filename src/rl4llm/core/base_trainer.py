"""
Base RL trainer for LLMs in distributed training setup.

This module defines an abstract `RLTrainer` class that provides the foundational
infrastructure for reinforcement learning algorithms with LLM environments,
including training loop, checkpointing, model sync, and logging.
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import deepspeed
import torch
import vllm
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.constants import EVAL_PHASE, LOGGING_PHASES, TRAIN_PHASE
from rl4llm.core.data_types import RLConfig
from rl4llm.core.distributed import DistributedManager
from rl4llm.core.training_mixin import TrainingMixin
from rl4llm.envs import EpisodeData, LLMEnv
from rl4llm.logging import LoggingManager

# @contextmanager
# def unwrap_deepspeed_model(
#     engine: deepspeed.DeepSpeedEngine, is_zero3_enabled: bool
# ) -> Generator[PreTrainedModel, None, None]:
#     """
#     Unwraps a DeepSpeed engine to yield the underlying model.

#     Args:
#         engine: The DeepSpeed engine containing the model
#         is_zero3_enabled: Whether Zero-3 optimization is enabled

#     Yields:
#         PreTrainedModel: The unwrapped model
#     """
#     if is_zero3_enabled:
#         with deepspeed.zero.GatheredParameters(engine.parameters()):
#             yield engine.module
#     else:
#         yield engine.module


class RLTrainer(ABC, TrainingMixin):
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
        train_env: LLMEnv,
        eval_env: Optional[LLMEnv] = None,
        vllm_engine: Optional[vllm.LLM] = None,
        seed: Optional[int] = 175,
    ):
        if policy_engine.zero_optimization_stage() == 3:
            raise RuntimeError('Zero-3 is not supported at the moment')
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
        self.torch_dtype = self._get_torch_dtype()

        self.checkpoint_dir = os.path.join(artifacts_path, 'checkpoints')
        if dist_manager.is_master:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        dist_manager.barrier()

        self.reference_model: Optional[PreTrainedModel] = (
            self._create_reference_model() if config.kl_loss_coef > 0 else None
        )

        self.vllm_engine = vllm_engine

        self.global_step = 0
        self.policy_update_count = 0
        self.ref_update_count = 0

        self.initialize_trainer()
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
        Converts collected experience into training batches.
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

    @abstractmethod
    def sync_policy_model(self):
        """
        Syncs policy model across distributed ranks.
        """
        pass

    @contextmanager
    def unwrapped_model_for_generation(
        self,
    ) -> Generator[Union[PreTrainedModel, vllm.LLM], None, None]:
        """
        Returns the unwrapped model for generation.
        """
        if self.is_vllm_inference_enabled():
            yield self.vllm_engine
        else:
            with self._unwrapped_deepspeed_model() as model:
                yield model
        self.clean_up()

    @contextmanager
    def _unwrapped_deepspeed_model(
        self,
    ) -> Generator[PreTrainedModel, None, None]:
        """Returns the unwrapped model from the DeepSpeed engine."""
        if self.is_zero3_enabled():
            with deepspeed.zero.GatheredParameters(
                self.policy_engine.parameters()
            ):
                yield self.policy_engine.module
        else:
            yield self.policy_engine.module

    def _prepare_for_generation(self):
        """Switch models to eval mode for rollout."""
        if self.vllm_engine is not None:
            try:
                # Load vLLM to CUDA
                self.vllm_engine.wake_up()
            except Exception as e:
                self.logger.warning(
                    f"Failed to wake up vLLM engine, error: {str(e)}"
                )

        self.policy_engine.eval()
        if self.reference_model:
            self.reference_model = self.reference_model.to(self.device)
            self.reference_model.eval()

        self.clean_up()

    def _prepare_for_training(self):
        """Switch models to train mode."""
        if self.vllm_engine is not None:
            try:
                # Offload vLLM and remove CUDA RAM caches
                self.vllm_engine.sleep(1)
            except Exception as e:
                self.logger.warning(
                    f"Failed to offload vLLM engine, error: {str(e)}"
                )

        self.policy_engine.train()
        if self.reference_model and self.reference_model.device != self.device:
            self.reference_model = self.reference_model.to('cpu')

        self.clean_up()

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

    def _get_torch_dtype(self) -> torch.dtype:
        """Determines appropriate torch dtype from policy engine config."""
        if self.policy_engine.bfloat16_enabled():
            return torch.bfloat16
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.float32

    def _sync_vllm_weights(self, state_dict: Dict) -> None:
        """A hacky way to update vLLM engine weights in non tensor parallel setting."""
        if self.vllm_engine is None:
            self.logger.warning('No vLLM engine specified, skipping')

        # only works with vLLM V0 engine with env variable "VLLM_USE_V1 = '0'"
        try:
            model = (
                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
            )
            model.load_weights(state_dict.items())
            self.clean_up()
        except Exception as e:
            self.logger.error(
                f"Failed to update vLLM engine weights, error: {str(e)}"
            )

    def is_zero3_enabled(self) -> bool:
        """Checks if ZeRO-3 is enabled."""
        return self.policy_engine.zero_optimization_stage() == 3

    def is_vllm_inference_enabled(self) -> bool:
        """Checks if vLLM inference mode is enabled."""
        return self.vllm_engine is not None and isinstance(
            self.vllm_engine, vllm.LLM
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
        token_metric_key = f"Tokens/{phase}"

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

            if ep.raw_data and 'ground_truth' in ep.raw_data:
                data_to_log['ground_truth'] = ep.raw_data['ground_truth']

            self.logger.log_sample(phase, data_to_log, step)

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

                with self.logger.timer('generation'):
                    self._prepare_for_generation()
                    local_experience = self.generate_experience()
                    self.clean_up()
                self.dist_manager.barrier()

                with self.logger.timer('train'):
                    self._prepare_for_training()
                    train_dataloader = self.build_train_loader(local_experience)
                    self.train_step(train_dataloader)
                    self.clean_up()
                self.dist_manager.barrier()

                with self.logger.timer('sync_policy_model'):
                    self.sync_policy_model()
                self.dist_manager.barrier()

                # Post training tasks
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
