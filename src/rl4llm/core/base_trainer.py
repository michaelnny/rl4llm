"""
Base RL trainer for LLMs in distributed training setup.

This module defines an abstract `BaseRLTrainer` class that provides the foundational
infrastructure for reinforcement learning algorithms with LLM environments,
including training loop, checkpointing, model sync, and logging.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
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
from rl4llm.core.training_mixin import TrainingMixin
from rl4llm.logging import LoggingManager


class BaseRLConfig(BaseModel):
    """Basic config for RL fine-tuning for LLM"""

    """For RL sample generation"""
    max_prompt_tokens: Optional[int] = Field(
        1024,
        ge=20,
        le=10240,
        description='Skip sample with prompt length greater than this to avoid peak memory spikes',
    )
    max_completion_tokens: Optional[int] = Field(
        4096, ge=10, description='Maximum number of new tokens to generate'
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
    normalize_rewards: bool = Field(False, description='Normalized rewards')
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
        if values.normalize_advantages and values.train_micro_batch_size < 4:
            raise ValueError(
                'Mini batch size must be at least 4 when normalize advantages is True'
            )
        return values

    class Config:
        arbitrary_types_allowed = True


class BaseRLTrainer(ABC, TrainingMixin, DeepSpeedUtilsMixin):
    """
    Base class for training RL algorithms on LLM environments using DeepSpeed.
    """

    _train_phase: str = TRAIN_PHASE
    _eval_phase: str = EVAL_PHASE
    _log_phases: List[str] = LOGGING_PHASES

    def __init__(
        self,
        config: BaseRLConfig,
        tokenizer: PreTrainedTokenizer,
        policy_engine: DeepSpeedEngine,
        log_config: Dict[str, Any],
        train_env: BaseEnv,
        eval_env: Optional[BaseEnv] = None,
        ref_model: Optional[Union[PreTrainedModel, DeepSpeedEngine]] = None,
        value_engine: Optional[DeepSpeedEngine] = None,
        inference_client: Optional[InferenceClient] = None,
        seed: Optional[int] = 175,
        **kwargs: Any,
    ):
        """Initialize the base trainer with DistributedOps instance"""

        TrainingMixin.__init__(self)

        self.config = config
        self.tokenizer = tokenizer
        self.policy_engine = policy_engine
        self.value_engine = value_engine
        self.logger = LoggingManager(self.dist_ops, **log_config)
        self.train_env = train_env
        self.eval_env = eval_env
        self.output_dir = log_config.get('output_dir')
        self.seed = seed
        self.device = self.dist_ops.device
        self._torch_dtype = None

        if not self.output_dir:
            raise ValueError('Invalid output_dir for logging')

        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        if self.dist_ops.is_master:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.dist_ops.barrier()

        self.reference_model: Optional[
            Union[PreTrainedModel, DeepSpeedEngine]
        ] = ref_model

        self.inference_client = inference_client

        self.global_step = 0
        self.policy_update_count = 0
        self.value_update_count = 0
        self.ref_update_count = 0

        self.called_release_inference_memory = None

        self.initialize_trainer()

        if self.is_inference_engine_enabled():
            self.logger.info('Using inference engine client.')

        self.logger.info('RL Trainer initialized.')

    @property
    def torch_dtype(self) -> torch.dtype:
        """Detects torch runtime data type from deepspeed engine"""
        if self._torch_dtype is None:
            if self.policy_engine is not None:
                self._torch_dtype = self.get_torch_dtype(self.policy_engine)
            elif self.value_engine is not None:
                self._torch_dtype = self.get_torch_dtype(self.value_engine)
            else:
                raise RuntimeError('Can not detect torch dtype')
        return self._torch_dtype

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

    @abstractmethod
    def save_checkpoint(self, tag: str) -> None:
        """Saves model checkpoint"""
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
            with self.with_unwrapped_model(self.policy_engine) as policy_model:
                yield policy_model
        self.clean_up()

    def post_step(self):
        """Post ops after a global step is done"""

        self.logger.aggregate_and_log(self.global_step)
        self.global_step += 1

    def _prepare_for_generation(self):
        """Free up GPU memory and switch models to eval mode for rollout."""

        self._configure_model(self.reference_model, 'cpu', mode='eval')
        if self.is_cohost_inference_engine():
            self._configure_model(
                self.policy_engine, 'cpu', state_action='offload'
            )
        else:
            self._configure_model(
                self.policy_engine,
                self.device,
                state_action='offload',
                mode='eval',
            )
        self._configure_model(self.value_engine, 'cpu', state_action='offload')
        self.clean_up()

    def _prepare_for_pre_processing(self):
        """Switch models handle pre-processing (like compute logprobs) before training."""

        self._release_inference_memory()
        self.clean_up()

        self._configure_model(
            self.policy_engine, self.device, state_action='offload', mode='eval'
        )
        self._configure_model(
            self.value_engine, self.device, state_action='offload', mode='eval'
        )
        self._configure_model(self.reference_model, self.device, mode='eval')

    def _prepare_for_training(self):
        """Switch models to train mode."""

        self._release_inference_memory()
        self._configure_model(self.reference_model, 'cpu')
        self.clean_up()
        self._configure_model(
            self.policy_engine, self.device, state_action='reload', mode='train'
        )
        self._configure_model(
            self.value_engine, self.device, state_action='reload', mode='train'
        )

    def _configure_model(
        self,
        model: Union[PreTrainedModel, DeepSpeedEngine],
        device: torch.device,
        state_action: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        """
        Configure a model by moving it to a specified device,
        managing its optimizer states (deepspeed engine), and setting its train/eval mode.

        Args:
            model (Union[PreTrainedModel, DeepSpeedEngine]): The model to configure, or None if no configuration is needed.
            device (str): The target device to move the model to (e.g., 'cpu', 'cuda').
            state_action (Optional[str], optional): Action to perform on deepspeed engine states, if supported.
                Can be 'offload' to offload states or 'reload' to reload states. Defaults to None.
            mode (Optional[str], optional): Mode to set the model to. Can be 'eval' for evaluation
                or 'train' for training. Defaults to None, in which case no mode is set.

        Returns:
            None
        """
        if state_action is not None and state_action not in [
            'offload',
            'reload',
        ]:
            raise ValueError(f"Invalid state_action {state_action}")
        if mode is not None and mode not in ['train', 'eval']:
            raise ValueError(f"Invalid mode {mode}")

        if model is not None:
            model = model.to(device)
            if self.can_offload_state(model) and state_action:
                if state_action == 'offload':
                    model.offload_states()
                elif state_action == 'reload':
                    model.reload_states()
            if mode == 'eval':
                model.eval()
            elif mode == 'train':
                model.train()

    def _release_inference_memory(self):
        """Try to release inference memory in co-hosting mode"""
        if (
            not self.is_cohost_inference_engine()
            or self.called_release_inference_memory
        ):
            return

        try:
            self.inference_client.release_memory()
            self.called_release_inference_memory = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to release inference engine memory, error: {str(e)}"
            )

    def sync_reference_model(self):
        """Copies current policy weights to reference model."""
        if not self.reference_model:
            return
        self.logger.info('Syncing reference model...')

        # Ensure models are on same device
        self.reference_model = self.reference_model.to(self.device)
        with self.with_unwrapped_model(self.policy_engine) as policy_model:
            # the policy state is already gathered with the unwrap context
            policy_state_dict = policy_model.state_dict()

            if isinstance(self.reference_model, DeepSpeedEngine):
                # For Zero-3, we need to handle parameter gathering
                if self.is_zero3_enabled(self.reference_model):
                    with deepspeed.zero.GatheredParameters(
                        self.reference_model.parameters()
                    ):
                        if self.dist_ops.is_master:
                            self.reference_model.module.load_state_dict(
                                policy_state_dict
                            )
                else:
                    # Reference model is DeepSpeed engine but not Zero-3
                    self.reference_model.module.load_state_dict(
                        policy_state_dict
                    )
            else:
                # Reference model is DeepSpeed engine but not Zero-3
                self.reference_model.load_state_dict(policy_state_dict)

        self.reference_model = self.reference_model.to('cpu')
        self.ref_update_count += 1
        self.logger.log_scalar('train/reference_update', self.ref_update_count)
        self.logger.debug('Reached barrier after sync attempt.')
        self.dist_ops.barrier()
        self.logger.debug('Passed barrier after sync attempt.')

        self.logger.info('Reference model sync process finished.')

    def sync_policy_model(self) -> None:
        """Update policy model weights with the inference engine."""

        if not self.is_inference_engine_enabled():
            self.logger.info(
                'Inference engine not enabled, skipping weight sync.'
            )
            return

        try:
            # We first save the full weights to a shared file system
            # then call the inference server to load the weights
            with tempfile.TemporaryDirectory(
                prefix=f"ckpt_sync_{self.global_step}_", dir=self.checkpoint_dir
            ) as temp_ckpt_path:

                self.logger.debug(
                    f"Attempting to save weights to {temp_ckpt_path}"
                )
                # Free up model
                self._configure_model(
                    self.value_engine,
                    'cpu',
                    state_action='offload',
                    mode='eval',
                )
                self.save_weights_hf_pretrained(
                    self.policy_engine, temp_ckpt_path
                )
                self.logger.debug(
                    f"Successfully saved weights to {temp_ckpt_path}"
                )
                self.clean_up()

                # Only the master process interacts with the inference client
                if self.dist_ops.is_master:
                    try:
                        self.logger.debug(
                            f"Master process updating inference engine from {temp_ckpt_path}..."
                        )
                        # Important, make sure inference engine is on CUDA before update the weights
                        self.inference_client.resume_memory()
                        self.inference_client.update_weights_from_file(
                            model_path=temp_ckpt_path
                        )
                        self.called_release_inference_memory = False
                        self.logger.debug(
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

                # Barrier to ensure all processes wait until the master (if applicable)
                # has finished its work within the 'with' block or an error occurred.
                self.logger.debug('Reached barrier after sync attempt.')
                self.dist_ops.barrier()
                self.logger.debug('Passed barrier after sync attempt.')

                self.logger.info('Policy model sync process finished.')

        except Exception as e:
            # Catch errors during saving, client update, or temp dir creation/cleanup
            self.logger.error(
                f"Error during policy model sync: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to sync policy model weights: {e}"
            ) from e

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
                'rank': self.dist_ops.global_rank,
                **ep.metadata.model_dump(),
            }
            self.logger.log_sample(phase, data_to_log, step)

            # Logging metrics
            for k, v in ep.metadata.reward_dict.items():
                self.logger.log_scalar(f"{metric_key}/{k}", v)
            self.logger.log_scalar(
                f"{token_metric_key}/completion_length",
                ep.metadata.completion_length,
            )
            self.logger.log_scalar(
                f"{token_metric_key}/prompt_length", ep.metadata.prompt_length
            )

    def extract_state_action_sequences(
        self, episodes: List[EpisodeData]
    ) -> Tuple[List[int], List[List[int]], List[List[int]]]:
        """Extract state and action sequences from the episode list"""

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

        return sequence_lengths, state_sequences, action_sequences

    def train(self, job_config: Dict):
        """
        Executes the main training loop.

        This method orchestrates the full RL training process, including rollout generation,
        data preprocessing, model training, evaluation, synchronization, and checkpointing.

        It continues until `self.global_step` reaches `self.config.max_steps`.

        Args:
            job_config (Dict): Configuration dictionary containing job parameters such as
                hyperparameters and other metadata useful for logging or reproducibility.
        """
        self.logger.info('Initializing training loop...')
        if job_config and self.dist_ops.is_master:
            self.logger.log_hyperparams(job_config)

        self.dist_ops.synchronize()

        if self.config.eval_interval > 0 and self.eval_env:
            self.logger.info('Running initial evaluation before training...')
            with self.logger.timer('evaluate_step'):
                self._prepare_for_generation()
                self.evaluate_step()
                self.clean_up()
            self.dist_ops.barrier()

        while self.global_step < self.config.max_steps:
            with self.logger.timer('global_step'):
                self.logger.info(
                    f'Step {self.global_step}: Running rollout on training environment...'
                )
                with self.logger.timer('generate_experience'):
                    self._prepare_for_generation()
                    local_experience = self.generate_experience()
                    self.clean_up()
                self.dist_ops.barrier()

                self.logger.info(
                    'Preprocessing collected experience for training...'
                )
                with self.logger.timer('pre_processing'):
                    self._prepare_for_pre_processing()
                    train_dataloader = self.build_train_loader(local_experience)
                    self.clean_up()
                self.dist_ops.barrier()

                self.logger.info('Starting train model...')
                with self.logger.timer('train_step'):
                    self._prepare_for_training()
                    self.train_step(train_dataloader)
                    self.clean_up()
                self.dist_ops.barrier()

                self.logger.info('Synchronizing policy model...')
                with self.logger.timer('sync_policy_model'):
                    self.sync_policy_model()
                self.dist_ops.barrier()

                if (
                    self.reference_model
                    and self.config.sync_reference_interval > 0
                    and (self.global_step + 1)
                    % self.config.sync_reference_interval
                    == 0
                ):
                    self.logger.info('Synchronizing reference model...')
                    with self.logger.timer('sync_reference_model'):
                        self.sync_reference_model()
                    self.dist_ops.barrier()

                if (
                    self.config.checkpoint_interval > 0
                    and (self.global_step + 1) % self.config.checkpoint_interval
                    == 0
                ):
                    self.logger.info('Saving checkpoint ...')
                    with self.logger.timer('checkpoint'):
                        self.save_checkpoint(self.global_step + 1)

                if (
                    self.config.eval_interval > 0
                    and self.eval_env
                    and (self.global_step + 1) % self.config.eval_interval == 0
                ):
                    self.logger.info('Running evaluation ...')
                    with self.logger.timer('evaluate_step'):
                        self._prepare_for_generation()
                        self.evaluate_step()
                        self.clean_up()
                    self.dist_ops.barrier()

            del local_experience, train_dataloader
            self.clean_up()

            self.post_step()

        self.logger.info('Training loop complete. Finalizing...')
        self.on_exit()

    def on_exit(self):
        """Final clean-up and checkpoint save."""
        self.save_checkpoint('last')
        self.logger.close()
