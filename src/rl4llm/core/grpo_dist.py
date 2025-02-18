"""Implements RL GRPO algorithm to train LLM on multiple GPUs using DeepSpeed"""

import logging
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Union

import deepspeed
import torch
import torch.distributed as dist
import yaml
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.graders import format_structure_grader, math_problem_grader
from rl4llm.utils import (
    DummyLogger,
    MetricsCollector,
    gather_tensor,
    masked_mean,
    masked_sum,
    masked_whiten,
    save_yaml_config_file,
)

from .data_types import GRPOConfig, GRPOSample
from .grpo import GRPOTrainer

logger = logging.getLogger(__name__)


class DistGRPOTrainer(GRPOTrainer):
    """RL GRPO for training LLMs on multiple GPUs"""

    def __init__(
        self,
        config: GRPOConfig,
        policy_engine: DeepSpeedEngine,
        reference_engine: DeepSpeedEngine,
        tokenizer: PreTrainedTokenizer,
        train_ds: Dataset,
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: logging.Logger = None,
    ):

        assert policy_engine.zero_optimization_stage() <= 2, 'Zero-3 is not supported yet'

        self.world_size = int(os.environ['WORLD_SIZE'])  # dist.get_world_size()
        self.global_rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])  # dist.get_rank()

        self.policy_engine = policy_engine
        self.reference_engine = reference_engine

        super.__init__(
            config=config,
            policy_model=self.policy_engine.module,
            tokenizer=tokenizer,
            optimizer=self.policy_engine.optimizer,
            scheduler=self.policy_engine.scheduler,
            train_ds=train_ds,
            device=device,
            torch_dtype=torch_dtype,
            artifacts_path=artifacts_path,
            logger=logger,
        )

    def is_master(self) -> bool:
        """Returns true if this is the global master rank (rank=0)"""
        return self.global_rank == 0

    def is_zero3_enabled(self) -> bool:
        """Returns true if Zero-3 is enabled"""
        return self.policy_engine.zero_optimization_stage() == 3

    def _initialize(self):
        """Setup logging and checkpoint directories"""
        self.tb_log_dir = os.path.join(self.artifacts_path, 'tb_logs')
        self.checkpoint_dir = os.path.join(self.artifacts_path, 'checkpoints')

        if self.is_master():
            for path in [self.tb_log_dir, self.checkpoint_dir]:
                os.makedirs(path, exist_ok=True)
            logger.info(f"Artifacts will be saved at: {self.artifacts_path!r}")
            self.writer = SummaryWriter(self.tb_log_dir)
        else:
            self.writer = None

        self.metrics = MetricsCollector()

        self.generation_mode = False
        self.explore_epsilon = 0
        self.episode_count = 0
        self.update_count = 0
        self.iteration_count = 0
        self.ref_update_count = 0

    def train(self, hyper_params: Optional[Dict] = None):
        """Start to train the model using RL GRPO.

        Args:
            hyper_params (Dict): Hyper parameters for the training job.
        """

        # log the params we use for this training run
        if hyper_params and self.is_master():
            save_yaml_config_file(hyper_params, os.path.join(self.artifacts_path, 'hyper_params.yaml'))
            self._log_hyper_params_to_tensorboard(hyper_params)

        for _ in tqdm(range(self.config.max_steps), desc='Training steps', disable=not self.is_master()):
            self.run_one_train_iteration()

    def run_one_train_iteration(self) -> None:
        """
        Runs one iteration of the RL GRPO algorithm.

        This method performs the following steps:
        1. Samples a batch of data from the training dataset.
        2. For each data point, generates a group of outcomes using the current policy.
        3. Computes the reward for each outcome using a verifier function.
        4. Updates the policy model using the collected samples.
        5. Logs the iteration statistics.
        6. Handles any post-training operations. Include checkpoint and optionally updates the reference policy.
        """

        self.metrics.reset()

        with self.metrics.timer('step'):
            dist.barrier()
            samples = self.generate_samples()
            dist.barrier()
            with torch.autograd.set_detect_anomaly(True):
                self.train_policy(samples)

        self.iteration_count += 1

        # Log all metrics
        metrics = self._get_metrics_summary()
        if self.is_master():
            self._log_stats_to_tensorboard(metrics, self.iteration_count)

        self._handle_post_train()

        dist.barrier()

    def train_policy(self, samples: List[GRPOSample]) -> None:
        """Train the policy model using the collected samples."""

        random.shuffle(samples)
        data_loader = DataLoader(
            samples,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self._train_collate_fn,
            drop_last=True,
        )

        # TODO adapt to deepspeed
        self.policy_engine.optimizer.zero_grad()

        assert self.policy_engine.training

        dist.barrier()

        with self.metrics.timer('train'):
            for _ in range(self.config.num_updates):
                for mini_batch in data_loader:
                    self._train_one_batch(mini_batch)
                    if self.policy_engine.is_gradient_accumulation_boundary():
                        self.update_count += 1

        self.metrics.add_metric('elapsed/policy_update', self.update_count)
        self.metrics.add_metric('elapsed/reference_update', self.ref_update_count)
        self.metrics.add_metric('train/learning_rate', self.policy_engine.optimizer.param_groups[0]['lr'])

    def save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        # TODO what about zero-3???
        if self.is_zero3_enabled():
            self.logger.info('Saving policy model checkpoint using DeepSpeed...')

        else:
            self.policy_model.save_pretrained(save_dir)

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        # gather metrics from all ranks
        metrics = {}
        local_metrics = self.metrics.get_metrics()
        for k, v in local_metrics.items():
            values = gather_tensor(torch.tensor(v, dtype=self.torch_dtype, device=self.device))
            metrics[k] = values.mean().item()
            if len(values) > self.world_size and 'loss' not in k:  # Add std dev and variance for multiple values
                metrics[f"{k}_std"] = values.std().item()
                metrics[f"{k}_var"] = values.var().item()

        return metrics

    def _prepare_for_generation(self):
        """Move unnecessary components to CPU during generation"""
        if self.generation_mode:
            return

        self.policy_engine.eval()
        self.reference_engine.eval()
        # Ensure both models are on GPU for generation
        self.policy_engine = self.policy_engine.to(self.device)
        self.reference_engine = self.reference_engine.to(self.device)

        # Clear gradients to free memory
        self.policy_engine.optimizer.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        self.generation_mode = True

    def _prepare_for_training(self):
        """Restore components for training"""
        if not self.generation_mode:
            return

        # Move reference model to CPU since it's not needed during training
        self.reference_engine = self.reference_engine.cpu()

        # Ensure policy model is on GPU for training
        self.policy_engine = self.policy_engine.to(self.device)

        self.policy_engine.train()

        torch.cuda.empty_cache()
        self.generation_mode = False

    def _train_one_batch(self, batch: GRPOSample) -> None:
        """Process a single training batch

        Args:
            batch (GRPOSample): A batch of samples
        """
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)

        pi_logprobs = self._compute_action_logprobs(self.policy_engine, states, actions)

        behavior_logprobs = batch.pi_logprobs.to(self.device)
        advantages = batch.advantages.to(self.device)
        loss_mask = batch.loss_mask.to(self.device)
        ref_logprobs = batch.ref_logprobs.to(self.device)
        advantages = batch.advantages.to(self.device)
        behavior_logprobs = batch.pi_logprobs.to(self.device)
        ref_logprobs = batch.ref_logprobs.to(self.device)
        loss_mask = batch.loss_mask.to(self.device)

        # Compute the KL divergence between the model and the reference model
        # per_token_kl = torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1

        # Clamp log differences for stability
        per_token_log_ratio = torch.clamp(ref_logprobs - pi_logprobs, min=-20, max=20)
        per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1.0
        # per_token_kl = torch.clamp(per_token_kl, min=-100.0, max=100.0)  # Prevent extreme large values

        if self.config.normalize_advantages:
            advantages = masked_whiten(advantages, loss_mask)

        # PPO clipped surrogate PG loss
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
        pg_losses = -torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

        pg_loss = masked_mean(pg_losses, loss_mask, dim=1).mean()
        kl = masked_mean(per_token_kl, loss_mask, dim=1).mean()

        # Only compute KL loss for incorrect samples
        # correctness_mask = (mini_batch.reward >= 1).to(self.device)  # Correct if reward >= 1
        # incorrect_mask = ~correctness_mask
        # kl_loss = self.config.kl_loss_coef * (kl * incorrect_mask).mean()

        kl_loss = self.config.kl_loss_coef * kl
        loss = pg_loss + kl_loss
        self.policy_engine.backward(loss)
        self.policy_engine.step()

        # These metrics will later be accumulated over mini batches
        self.metrics.add_metric('train/total_loss', loss.detach().item())
        self.metrics.add_metric('train/pg_loss', pg_loss.detach().item())
        self.metrics.add_metric('train/kl_loss', kl_loss.detach().item())
        self.metrics.add_metric('train/kl', kl.detach().item())

    def _handle_post_train(self):
        """Handle post-training operations"""
        if self.iteration_count < 1:
            return

        if self.iteration_count % self.config.sync_reference_interval == 0:
            logger.info('Updating reference model...')
            self._sync_reference_model()
            dist.barrier()

        if self.iteration_count % self.config.checkpoint_interval == 0:
            if self.is_master():
                logger.info('Saving policy model checkpoint...')
                save_dir = os.path.join(self.checkpoint_dir, f"iteration_{self.iteration_count}")
                self.save_checkpoint(save_dir)

            dist.barrier()

    def _create_deepspeed_inference_engine(
        self,
        model: PreTrainedModel,
    ) -> deepspeed.InferenceEngine:
        """Creates DeepSpeed inference engine."""
        if self.logger:
            self.logger.info('Creating inference engine...')
        tp_size = dist.get_world_size() if self.is_zero3_enabled() else 1
        ds_infer_config = {
            'tensor_parallel': {'tp_size': tp_size},
            'dtype': self.torch_dtype,
            'replace_with_kernel_inject': True,
            # "use_triton": True,
            'max_out_tokens': self.tokenizer.model_max_length,
        }

        inference_engine: deepspeed.InferenceEngine = None
        inference_engine = deepspeed.init_inference(
            model=model,
            config=ds_infer_config,
            # base_dir="/dev/shm",
            checkpoint=None,
        )

        return inference_engine

    def _sync_reference_model(self):
        """Sync reference model by copying latest policy model weights"""
        # TODO handle zero-3???

        if self.is_zero3_enabled():
            raise NotImplementedError('Zero-3 is not supported yet')
        else:
            self.reference_model.load_state_dict(self.policy_model.state_dict())
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model = self.reference_model.eval()
            self.ref_update_count += 1
            torch.cuda.empty_cache()
