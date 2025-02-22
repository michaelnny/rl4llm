"""Implements RL GRPO algorithm to train LLM on multiple GPUs using DeepSpeed"""

import logging
import os
import random
from contextlib import contextmanager
from typing import Any, Dict, List

import deepspeed
import torch
import torch.distributed as dist
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.utils import gather_tensor

from .base_grpo import BaseGRPOTrainer
from .data_types import GRPOConfig, GRPOSample

logger = logging.getLogger(__name__)


# TODO consider adapting to support deepspeed zero-3
class GRPOTrainer(BaseGRPOTrainer):
    """RL GRPO for training LLMs on multiple GPUs using deepspeed.

    Important: due to custom generator class, this code only supports ZeRO stage <=2
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy_engine: DeepSpeedEngine,
        reference_engine: DeepSpeedEngine,
        tokenizer: PreTrainedTokenizer,
        train_ds: Dataset,
        test_ds: Dataset,
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: logging.Logger = None,
    ):

        assert policy_engine.zero_optimization_stage() <= 2, 'Zero-3 is not supported yet'

        self.world_size = int(os.environ['WORLD_SIZE'])  # dist.get_world_size()
        self.global_rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])  # dist.get_rank()

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            artifacts_path=artifacts_path,
            logger=logger,
            rank=self.global_rank,
        )

        self.policy_engine = policy_engine
        self.reference_engine = reference_engine

        self.policy_model: PreTrainedModel = self.policy_engine.module
        self.reference_model: PreTrainedModel = self.reference_engine.module
        self.llm_generator = CustomLLMGenerator(self.policy_model)

        # we only sample one item at a time for training, so no need loader
        self.train_ds = train_ds.shuffle(seed=None)
        self.train_iter = iter(self.train_ds)

        self.test_ds = test_ds
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.config.eval_batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

    def is_zero3_enabled(self) -> bool:
        """Returns true if Zero-3 is enabled"""
        return self.policy_engine.zero_optimization_stage() == 3

    def run_one_iteration(self) -> None:
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

        self._metrics.reset()

        with self._metrics.timer('step'):
            dist.barrier()
            samples = self.generate_train_samples()
            dist.barrier()
            with torch.autograd.set_detect_anomaly(True):
                self.train_policy(samples)

        self.iteration_count += 1

        self._handle_post_train()

        # Log all metrics
        metrics = self._get_metrics_summary()
        if self.is_master:
            self._log_training_stats(metrics, self.iteration_count)

        dist.barrier()

    def run_evaluation(self):
        """Evaluate the model on the test dataset"""
        with self._generation_context(is_training=False):
            self.evaluate_policy(self.policy_model, self.test_loader)

    def generate_train_samples(
        self,
    ) -> List[GRPOSample]:
        """Generates samples using the current policy."""

        with self._generation_context(is_training=True):
            assert not self.policy_model.training
            assert not self.reference_model.training
            collected_samples: List[GRPOSample] = []

            with self._metrics.timer('generation'):
                while len(collected_samples) < self.config.rollout_size // self.world_size:
                    sample = self._get_next_data_item()
                    samples = self.generate_group_samples(
                        sample,
                        policy_model=self.policy_model,
                        reference_model=self.reference_model,
                        generator=self.llm_generator,
                    )
                    collected_samples.extend(samples)

            self._metrics.add_metric('elapsed/generation_episodes', self.train_episode_count * self.world_size)
            self._metrics.add_metric('elapsed/explore_epsilon', self.explore_epsilon)

            return collected_samples

    def train_policy(self, samples: List[GRPOSample]) -> None:
        """Train the policy model using the collected samples."""

        random.shuffle(samples)
        data_loader = DataLoader(
            samples,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self._train_collate_function,
            drop_last=True,
        )
        self.policy_engine.optimizer.zero_grad()
        assert self.policy_engine.training

        dist.barrier()

        with self._metrics.timer('train'):
            for _ in range(self.config.num_updates):
                for mini_batch in data_loader:
                    self._train_one_batch(mini_batch)
                    if self.policy_engine.is_gradient_accumulation_boundary():
                        self.update_count += 1

        self._metrics.add_metric('elapsed/policy_update', self.update_count)
        self._metrics.add_metric('elapsed/reference_update', self.ref_update_count)
        self._metrics.add_metric('training/learning_rate', self.policy_engine.optimizer.param_groups[0]['lr'])

    def save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        if self.is_zero3_enabled():
            # self.logger.info('Saving policy model checkpoint using DeepSpeed...')
            raise NotImplementedError

        else:
            self.policy_model.save_pretrained(save_dir)

    @contextmanager
    def _generation_context(self, is_training: bool = True):
        """Context manager for handling model and optimizer states during generation"""
        try:
            self._prepare_for_generation(is_training)
            yield
        finally:
            self._prepare_for_training()

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        # gather metrics from all ranks
        metrics = {}
        local_metrics = self._metrics.get_metrics()
        for k, v in local_metrics.items():
            values = gather_tensor(torch.tensor(v, dtype=self.torch_dtype, device=self.device))
            metrics[k] = values.mean().item()
            if (
                len(values) > self.world_size and 'loss' not in k and 'grad_norm' not in k
            ):  # Add std dev and variance for multiple values
                metrics[f"{k}_std"] = values.std().item()

        return metrics

    def _prepare_for_generation(self, is_training: bool = True):
        """Move unnecessary components to CPU during generation"""
        if self.generation_mode:
            return

        self.policy_engine.eval()
        self.reference_engine.eval()
        # Ensure both models are on GPU for generation
        self.policy_engine = self.policy_engine.to(self.device)
        if is_training:
            self.reference_engine = self.reference_engine.to(self.device)
        else:
            self.reference_engine = self.reference_engine.cpu()

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

        loss, metrics = self._compute_loss(pi_logprobs, batch)

        self.policy_engine.backward(loss)
        self.policy_engine.step()

        # These metrics will later be accumulated over mini batches
        for k, v in metrics.items():
            self._metrics.add_metric(f'training/{k}', v)

    def _get_next_data_item(self) -> Dict:
        """Fetches the next sample for generation, handles epoch reset.

        Returns:
            Dict: A single item containing question and ground truth
        """
        try:
            item = next(self.train_iter)
            return item
        except StopIteration:
            # Epoch finished! Reshuffle and recreate the iterator
            self.train_ds = self.train_ds.shuffle(seed=None)
            self.train_iter = iter(self.train_ds)
            item = next(self.train_iter)  # Get the first item of the new epoch
            return item

    def _handle_post_train(self):
        """Handle post-training operations"""
        if self.iteration_count < 1:
            return

        if self.iteration_count % self.config.sync_reference_interval == 0:
            logger.info('Updating reference model...')
            self._sync_reference_model()
            dist.barrier()

        if self.iteration_count % self.config.checkpoint_interval == 0:
            if self.is_master:
                logger.info('Saving policy model checkpoint...')
                save_dir = os.path.join(self._checkpoint_dir, f"iteration_{self.iteration_count}")
                self.save_checkpoint(save_dir)
            dist.barrier()

        if self.iteration_count % self.config.eval_interval == 0:
            self.logger.info('Run evaluation...')
            self.run_evaluation()
            dist.barrier()

    def _sync_reference_model(self):
        """Sync reference model by copying latest policy model weights"""

        if self.is_zero3_enabled():
            raise NotImplementedError('Zero-3 is not supported yet')
        else:
            self.reference_model.load_state_dict(self.policy_model.state_dict())
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model = self.reference_model.eval()
            self.ref_update_count += 1
            torch.cuda.empty_cache()

    # def _create_deepspeed_inference_engine(
    #     self,
    #     model: PreTrainedModel,
    # ) -> deepspeed.InferenceEngine:
    #     """Creates DeepSpeed inference engine."""
    #     if self.logger:
    #         self.logger.info('Creating inference engine...')
    #     tp_size = dist.get_world_size() if self.is_zero3_enabled() else 1
    #     ds_infer_config = {
    #         'tensor_parallel': {'tp_size': tp_size},
    #         'dtype': self.torch_dtype,
    #         'replace_with_kernel_inject': True,
    #         # "use_triton": True,
    #         'max_out_tokens': self.tokenizer.model_max_length,
    #     }

    #     inference_engine: deepspeed.InferenceEngine = None
    #     inference_engine = deepspeed.init_inference(
    #         model=model,
    #         config=ds_infer_config,
    #         # base_dir="/dev/shm",
    #         checkpoint=None,
    #     )

    #     return inference_engine
