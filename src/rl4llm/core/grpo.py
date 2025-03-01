"""Implements RL GRPO algorithm to train LLM"""

import logging
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator

from .base_grpo import BaseGRPOTrainer
from .data_types import GRPOConfig, GRPOSample


class GRPOTrainer(BaseGRPOTrainer):
    """RL GRPO for training LLMs for reasoning on math tasks on a single GPU.

    This implementation uses GRPO policy optimization with KL-divergence regularization to fine-tune
    language models through reinforcement learning.
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_ds: Dataset,
        test_ds: Dataset,
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: logging.Logger = None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            artifacts_path=artifacts_path,
            logger=logger,
            rank=0,  # always set on single GPU
        )

        self.policy_model = policy_model

        self.reference_model = self._create_reference_model(policy_model) if self.config.kl_loss_coef > 0 else None

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.llm_generator = CustomLLMGenerator(self.policy_model)

        self.logger.info('Preprocessing datasets...')
        self.train_ds = self.preprocess_dataset(train_ds)
        self.test_ds = self.preprocess_dataset(test_ds)

        # we only sample one item at a time for training, so no need loader
        self.train_iter = iter(self.train_ds)
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.config.eval_batch_size,
            collate_fn=self._eval_collate_function,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
        )

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

        if self.iteration_count == 0:
            # do an initial evaluation before apply any training
            self.run_evaluation()

        with self._metrics.timer('step'):
            samples = self.generate_train_samples()
            with torch.autograd.set_detect_anomaly(True):
                self.train_policy(samples)

        self._handle_post_train()

        # Log all metrics
        metrics = self._get_metrics_summary()
        self._log_training_stats(metrics, self.iteration_count)

        self.iteration_count += 1

    def run_evaluation(self):
        """Evaluate the model on the test dataset"""
        self.logger.info('Run evaluation...')
        with self._generation_context(is_training=False):
            self.evaluate_policy(self.policy_model, self.test_loader)

    def generate_train_samples(
        self,
    ) -> List[GRPOSample]:
        """Generates samples using the current policy."""

        with self._generation_context(is_training=True):
            assert not self.policy_model.training
            collected_samples: List[GRPOSample] = []

            with self._metrics.timer('generation'):
                while len(collected_samples) < self.config.rollout_size:
                    sample = self._get_next_data_item()
                    samples = self.generate_group_samples(
                        sample,
                        policy_model=self.policy_model,
                        reference_model=self.reference_model,
                        generator=self.llm_generator,
                    )
                    if samples:
                        collected_samples.extend(samples)

                if len(collected_samples) > self.config.rollout_size:
                    collected_samples = collected_samples[: self.config.rollout_size]

            self._metrics.add_metric('elapsed/generation_episodes', self.train_episode_count)
            self._metrics.add_metric('elapsed/explore_epsilon', self.explore_epsilon)

            return collected_samples

    def train_policy(self, train_samples: List[GRPOSample]) -> None:
        """Train the policy model using the collected samples."""

        data_loader = DataLoader(
            train_samples,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=self._train_collate_function,
            drop_last=True,
        )

        self.optimizer.zero_grad()

        assert self.policy_model.training

        mini_steps = 0

        with self._metrics.timer('train'):
            for _ in range(self.config.num_updates):
                for mini_batch in data_loader:
                    self._train_one_batch(mini_batch)
                    mini_steps += 1

                    if mini_steps % self.config.gradient_accumulate_steps == 0:
                        grad_norm = self.get_grad_norm(self.policy_model)
                        self._metrics.add_metric('training/grad_norm', grad_norm.item())
                        if self.config.clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.policy_model.parameters(), max_norm=self.config.clip_grad_norm, error_if_nonfinite=True
                            )
                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.update_count += 1
                        mini_steps = 0

        # handle any remaining gradients
        if mini_steps > 0:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.update_count += 1

        # add more metrics
        self._metrics.add_metric('elapsed/policy_update', self.update_count)
        self._metrics.add_metric('elapsed/reference_update', self.ref_update_count)
        self._metrics.add_metric('training/learning_rate', self.optimizer.param_groups[0]['lr'])

    def save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        self.logger.info('Saving policy model checkpoint...')
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
        return self._metrics.get_summary()

    def _prepare_for_generation(self, is_training: bool = True):
        """Move unnecessary components to CPU during generation"""
        if self.generation_mode:
            return

        self.policy_model = self.policy_model.eval()
        self.policy_model = self.policy_model.to(self.device)

        if self.reference_model is not None:
            to_device = self.device if is_training else torch.device('cpu')
            self.reference_model = self.reference_model.to(to_device)

        # Clear gradients to free memory
        self.policy_model.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        self.generation_mode = True

    def _prepare_for_training(self):
        """Restore components for training"""
        if not self.generation_mode:
            return

        if self.reference_model is not None:
            # Move reference model to CPU since it's not needed during training
            self.reference_model = self.reference_model.cpu()

        # Ensure policy model is on GPU for training
        self.policy_model = self.policy_model.to(self.device)

        self.policy_model = self.policy_model.train()

        torch.cuda.empty_cache()
        self.generation_mode = False

    def _train_one_batch(self, batch: GRPOSample) -> None:
        """Process a single training batch

        Args:
            batch (GRPOSample): A batch of samples
        """
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)

        pi_logprobs = self._compute_action_logprobs(self.policy_model, states, actions)

        loss, metrics = self._compute_loss(pi_logprobs, batch)

        scaled_loss = loss / self.config.gradient_accumulate_steps
        scaled_loss.backward()

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
            random.shuffle(self.train_ds)
            self.train_iter = iter(self.train_ds)
            item = next(self.train_iter)  # Get the first item of the new epoch
            return item

    def _handle_post_train(self):
        """Handle post-training operations"""
        if self.iteration_count < 1:
            return

        if self.iteration_count % self.config.sync_reference_interval == 0:
            self._sync_reference_model()

        if self.iteration_count % self.config.checkpoint_interval == 0:
            save_dir = os.path.join(self._checkpoint_dir, f"iteration_{self.iteration_count}")
            self.save_checkpoint(save_dir)

        if self.iteration_count % self.config.eval_interval == 0:
            self.run_evaluation()

    def _sync_reference_model(self):
        """Sync reference model by copying latest policy model weights"""
        if self.reference_model is None:
            return
        self.logger.info('Updating reference model...')
        self.reference_model.load_state_dict(self.policy_model.state_dict())
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model = self.reference_model.eval()
        self.ref_update_count += 1
        torch.cuda.empty_cache()
