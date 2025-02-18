"""Implements RL GRPO algorithm to train LLM"""

import logging
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.utils import masked_mean, masked_sum, masked_whiten, save_yaml_config_file

from .data_types import GRPOConfig, GRPOSample
from .base_grpo import BaseGRPOTrainer


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
            is_master=True,  # always set to True on single GPU
        )

        self.policy_model = policy_model.to(device)
        self.reference_model = self._create_reference_model()
        self.reference_model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.llm_generator = CustomLLMGenerator(self.policy_model)

        # self._train_collate_fn = partial(self._collate_function, pad_token_id=self.pad_token_id, torch_dtype=self.torch_dtype)

        # we only sample one item at a time for training
        self.train_ds = train_ds.shuffle(seed=None)
        self.train_iter = iter(self.train_ds)

        self.test_ds = test_ds
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )

    def train(self, hyper_params: Optional[Dict] = None):
        """Start to train the model using RL GRPO.

        Args:
            hyper_params (Dict): Hyper parameters for the training job.
        """

        # log the params we use for this training run
        if hyper_params:
            save_yaml_config_file(hyper_params, os.path.join(self.artifacts_path, 'hyper_params.yaml'))
            self._log_hyper_params_to_tensorboard(hyper_params)

        for _ in tqdm(range(self.config.max_steps), desc='Training steps'):
            self.run_one_train_iteration()

            if self.iteration_count > 1 and self.iteration_count % self.config.eval_interval == 0:
                self.run_evaluation()

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
            samples = self.generate_samples()
            with torch.autograd.set_detect_anomaly(True):
                self.train_policy(samples)

        self.iteration_count += 1

        # Log all metrics
        metrics = self._get_metrics_summary()
        self._log_stats_to_tensorboard(metrics, self.iteration_count)

        self._handle_post_train()

    def run_evaluation(self):
        """Evaluate the model on the test dataset"""

        with self._generation_context(is_training=False):
            system_prompt = self.config.system_prompt

            # use greedy sampling for evaluation
            eval_kwargs = {
                'eos_token_id': self.eos_token_id,
                'pad_token_id': self.pad_token_id,
                'max_new_tokens': self.config.max_new_tokens,
                'temperature': 0.0,
                'do_sample': False,
                'use_cache': True,
                'output_scores': False,
                'output_logits': False,
                'return_dict_in_generate': True,
                'return_legacy_cache': False,
            }

            total_samples = 0
            total_correct = 0

            with self.metrics.timer('evaluation'):
                for batch in self.test_loader:
                    questions = batch['question']
                    ground_truths = batch['ground_truth']
                    task_types = batch['task_type']

                    batch_messages = []
                    for question in questions:
                        if not system_prompt:
                            sample_message = [{'role': 'user', 'content': question.strip()}]
                        else:
                            sample_message = [
                                {'role': 'system', 'content': system_prompt.strip()},
                                {'role': 'user', 'content': question.strip()},
                            ]
                        batch_messages.append(sample_message)

                    # Tokenize single message
                    message_prompt = self.tokenizer.apply_chat_template(
                        batch_messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.tokenizer(
                        message_prompt,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        padding_side='left',
                        max_length=self.tokenizer.model_max_length,
                    ).to(self.device)

                    outputs = self.policy_model.generate(**inputs, **eval_kwargs)

                    reward_dict = self._process_generation_outputs(
                        questions, ground_truths, task_types, inputs.input_ids, outputs.sequences
                    )

                    total_samples += len(questions)
                    total_correct += (reward_dict['accuracy_reward'] == 1.0).sum().item()

            accuracy = total_correct / total_samples
            self.logger.info(f"Evaluation accuracy: {accuracy:.4f}")

    def generate_samples(
        self,
    ) -> List[GRPOSample]:
        """Generates samples using the current policy."""

        with self._generation_context(is_training=True):
            assert not self.policy_model.training
            assert not self.reference_model.training
            collected_samples: List[GRPOSample] = []

            with self.metrics.timer('generation'):
                while len(collected_samples) < self.config.rollout_size:
                    sample = self._get_next_data_item()
                    samples = self.generate_group_samples(
                        sample,
                        policy_model=self.policy_model,
                        reference_model=self.reference_model,
                        generator=self.llm_generator,
                    )
                    collected_samples.extend(samples)

            self.metrics.add_metric('elapsed/generation_episodes', self.train_episode_count)
            self.metrics.add_metric('elapsed/explore_epsilon', self.explore_epsilon)

            return collected_samples

    def train_policy(self, train_samples: List[GRPOSample]) -> None:
        """Train the policy model using the collected samples."""

        random.shuffle(train_samples)
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

        with self.metrics.timer('train'):
            for _ in range(self.config.num_updates):
                for mini_batch in data_loader:
                    self._train_one_batch(mini_batch)
                    mini_steps += 1

                    if mini_steps % self.config.gradient_accumulate_steps == 0:
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
        self.metrics.add_metric('elapsed/policy_update', self.update_count)
        self.metrics.add_metric('elapsed/reference_update', self.ref_update_count)
        self.metrics.add_metric('train/learning_rate', self.optimizer.param_groups[0]['lr'])

    @contextmanager
    def _generation_context(self, is_training: bool = True):
        """Context manager for handling model and optimizer states during generation"""
        try:
            self._prepare_for_generation(is_training)
            yield
        finally:
            if is_training:
                self._prepare_for_training()

    def save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        self.policy_model.save_pretrained(save_dir)

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return self.metrics.get_summary()

    def _prepare_for_generation(self, is_training: bool = True):
        """Move unnecessary components to CPU during generation"""
        if self.generation_mode:
            return

        self.policy_model = self.policy_model.eval()
        self.policy_model = self.policy_model.to(self.device)

        if is_training:
            self.reference_model = self.reference_model.eval()
            self.reference_model = self.reference_model.to(self.device)
        else:
            self.reference_model = self.reference_model.cpu()

        # Clear gradients to free memory
        self.policy_model.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        self.generation_mode = True

    def _prepare_for_training(self):
        """Restore components for training"""
        if not self.generation_mode:
            return

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

        # scaled_loss = loss / self.config.gradient_accumulate_steps
        # scaled_loss.backward()
        loss.backward()

        # These metrics will later be accumulated over mini batches
        for k, v in metrics.items():
            self.metrics.add_metric(f'train/{k}', v)

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

    def _create_reference_model(self) -> PreTrainedModel:
        """Create a reference model from the policy model"""
        ref_model = deepcopy(self.policy_model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = ref_model.eval()
        return ref_model

    def _handle_post_train(self):
        """Handle post-training operations"""
        if self.iteration_count < 1:
            return

        if self.iteration_count % self.config.sync_reference_interval == 0:
            self.logger.info('Updating reference model...')
            self._sync_reference_model()

        if self.iteration_count % self.config.checkpoint_interval == 0:
            self.logger.info('Saving policy model checkpoint...')
            save_dir = os.path.join(self.checkpoint_dir, f"iteration_{self.iteration_count}")
            self.save_checkpoint(save_dir)

    def _sync_reference_model(self):
        """Sync reference model by copying latest policy model weights"""
        self.reference_model.load_state_dict(self.policy_model.state_dict())
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model = self.reference_model.eval()
        self.ref_update_count += 1
        torch.cuda.empty_cache()
