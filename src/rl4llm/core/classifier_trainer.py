"""Classifier model trainer for coherent detection"""

import logging
import math
import multiprocessing as mp
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import LongformerForSequenceClassification, PreTrainedTokenizer

from .base_trainer import BaseTrainer
from .data_types import ClassifierConfig

# from rl4llm.models import ClassifierModel


class ClassifierTrainer(BaseTrainer):
    """
    Classifier trainer.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        model: LongformerForSequenceClassification,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_ds: List[Dict],
        test_ds: List[Dict],
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, tokenizer, device, torch_dtype, artifacts_path, logger, 0)

        self.model = model.to(dtype=torch_dtype, device=device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.step = 0

        self.logger.info('Preprocessing datasets...')
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            collate_fn=self._collate_function,
            pin_memory=True if device.type == 'cuda' else False,  # Optimize for GPU
            shuffle=True,
            drop_last=True,
        )

        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.config.eval_batch_size,
            collate_fn=self._collate_function,
            pin_memory=True if device.type == 'cuda' else False,  # Optimize for GPU
            shuffle=False,
            drop_last=True,
        )

    @torch.no_grad()
    def _evaluate(self) -> None:
        """Evaluate the policy model on the test dataset."""
        self.model.eval()
        metrics_accumulator = {'loss': 0.0, 'correct': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'samples': 0}

        for batch in self.test_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            atten_mask = batch['attention_mask'].to(self.device)

            output = self.model(input_ids, attention_mask=atten_mask)
            loss, batch_metrics = self._compute_loss(output.logits, labels)

            batch_size = input_ids.size(0)
            metrics_accumulator['loss'] += loss.item() * batch_size
            metrics_accumulator['correct'] += batch_metrics['correct']
            metrics_accumulator['TP'] += batch_metrics['TP']
            metrics_accumulator['FP'] += batch_metrics['FP']
            metrics_accumulator['FN'] += batch_metrics['FN']
            metrics_accumulator['samples'] += batch_size

        # Calculate final metrics
        metrics = self._calculate_metrics(
            metrics_accumulator['loss'],
            metrics_accumulator['correct'],
            metrics_accumulator['TP'],
            metrics_accumulator['FP'],
            metrics_accumulator['FN'],
            metrics_accumulator['samples'],
        )

        # Format for logging
        self._log_stats_to_tensorboard({f'evaluation/{k}': v for k, v in metrics.items()}, step=self.step)

        self.logger.info(
            f"Evaluation - Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )

    def _train(self) -> None:
        """Train the policy model on a training dataset."""
        self.model.train()
        self._metrics.reset()
        total_updates_per_epoch = math.ceil(len(self.train_loader) / self.config.gradient_accumulate_steps)
        total_steps = self.config.num_epochs * total_updates_per_epoch  # Total gradient update steps
        progress_bar = tqdm(total=total_steps, desc='Training Progress', unit='step')

        metrics_accumulator = {'loss': 0.0, 'correct': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'samples': 0}

        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.config.num_epochs):
                for i, batch in enumerate(self.train_loader):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    atten_mask = batch['attention_mask'].to(self.device)

                    output = self.model(input_ids, attention_mask=atten_mask)
                    loss, batch_metrics = self._compute_loss(output.logits, labels)

                    batch_size = input_ids.size(0)

                    metrics_accumulator['loss'] += loss.item() * batch_size
                    metrics_accumulator['correct'] += batch_metrics['correct']
                    metrics_accumulator['TP'] += batch_metrics['TP']
                    metrics_accumulator['FP'] += batch_metrics['FP']
                    metrics_accumulator['FN'] += batch_metrics['FN']
                    metrics_accumulator['samples'] += batch_size

                    if self.config.gradient_accumulate_steps > 1:
                        loss = loss / self.config.gradient_accumulate_steps

                    loss.backward()

                    # Perform gradient update step
                    if (i + 1) % self.config.gradient_accumulate_steps == 0 or (i + 1) == len(self.train_loader):
                        grad_norm = self.get_grad_norm(self.model).item()
                        # self._metrics.add_metric('training/grad_norm', grad_norm.item())
                        if self.config.clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=self.config.clip_grad_norm, error_if_nonfinite=True
                            )
                        self.optimizer.step()
                        self.step += 1

                        if self.scheduler is not None:
                            self.scheduler.step()

                        self.optimizer.zero_grad()

                        # Update progress bar
                        progress_bar.update(1)

                        train_metrics = self._calculate_metrics(
                            metrics_accumulator['loss'],
                            metrics_accumulator['correct'],
                            metrics_accumulator['TP'],
                            metrics_accumulator['FP'],
                            metrics_accumulator['FN'],
                            metrics_accumulator['samples'],
                        )

                        train_metrics['grad_norm'] = grad_norm
                        train_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                        train_metrics['elapsed/update'] = self.step

                        # Evaluate periodically
                        if self.config.eval_interval > 0 and self.step % self.config.eval_interval == 0:
                            self._evaluate()
                            self.model.train()

                        if self.config.checkpoint_interval > 0 and self.step % self.config.checkpoint_interval == 0:
                            save_dir = os.path.join(self._checkpoint_dir, f"step_{self.step}")
                            self._save_checkpoint(save_dir)

                        # Log training metrics to TensorBoard or any other logging utility
                        self._log_stats_to_tensorboard({f'training/{k}': v for k, v in train_metrics.items()}, step=self.step)

                        metrics_accumulator = {'loss': 0.0, 'correct': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'samples': 0}

        progress_bar.close()

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute binary classification loss and metrics for a single batch."""

        loss = self.loss_fn(logits.view(-1, self.model.num_labels), labels.view(-1))

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        TP = ((preds == 1) & (labels == 1)).sum().item()
        FP = ((preds == 1) & (labels == 0)).sum().item()
        FN = ((preds == 0) & (labels == 1)).sum().item()

        metrics = {
            'loss': loss.detach().cpu().item(),
            'correct': correct,
            'TP': TP,
            'FP': FP,
            'FN': FN,
        }

        return loss, metrics

    def _calculate_metrics(self, total_loss, total_correct, total_TP, total_FP, total_FN, total_samples):
        """Calculate evaluation metrics from accumulated values."""
        if total_samples == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Safe division for precision and recall
        precision = total_TP / max(total_TP + total_FP, 1)
        recall = total_TP / max(total_TP + total_FN, 1)

        # Safe calculation for F1 score
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        return {'loss': average_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def _collate_function(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader during training"""

        batch_inputs = self.tokenizer(
            [item['text'] for item in batch],
            return_tensors='pt',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
        )

        batch_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        return {
            **batch_inputs,
            'labels': batch_labels,
        }

    def _save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        self.logger.info('Saving policy model checkpoint...')
        self.model.save_pretrained(save_dir)
