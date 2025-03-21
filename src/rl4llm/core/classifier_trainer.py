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
from transformers import AutoTokenizer, PreTrainedTokenizer

from rl4llm.models import ClassifierModel

from .base_trainer import BaseTrainer
from .data_types import ClassifierConfig


class ClassifierTrainer(BaseTrainer):
    """
    Classifier trainer.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        model: ClassifierModel,
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
        total_loss = 0.0
        total_correct = 0
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_samples = 0

        for batch in self.test_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            atten_mask = batch['atten_mask'].to(self.device)

            logits = self.model(input_ids, attention_mask=atten_mask)
            loss, metrics = self._compute_loss(logits, labels)

            batch_size = input_ids.size(0)

            total_loss += loss.item() * batch_size
            total_correct += metrics['correct']
            total_TP += metrics['TP']
            total_FP += metrics['FP']
            total_FN += metrics['FN']
            total_samples += batch_size

        average_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        eval_metrics = {
            'loss': average_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

        # These metrics will later be accumulated over mini batches
        for k, v in eval_metrics.items():
            self._metrics.add_metric(f'evaluation/{k}', v)

        self.logger.info(
            f"Evaluation - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

    def _train(self) -> None:
        """Train the policy model on a training dataset."""
        self.model.train()
        self._metrics.reset()
        total_updates_per_epoch = math.ceil(len(self.train_loader) / self.config.gradient_accumulate_steps)
        total_steps = self.config.num_epochs * total_updates_per_epoch  # Total gradient update steps
        progress_bar = tqdm(total=total_steps, desc='Training Progress', unit='step')

        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.config.num_epochs):
                for i, batch in enumerate(self.train_loader):
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    atten_mask = batch['atten_mask'].to(self.device)

                    logits = self.model(input_ids, attention_mask=atten_mask)
                    loss, train_metrics = self._compute_loss(logits, labels)

                    if self.config.gradient_accumulate_steps > 1:
                        loss = loss / self.config.gradient_accumulate_steps

                    loss.backward()

                    # Accumulate metrics over mini batches
                    for k, v in train_metrics.items():
                        self._metrics.add_metric(f'training/{k}', v)

                    # Perform gradient update step
                    if (i + 1) % self.config.gradient_accumulate_steps == 0 or (i + 1) == len(self.train_loader):
                        grad_norm = self.get_grad_norm(self.model)
                        self._metrics.add_metric('training/grad_norm', grad_norm.item())
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

                        # Optional: Log metrics per gradient update step
                        self._metrics.add_metric('elapsed/update', self.step)
                        self._metrics.add_metric('training/learning_rate', self.optimizer.param_groups[0]['lr'])

                        # Evaluate periodically
                        if self.config.eval_interval > 0 and self.step % self.config.eval_interval == 0:
                            self._evaluate()
                            self.model.train()

                        if self.config.checkpoint_interval > 0 and self.step % self.config.checkpoint_interval == 0:
                            save_dir = os.path.join(self._checkpoint_dir, f"step_{self.step}")
                            self._save_checkpoint(save_dir)

                        # Log training metrics to TensorBoard or any other logging utility
                        metrics = self._metrics.get_summary()
                        self._log_stats_to_tensorboard(metrics, step=self.step)

                        # Reset metrics after logging
                        self._metrics.reset()

        progress_bar.close()

    def _save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        self.logger.info('Saving policy model checkpoint...')
        self.model.save_pretrained(save_dir)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute binary classification loss and metrics for a single batch."""
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        TP = ((preds == 1) & (targets == 1)).sum().item()
        FP = ((preds == 1) & (targets == 0)).sum().item()
        FN = ((preds == 0) & (targets == 1)).sum().item()
        batch_size = targets.size(0)

        accuracy = correct / batch_size if batch_size > 0 else 0.0
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        metrics = {
            'loss': loss.detach().cpu().item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            # 'TP': TP,
            # 'FP': FP,
            # 'FN': FN,
        }

        return loss, metrics

    def _collate_function(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader during training"""
        pad_token_id = self.pad_token_id

        # Pad states and actions (long tensors)
        batch_input_ids = pad_sequence(
            [
                item['tokens'] if isinstance(item['tokens'], torch.Tensor) else torch.tensor(item['tokens'], dtype=torch.long)
                for item in batch
            ],
            batch_first=True,
            padding_value=pad_token_id,
        )
        batch_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        batch_atten_mask = (batch_input_ids != self.pad_token_id).bool()

        return {
            'input_ids': batch_input_ids,
            'atten_mask': batch_atten_mask,
            'labels': batch_labels,
        }

    # def _preprocess_dataset(self, dataset: List[Dict]) -> List[Dict]:
    #     """Parallelized pre-tokenization of the entire dataset."""
    #     self.logger.info(f"Preprocessing dataset with {len(dataset)} examples...")

    #     return [self._preprocess_item(item) for item in dataset]

    # def _preprocess_item(self, item: Dict) -> Dict:
    #     """Helper function to preprocess a single item."""
    #     text = item['text']
    #     label = item['label']

    #     inputs = self.tokenizer(
    #         text,
    #         return_tensors='pt',
    #         truncation=True,
    #         padding=False,
    #         max_length=self.config.max_sequence_length,
    #     )

    #     return {
    #         'input_ids': inputs['input_ids'].squeeze(0),
    #         'attention_mask': inputs['attention_mask'].squeeze(0),
    #         'label': label,
    #     }
