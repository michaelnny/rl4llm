"""Base GRPO trainer with common functionality for both single and distributed training"""

import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.graders import format_structure_grader, math_problem_grader
from rl4llm.utils import (
    DummyLogger,
    MetricsCollector,
    masked_mean,
    masked_sum,
    masked_whiten,
    save_yaml_config_file,
)

from .data_types import GRPOConfig, GRPOSample


class BaseGRPOTrainer(ABC):
    """Base GRPO trainer with common functionality for both single and distributed training"""

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: Optional[logging.Logger] = None,
        is_master: Optional[bool] = True,
    ):
        self.config = config
        self.device = device
        self.torch_dtype = torch_dtype
        self.tokenizer = tokenizer
        self.artifacts_path = artifacts_path
        self.logger = logger or DummyLogger()  # Use dummy logger to make coding easier

        # Initialize counters
        self.train_episode_count = 0
        self.eval_episode_count = 0
        self.update_count = 0
        self.iteration_count = 0
        self.ref_update_count = 0
        self.explore_epsilon = 0
        self.generation_mode = False
        self.is_master = is_master

        self.stats_completion_lengths = []

        self._initialize()

    def _initialize(self):
        """Initialize training components"""
        # Setup special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.metrics = MetricsCollector()

        # Setup directories and logging
        self.tb_log_dir = os.path.join(self.artifacts_path, 'tb_logs')
        self.checkpoint_dir = os.path.join(self.artifacts_path, 'checkpoints')

        if self.is_master:
            for path in [self.tb_log_dir, self.checkpoint_dir]:
                os.makedirs(path, exist_ok=True)

            self.writer = SummaryWriter(self.tb_log_dir)
        else:
            self.writer = None

    def train(self, log_hyper_params: Optional[Dict] = None):
        """Start to train the model using RL GRPO.

        Args:
            hyper_params (Dict): Hyper parameters for the training job.
        """

        # log the params we use for this training run
        if log_hyper_params and self.is_master:
            save_yaml_config_file(log_hyper_params, os.path.join(self.artifacts_path, 'config.yaml'))
            self._log_hyper_params_to_tensorboard(log_hyper_params)

        for _ in tqdm(range(self.config.max_steps), desc='Training steps', disable=not self.is_master):
            self.run_one_iteration()

    @abstractmethod
    def run_one_iteration(self):
        """Run one training iteration - implement in subclass"""
        pass

    @abstractmethod
    def train_policy(self, policy_model: PreTrainedModel, samples: List[GRPOSample]) -> None:
        """Train the policy model using the collected samples - implement in subclass"""
        pass

    @torch.no_grad()
    def evaluate_policy(self, policy_model: PreTrainedModel, test_loader: DataLoader) -> None:
        """Run evaluation"""

        system_prompt = self.config.system_prompt

        # use greedy sampling for evaluation
        eval_kwargs = {
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': 0.0,
            'top_p': 1.0,
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
            for batch in test_loader:
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
                message_prompt = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(
                    message_prompt,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    padding_side='left',
                    max_length=self.tokenizer.model_max_length,
                ).to(self.device)

                outputs = policy_model.generate(**inputs, **eval_kwargs)

                reward_dict = self._process_generation_outputs(
                    questions, ground_truths, task_types, inputs.input_ids, outputs.sequences
                )

                total_samples += len(questions)
                total_correct += (reward_dict['accuracy_rewards'] == 1.0).sum().item()

        # accuracy = total_correct / total_samples
        # self.logger.info(f"Evaluation accuracy: {accuracy:.4f}")

    @torch.no_grad()
    def generate_group_samples(
        self,
        sample: Dict[str, str],
        policy_model: PreTrainedModel,
        reference_model: PreTrainedModel,
        generator: Optional[CustomLLMGenerator] = None,
    ) -> List[GRPOSample]:
        """Generate responses for a batch of questions and ground truth answers

        Args:
            sample: Dict containing question, ground_truth and task_type
            policy_model: Policy model to compute log probabilities
            reference_model: Reference model to compute log probabilities
            generator: CustomLLMGenerator instance, if None, use HF default 'model.generate'

        Returns:
            List[Dict]: List of samples for all groups in the batch
        """

        # Prepare messages for the entire batch
        question = sample['question']
        ground_truth = sample['ground_truth']
        task_type = sample['task_type'].upper()

        assert isinstance(question, str)

        if task_type not in ['MATH', 'GSM']:
            raise ValueError(f"Invalid task type: {task_type}, only support 'MATH' or 'GSM'")

        # Create the basic message template
        system_prompt = self.config.system_prompt
        if not system_prompt:
            sample_message = [{'role': 'user', 'content': question.strip()}]
        else:
            sample_message = [
                {'role': 'system', 'content': system_prompt.strip()},
                {'role': 'user', 'content': question.strip()},
            ]
        # Build a list of messages for the entire group
        batch_messages = [sample_message] * self.config.group_size
        # Tokenize single message
        message_prompt = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            message_prompt,
            return_tensors='pt',
            truncation=True,
            padding=False,  # no need padding for single data point
            padding_side='left',
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

        use_custom_generator = (
            generator is not None
            and hasattr(generator, 'generate')
            and (self.config.group_temperature or self.config.explore_start_ratio > 0)
        )
        if not use_custom_generator:
            generator = policy_model

        generation_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'do_sample': True,
            'use_cache': True,
            'output_scores': False,
            'output_logits': False,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
        }

        if use_custom_generator:
            if self.config.group_temperature:
                # Spread temperature values according to self.config.group_size, where 0.0 means greedy sampling
                # this idea is similar how we do it in distributed RL training in classical RL
                # where we have multiple agents running in parallel, some agents are more exploratory than others
                temperature = torch.linspace(
                    0.0, self.config.temperature, steps=self.config.group_size, dtype=self.torch_dtype, device=self.device
                )
                # temperature = (
                #     torch.pow(torch.linspace(0.0, 1.0, steps=self.config.group_size, dtype=self.torch_dtype, device=self.device), 2)
                #     * self.config.temperature
                # )
            else:
                # make code compatible
                temperature = torch.tensor(
                    [self.config.temperature] * self.config.group_size, dtype=self.torch_dtype, device=self.device
                )

            explore_epsilon = self._get_exploration_epsilon()
            enable_exploration = (explore_epsilon > 0) and (random.random() < self.explore_epsilon)

            # add exploration parameters
            generation_kwargs['temperature'] = temperature
            if (self.config.explore_start_ratio > 0 or self.config.explore_uncertainty > 0.0) and enable_exploration:
                explore_start_steps = self._get_moving_average_completion_length() * self.config.explore_start_ratio
                generation_kwargs['enable_exploration'] = enable_exploration
                generation_kwargs['explore_start_steps'] = explore_start_steps
                # generation_kwargs["explore_uncertainty"] = self.config.explore_uncertainty
                generation_kwargs['explore_top_k'] = self.config.explore_top_k
                generation_kwargs['explore_top_k_beta'] = self.config.explore_top_k_beta

        outputs = generator.generate(**generation_kwargs)

        return self._process_generation_outputs(
            question,
            ground_truth,
            task_type,
            inputs.input_ids,
            outputs.sequences,
            policy_model=policy_model,
            reference_model=reference_model,
        )

    @staticmethod
    def compute_masked_monte_carlo_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.FloatTensor:
        """
        Computes monte carlo returns considering only assistant turns.

        Args:
            rewards (torch.Tensor): Float tensor with rewards (0 for user), shape [seq_len]
            mask (torch.Tensor): Binary mask (0 for user, 1 for assistant), shape [seq_len]
            gamma (float): Discount factor

        Returns:
            torch.Tensor: Tensor of the original shape, with discounted returns
                for assistant turns and zeros for user turns
        """
        # Input validation
        assert rewards.dim() == mask.dim() == 1, 'Inputs must be 1-dimensional'
        assert rewards.size(0) == mask.size(0), 'Rewards and mask must have same length'
        assert gamma > 0.0 and gamma <= 1.0, 'Discount factor must be in (0, 1]'

        # Initialize returns tensor
        returns = torch.zeros_like(mask, dtype=rewards.dtype)

        # Get assistant rewards using boolean indexing
        assistant_rewards = rewards[mask.bool()]
        seq_len = len(assistant_rewards)

        # Handle empty case
        if seq_len == 0:
            return returns

        # Initialize assistant returns
        assistant_returns = torch.zeros_like(assistant_rewards, dtype=rewards.dtype)

        R = 0
        for t in reversed(range(len(assistant_rewards))):
            R = assistant_rewards[t] + gamma * R
            assistant_returns[t] = R

        # Place assistant returns back in the original tensor
        returns[mask.bool()] = assistant_returns

        return returns

    @staticmethod
    def normalize_group_rewards(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize group rewards by subtracting the mean and dividing by the standard deviation.

        Args:
            rewards (torch.Tensor): List of rewards for the group.
            eps (float): Small value to prevent division by zero.

        Returns:
            torch.Tensor: Normalized rewards.
        """
        assert eps > 0.0, 'Epsilon must be positive'
        assert rewards.dim() == 1, 'Rewards must be 1-dimensional'
        if len(rewards) <= 1:
            return rewards

        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)

        if std_reward < eps:
            return torch.zeros_like(rewards)

        return (rewards - mean_reward) / (std_reward + eps)

    def compute_dynamic_discount(self, episode_length: int) -> float:
        """Compute dynamic discount factor."""
        assert episode_length > 0, 'Episode length must be greater than 0.'
        scaled_length = min(episode_length / self.config.max_completion_length, 1.0)
        gamma = self.config.min_gamma + (self.config.max_gamma - self.config.min_gamma) * scaled_length
        return gamma

    def _get_exploration_epsilon(self) -> float:
        """Computes exploration epsilon based on the current iteration step count."""
        if self.config.explore_decay_steps == 0:
            self.explore_epsilon = 0.0
        elif self.iteration_count >= self.config.explore_decay_steps:
            self.explore_epsilon = self.config.explore_min_epsilon
        else:
            # Cosine decay schedule
            progress = self.iteration_count / self.config.explore_decay_steps
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
            self.explore_epsilon = (
                self.config.explore_min_epsilon
                + (self.config.explore_init_epsilon - self.config.explore_min_epsilon) * cosine_decay
            )

        return self.explore_epsilon

    def _process_generation_outputs(
        self,
        questions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
        task_types: Union[str, List[str]],
        input_ids: torch.Tensor,
        full_sequences: torch.Tensor,
        policy_model: Optional[PreTrainedModel] = None,
        reference_model: Optional[PreTrainedModel] = None,
    ) -> Union[List[GRPOSample], Dict[str, torch.Tensor]]:
        """Process generated outputs for both training and evaluation

        Args:
            questions: Single question (training) or list of questions (evaluation)
            ground_truths: Single ground truth (training) or list (evaluation)
            task_types: Single task type (training) or list (evaluation)
            input_ids: Prompt token ids [batch_size, prompt_seq_len]
            full_sequences: Full sequence token ids [batch_size, seq_len]
            policy_model: Policy model to compute log probabilities, only required for training
            reference_model: Reference model to compute log probabilities, only required for training

        Returns:
            Training: List[GRPOSample]
            Evaluation: Dict[str, torch.Tensor] containing rewards
        """
        # Standardize inputs to lists
        batch_size = full_sequences.size(0)
        questions = [questions] * batch_size if isinstance(questions, str) else questions
        ground_truths = [ground_truths] * batch_size if isinstance(ground_truths, str) else ground_truths
        task_types = [task_types] * batch_size if isinstance(task_types, str) else task_types

        assert len(questions) == len(ground_truths) == len(task_types) == batch_size

        # Extract completions
        prompt_length = input_ids.size(1)
        completion_ids = full_sequences[:, prompt_length:]
        completion_tokens_count = (completion_ids != self.pad_token_id).sum(dim=1).cpu()
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards
        reward_output = self._compute_rewards(completion_texts, ground_truths, completion_tokens_count.tolist())
        accuracy_rewards = reward_output['accuracy_rewards']
        format_rewards = reward_output['format_rewards']
        total_rewards = reward_output['total_rewards']

        is_training = policy_model is not None and reference_model is not None

        # Log samples
        sample_indices = (
            random.sample(range(len(completion_texts)), k=min(2, 0.1 * self.config.group_size))
            if is_training
            else range(len(completion_texts))
        )
        tb_tag = 'training' if is_training else 'evaluation'

        for i in range(len(completion_texts)):
            if is_training:
                self.train_episode_count += 1
            else:
                self.eval_episode_count += 1

            metric_prefix = 'objective' if is_training else 'evaluation'
            self.metrics.add_metric(f'{metric_prefix}/accuracy_reward', accuracy_rewards[i].item())
            self.metrics.add_metric(f'{metric_prefix}/format_reward', format_rewards[i].item())
            self.metrics.add_metric(f'{metric_prefix}/total_rewards', total_rewards[i].item())
            self.metrics.add_metric(f'{metric_prefix}/completion_length', completion_tokens_count[i].item())

            if i in sample_indices or not is_training:
                # we only log few samples for training
                formatted_text = self._format_sample_text(
                    task_types[i], questions[i], ground_truths[i], total_rewards[i].item(), completion_texts[i]
                )
                self._log_sample_to_tensorboard(
                    tb_tag, formatted_text, self.train_episode_count if is_training else self.eval_episode_count
                )

        if not is_training:
            return reward_output

        # store historical completion lengths
        self.stats_completion_lengths.append(completion_tokens_count.float().mean().item())

        # Training specific processing
        normalized_rewards = (
            self.normalize_group_rewards(total_rewards) if self.config.normalize_group_rewards else total_rewards
        )

        states = full_sequences[:, :-1]
        actions = full_sequences[:, 1:]
        pi_logprobs = self._compute_action_logprobs(policy_model, states, actions).cpu()
        ref_logprobs = self._compute_action_logprobs(reference_model, states, actions).cpu()

        # Do not include the prompt or pad tokens in the loss
        # for example, if we have a sequence token ids: [1, 2, 3, 4, 5, 6, 7, -1, -1]
        # where [1, 2, 3, 4] are the prompt tokens
        # and [5, 6, 7] are the completion tokens
        # -1 is the pad token
        # the, the loss mask will be [0, 0, 0, 1, 1, 1, 0, 0, 0]
        loss_mask = (actions != self.pad_token_id).bool()
        loss_mask[:, : prompt_length - 1] = 0

        results = []

        # construct a list of samples by trim the sequence to the first EOS token and ignore EOS tokens in the prompt
        eos_mask = actions == self.eos_token_id
        eos_mask[:, :prompt_length] = False

        cut_positions = torch.where(
            eos_mask.any(dim=1),
            eos_mask.float().argmax(dim=1) + 1,
            actions.size(1) + 1,
        )

        # Cut sequences to the first eos token in completion
        for i, cut_position in enumerate(cut_positions):
            assert completion_tokens_count[i].item() > 0
            assert loss_mask[i, ...].sum().item() == completion_tokens_count[i]
            assert loss_mask[i, :cut_position].sum().item() == completion_tokens_count[i]

            seq_rewards = torch.zeros_like(actions[i, :cut_position], dtype=self.torch_dtype)
            seq_rewards[-1] = normalized_rewards[i]  # important to use normalized rewards here

            gamma = (
                self.compute_dynamic_discount(
                    completion_tokens_count[i].item(), min_gamma=self.config.min_gamma, max_gamma=self.config.max_gamma
                )
                if self.config.dynamic_discount
                else self.config.gamma
            )

            returns = self.compute_masked_monte_carlo_returns(
                rewards=seq_rewards, mask=loss_mask[i, :cut_position], gamma=gamma
            )

            results.append(
                GRPOSample(
                    states=states[i, :cut_position].cpu(),
                    actions=actions[i, :cut_position].cpu(),
                    loss_mask=loss_mask[i, :cut_position].cpu(),
                    reward=total_rewards[i].cpu(),
                    advantages=returns.cpu(),
                    pi_logprobs=pi_logprobs[i, :cut_position].cpu(),
                    ref_logprobs=ref_logprobs[i, :cut_position].cpu(),
                )
            )

        return results

    def _compute_rewards(
        self, completion_texts: List[str], ground_truths: List[str], completion_tokens_count: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for completions against ground truth(s)

        Args:
            completion_texts: List of generated completion texts
            ground_truths: A list of ground truths
            completion_tokens_count: List of completion lengths

        Returns:
            Dict: containing accuracy, format and total rewards

        """
        assert len(completion_texts) == len(ground_truths) == len(completion_tokens_count)

        accuracy_rewards = []
        format_rewards = []

        # TODO consider support other tasks other than math
        for completion, ground_truth, completion_len in zip(completion_texts, ground_truths, completion_tokens_count):
            accuracy_rewards.append(math_problem_grader(completion, ground_truth))
            format_rewards.append(
                format_structure_grader(
                    completion,
                    seq_length=completion_len,
                    min_length=min(self.config.min_completion_length, 50),
                    xml_format=self.config.xml_format,
                )
            )

        accuracy_rewards = torch.tensor(accuracy_rewards, dtype=self.torch_dtype)
        format_rewards = torch.tensor(format_rewards, dtype=self.torch_dtype)
        total_rewards = accuracy_rewards + format_rewards

        return {'accuracy_rewards': accuracy_rewards, 'format_rewards': format_rewards, 'total_rewards': total_rewards}

    def _compute_action_logprobs(
        self, model: PreTrainedModel, input_ids: torch.LongTensor, actions: torch.LongTensor
    ) -> torch.Tensor:
        """Compute log probabilities of actions given the input states.

        Args:
            model (PreTrainedModel): Model to compute log probabilities, shape [batch_size, seq_len]
            input_ids (torch.LongTensor): Input token ids, shape [batch_size, seq_len]
            actions (torch.LongTensor): Action token ids, shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Log probabilities of actions, shape [batch_size, seq_len]
        """

        assert input_ids.dim() == actions.dim() == 2
        assert input_ids.shape == actions.shape

        attention_mask = (input_ids != self.pad_token_id).bool()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # this runs into CUDA OOM
        # logprobs = torch.log_softmax(logits, dim=-1)
        # return torch.gather(logprobs, dim=2, index=actions.unsqueeze(2)).squeeze(2)

        # Process log_softmax and gather operations one sample at a time
        batch_size = logits.shape[0]
        sample_logprobs = []

        for i in range(batch_size):
            # Process single sample
            sample_logits = logits[i, ...]
            sample_logprobs_all = torch.log_softmax(sample_logits, dim=-1)
            sample_actions = actions[i, ...].unsqueeze(1)
            sample_logprob = torch.gather(sample_logprobs_all, dim=1, index=sample_actions).squeeze(1)
            sample_logprobs.append(sample_logprob)

        # Concatenate results
        return torch.stack(sample_logprobs, dim=0)

    def _compute_loss(self, pi_logprobs: torch.Tensor, batch: GRPOSample) -> Tuple[torch.Tensor, Dict]:
        """Process a single training batch

        Args:
            pi_logprobs (torch.Tensor): Log probabilities of actions computed using current policy, shape [batch_size, seq_len]
            batch (GRPOSample): A batch of samples collected during generation

        Returns:
            Tuple[torch.Tensor, Dict]: Tuple containing the total loss tensor and a dictionary of metrics
        """

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

        # First average over the sequence length, then average over the batch
        pg_loss = masked_mean(pg_losses, loss_mask, dim=1).mean()
        kl = masked_mean(per_token_kl, loss_mask, dim=1).mean()

        # Only compute KL loss for incorrect samples
        # correctness_mask = (mini_batch.reward >= 1).to(self.device)  # Correct if reward >= 1
        # incorrect_mask = ~correctness_mask
        # kl_loss = self.config.kl_loss_coef * (kl * incorrect_mask).mean()

        kl_loss = self.config.kl_loss_coef * kl
        loss = pg_loss + kl_loss

        metrics = {
            'total_loss': loss.detach().item(),
            'pg_loss': pg_loss.detach().item(),
            'kl_loss': kl_loss.detach().item(),
            'kl': kl.detach().item(),
        }

        return loss, metrics

    def _get_moving_average_completion_length(self, window_size: int = 10) -> int:
        """Compute moving average completion lengths over the past N iterations"""
        assert window_size > 1

        if not self.stats_completion_lengths or len(self.stats_completion_lengths) < window_size:
            return 200

        # Convert to numpy array if not already
        values = np.array(self.stats_completion_lengths)

        # Calculate moving average using numpy's convolve
        weights = np.ones(window_size) / window_size
        moving_averages = np.convolve(values, weights, mode='valid')

        # Get the last moving average value
        last_ma_value = moving_averages[-1]

        return last_ma_value

    def _train_collate_function(self, batch: List[GRPOSample]) -> GRPOSample:
        """Collate function for DataLoader during training"""
        pad_token_id = self.pad_token_id
        torch_dtype = self.torch_dtype

        batch_size = len(batch)
        max_seq_len = max([len(item.states) for item in batch])
        batch_state_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        batch_action_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        batch_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)

        batch_advantages = torch.full((batch_size, max_seq_len), 0.0, dtype=torch_dtype)
        batch_pi_logprobs = torch.full((batch_size, max_seq_len), 0.0, dtype=torch_dtype)
        batch_ref_logprobs = torch.full((batch_size, max_seq_len), 0.0, dtype=torch_dtype)
        batch_reward = torch.full((batch_size,), 0.0, dtype=torch_dtype)

        for i, item in enumerate(batch):
            seq_len = len(item.states)
            batch_state_ids[i, :seq_len] = item.states.to(dtype=torch.long)
            batch_action_ids[i, :seq_len] = item.actions.to(dtype=torch.long)
            batch_loss_mask[i, :seq_len] = item.loss_mask.to(dtype=torch.bool)
            batch_advantages[i, :seq_len] = item.advantages.to(dtype=torch_dtype)
            batch_pi_logprobs[i, :seq_len] = item.pi_logprobs.to(dtype=torch_dtype)
            batch_ref_logprobs[i, :seq_len] = item.ref_logprobs.to(dtype=torch_dtype)
            batch_reward[i] = item.reward

        return GRPOSample(
            states=batch_state_ids,
            actions=batch_action_ids,
            loss_mask=batch_loss_mask,
            pi_logprobs=batch_pi_logprobs,
            ref_logprobs=batch_ref_logprobs,
            advantages=batch_advantages,
            reward=batch_reward,
        )

    def _format_sample_text(
        self, task_type: str, question: str, ground_truth: str, total_reward: float, completion_text: str
    ) -> str:
        """Format sample text for logging"""

        formatted_text = (
            f"**Question [{task_type}]**: {question}\n\n"
            f"**Ground Truth**: {ground_truth}\n\n"
            f"**Graded Reward**: {total_reward}\n\n"
            f"**Full Answer**:\n```json\n{completion_text}\n```"
        )

        return formatted_text

    def _log_hyper_params_to_tensorboard(self, config: Dict[str, Any]):
        """Log hyper parameters used for the job"""
        if self.writer and config:
            try:
                config_str = yaml.dump(config, sort_keys=False, indent=4)
                self.writer.add_text('config/parameters', f"```yaml\n{config_str}\n```", 0)
            except Exception as _e:
                self.logger.warning('Failed to log hyper parameters to tensorboard')

    def _log_sample_to_tensorboard(self, tag: str, formatted_text: str, episode_count: int):
        """Log a sample text to tensorboard"""
        if self.writer:
            try:
                self.writer.add_text(f'{tag}/sample', formatted_text, episode_count)
            except Exception as _e:
                self.logger.warning('Failed to log sample to tensorboard')

    def _log_stats_to_tensorboard(self, stats: Dict[str, Any], step: int):
        """Log stats to tensorboard"""
        if self.writer:
            try:
                for name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"{name}", value, step)
            except Exception as _e:
                self.logger.warning('Failed to log stats to tensorboard')
