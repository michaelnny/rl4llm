"""Base GRPO trainer with common functionality for both single and distributed training"""

import logging
import multiprocessing as mp
import os
import random
from abc import ABC
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.graders import format_structure_grader, math_problem_grader
from rl4llm.utils import (
    DummyLogger,
    FileHandler,
    MetricsCollector,
    compute_grad_norm,
    masked_mean,
    masked_sum,
    masked_whiten,
    save_yaml_config_file,
)

from .data_types import GRPOConfig, GRPOSample, SampleLog


class BaseGRPOTrainer(ABC):
    """
    Base GRPO trainer with common functionality for both single and distributed training.
    With focus on code reusability and execution speed.
    """

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: Optional[logging.Logger] = None,
        rank: Optional[int] = 0,
    ):
        """
        Initialize the BaseGRPOTrainer with training components.

        Args:
            config (GRPOConfig): Configuration object for training parameters.
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding/decoding text.
            device (torch.device): Device (CPU/GPU) for computation.
            torch_dtype (torch.dtype): Data type for PyTorch tensors (e.g., float32).
            artifacts_path (str): Directory path for saving logs and checkpoints.
            logger (Optional[logging.Logger]): Logger for training events; defaults to DummyLogger if None.
        """

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
        self.rank = rank
        self.is_master = rank == 0

        # For custom exploring start where we skip do exploration for the <think> token
        self.think_token_len = len(self.tokenizer.encode('<think>')) if self.config.xml_format else 0

        # For moving average of completion lengths
        self._completion_lengths = deque(maxlen=1000)

        self._initialize()

    def _initialize(self):
        """Initialize training components"""

        # Setup special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Initialize metrics
        self._metrics = MetricsCollector()

        # Setup directories and logging
        self._setup_directories()

        if self.is_master:
            # Initialize TensorBoard writer
            self._writer = SummaryWriter(self._tb_log_dir)
        else:
            self._writer = None

        # Initialize sample file handlers
        train_sample_file = os.path.join(self._samples_dir, f'training_samples_rank{self.rank}.jsonl')
        eval_sample_file = os.path.join(self._samples_dir, f'evaluation_samples_rank{self.rank}.jsonl')
        train_stats = os.path.join(self._samples_dir, f'training_stats_rank{self.rank}.csv')
        self._train_sample_handler = FileHandler(train_sample_file, 'jsonl', True)
        self._eval_sample_handler = FileHandler(eval_sample_file, 'jsonl', True)
        self._train_stats_handler = FileHandler(train_stats, 'csv', False)

    def _setup_directories(self):
        """Helper method to create necessary directories"""
        self._tb_log_dir = os.path.join(self.artifacts_path, 'tb_logs')
        self._checkpoint_dir = os.path.join(self.artifacts_path, 'checkpoints')
        self._samples_dir = os.path.join(self.artifacts_path, 'samples')

        for path in [self._tb_log_dir, self._checkpoint_dir, self._samples_dir]:
            os.makedirs(path, exist_ok=True)

    def on_exit(self):
        self._train_sample_handler.close()
        self._eval_sample_handler.close()
        self._train_stats_handler.close()

    def train(self, log_hyper_params: Optional[Dict] = None):
        """Start to train the model using RL GRPO.

        Args:
            log_hyper_params (Dict[str, Any], optional): Hyperparameters to log.
        """

        # log the params we use for this training run
        if log_hyper_params and self.is_master:
            save_yaml_config_file(log_hyper_params, os.path.join(self.artifacts_path, 'config.yaml'))
            self._log_hyper_params_to_tensorboard(log_hyper_params)

        for _ in tqdm(range(self.config.max_steps), desc='Training steps', disable=not self.is_master):
            self.run_one_iteration()

    def run_one_iteration(self):
        """Run a single training iteration. Must be implemented by subclasses."""
        raise NotImplementedError('Subclasses must implement this method')

    def train_policy(self, samples: List[GRPOSample]) -> None:
        """Train the policy model using collected samples. Must be implemented by subclasses."""
        raise NotImplementedError('Subclasses must implement this method')

    @torch.no_grad()
    def evaluate_policy(self, policy_model: PreTrainedModel, test_loader: DataLoader) -> None:
        """Evaluate the policy model on a test dataset.

        Args:
            policy_model (PreTrainedModel): The model to evaluate.
            test_loader: DataLoader providing test batches.
        """

        eval_kwargs = {
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': None,
            'top_p': None,
            'top_k': None,
            'do_sample': False,  # Greedy sampling for evaluation
            'use_cache': True,
            'output_scores': False,
            'output_logits': False,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
        }

        with self._metrics.timer('evaluation'):
            for batch in test_loader:
                questions = batch['questions']
                ground_truths = batch['ground_truths']
                task_types = batch['task_types']
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = policy_model.generate(input_ids=input_ids, attention_mask=attention_mask, **eval_kwargs)

                self._process_evaluation_outputs(questions, ground_truths, task_types, input_ids, outputs.sequences)

    @torch.no_grad()
    def generate_group_samples(
        self,
        item: Dict[str, str],
        policy_model: PreTrainedModel,
        reference_model: PreTrainedModel,
        generator: Optional[CustomLLMGenerator] = None,
    ) -> List[GRPOSample]:
        """Generate responses for a batch of questions and ground truth answers

        Args:
            item (Dict[str, str]): Dictionary with 'question', 'ground_truth', and 'task_type'.
            policy_model (PreTrainedModel): Model to generate responses and compute log probs.
            reference_model (PreTrainedModel): Model to compute reference log probs.
            generator (CustomLLMGenerator, optional): Custom generator for responses.

        Returns:
            List[Dict]: List of samples for all groups in the batch
        """

        # Prepare messages for the entire batch
        task_type = item['task_type'].upper()
        question = item['question']
        ground_truth = item['ground_truth']
        assert isinstance(question, str)
        if task_type not in ['MATH', 'GSM']:
            raise ValueError(f"Invalid task type: {task_type}, only support 'MATH' or 'GSM'")

        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        if self.config.max_prompt_length >= 512 and input_ids.size(0) > self.config.max_prompt_length:
            self.logger.warning(f"Skip sample with prompt size grater than {self.config.max_prompt_length}")
            return []

        # Expand to have a "group" batch dimension
        input_ids = input_ids.repeat(self.config.group_size, 1).to(self.device)
        attention_mask = attention_mask.repeat(self.config.group_size, 1).to(self.device)

        use_custom_generator = (
            generator is not None
            and hasattr(generator, 'generate')
            and (self.config.group_temperature or self.config.explore_start_ratio > 0)
        )
        if not use_custom_generator:
            generator = policy_model

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'top_p': 1.0,  # self.config.top_p,
            'top_k': 0,  # self.config.top_k,
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
                    0.3, self.config.temperature, steps=self.config.group_size, dtype=self.torch_dtype, device=self.device
                )
                # over written to use group temperatures
                generation_kwargs['temperature'] = temperature

            explore_epsilon = self._get_exploration_epsilon()
            enable_exploration = (
                (self.config.explore_start_ratio > 0) and (explore_epsilon > 0) and (random.random() < explore_epsilon)
            )

            # add random start exploration params
            if enable_exploration:
                generation_kwargs['explore_start_steps'] = self._get_explore_start_steps()
                generation_kwargs['explore_top_k'] = self.config.explore_top_k
                generation_kwargs['explore_entropy_ratio'] = self.config.explore_entropy_ratio
                generation_kwargs['explore_skip_first_n'] = self.think_token_len

        outputs = generator.generate(**generation_kwargs)

        torch.cuda.empty_cache()

        return self._process_training_outputs(
            question,
            ground_truth,
            task_type,
            input_ids,
            outputs.sequences,
            policy_model=policy_model,
            reference_model=reference_model,
        )

    def compute_masked_monte_carlo_returns(self, rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.FloatTensor:
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

    def normalize_group_rewards(self, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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

    def get_grad_norm(self, model: PreTrainedModel) -> torch.Tensor:
        """Compute gradient norm for the given model"""
        return compute_grad_norm(model)

    def preprocess_dataset(self, dataset: Dataset) -> List[Dict]:
        """Pre-tokenize the entire dataset and return a list of tokenized inputs."""

        tokenized_data = []
        for item in dataset:
            question = item['question']
            ground_truth = item['ground_truth']
            task_type = item['task_type']

            sample_message = self._prepare_single_message(question, self.config.system_prompt)
            prompt_str = self.tokenizer.apply_chat_template(sample_message, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                prompt_str,
                return_tensors='pt',
                truncation=True,
                padding=False,
                max_length=self.tokenizer.model_max_length,
            )

            tokenized_data.append(
                {
                    'input_ids': inputs['input_ids'].squeeze(0),  # Shape: [seq_len]
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'question': question,
                    'ground_truth': ground_truth,
                    'task_type': task_type,
                }
            )

        return tokenized_data

    def _prepare_single_message(self, question: str, system_prompt: str = None) -> List[Dict[str, str]]:
        """Prepare a single message for tokenization.

        Args:
            question (str): The user question.
            system_prompt (str): The system prompt to prepend.

        Returns:
            List[Dict[str, str]]: Formatted message list.
        """
        if not system_prompt:
            return [{'role': 'user', 'content': question.strip()}]
        return [
            {'role': 'system', 'content': system_prompt.strip()},
            {'role': 'user', 'content': question.strip()},
        ]

    def _tokenize_function(self, example: Dict) -> Dict:
        """
        Tokenize a single example from the dataset.

        Args:
            example (dict): A single example from the dataset, e.g., {'question': '...', ...}

        Returns:
            dict: Tokenized inputs including 'input_ids' and 'attention_mask'
        """
        message = self._prepare_single_message(example['question'], self.config.system_prompt)
        inputs = self.tokenizer.apply_chat_template(
            message, tokenize=True, return_tensors='pt', return_dict=True, padding=False, add_generation_prompt=True
        )

        # Return a dictionary with squeezed tensors (remove batch dimension)
        example['input_ids'] = inputs['input_ids'].squeeze(0)
        example['attention_mask'] = inputs['attention_mask'].squeeze(0)
        return example

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

    @torch.no_grad()
    def _process_evaluation_outputs(
        self,
        questions: List[str],
        ground_truths: List[str],
        task_types: List[str],
        input_ids: torch.Tensor,
        full_sequences: torch.Tensor,
    ) -> None:
        """Process generated sequences for evaluation, logging metrics only.

        Args:
            questions (List[str]): List of questions.
            ground_truths (List[str]): List of ground truth answers.
            task_types (List[str]): List of task types.
            input_ids (torch.Tensor): Input token IDs.
            full_sequences (torch.Tensor): Generated sequences including prompts.
        """

        outputs = self._process_generation_common_outputs(
            questions=questions,
            ground_truths=ground_truths,
            task_types=task_types,
            input_ids=input_ids,
            full_sequences=full_sequences,
        )

        self._log_sample_metrics(
            is_training=False,
            task_types=task_types,
            questions=questions,
            ground_truths=ground_truths,
            accuracy_rewards=outputs['accuracy_rewards'],
            format_rewards=outputs['format_rewards'],
            total_rewards=outputs['total_rewards'],
            prompt_lengths=outputs['prompt_lengths'],
            completion_lengths=outputs['completion_lengths'],
            completion_texts=outputs['completion_texts'],
        )

    def _process_training_outputs(
        self,
        question: str,
        ground_truth: str,
        task_type: str,
        input_ids: torch.Tensor,
        full_sequences: torch.Tensor,
        policy_model: PreTrainedModel,
        reference_model: PreTrainedModel,
    ) -> List[GRPOSample]:
        """Process generated outputs after generation.

        Args:
            question: Single question
            ground_truth: Single ground truth
            task_type: Single task type
            input_ids: Prompt token ids [batch_size, prompt_seq_len]
            full_sequences: Full sequence token ids [batch_size, seq_len]
            policy_model: Policy model to compute log probabilities, only required for training
            reference_model: Reference model to compute log probabilities, only required for training

        Returns:
            List[GRPOSample] for training
        """
        # Standardize single inputs to lists
        batch_size = full_sequences.size(0)
        questions = [question] * batch_size
        ground_truths = [ground_truth] * batch_size
        task_types = [task_type] * batch_size

        prompt_length = input_ids.size(1)

        outputs = self._process_generation_common_outputs(
            questions=questions,
            ground_truths=ground_truths,
            task_types=task_types,
            input_ids=input_ids,
            full_sequences=full_sequences,
        )

        self._log_sample_metrics(
            is_training=True,
            task_types=task_types,
            questions=questions,
            ground_truths=ground_truths,
            accuracy_rewards=outputs['accuracy_rewards'],
            format_rewards=outputs['format_rewards'],
            total_rewards=outputs['total_rewards'],
            prompt_lengths=outputs['prompt_lengths'],
            completion_lengths=outputs['completion_lengths'],
            completion_texts=outputs['completion_texts'],
        )

        completion_lengths = outputs['completion_lengths']

        # Store historical completion lengths
        self._completion_lengths.append(completion_lengths.float().mean().item())

        # Training specific processing
        total_rewards = outputs['total_rewards']
        normalized_rewards = (
            self.normalize_group_rewards(total_rewards) if self.config.normalize_group_rewards else total_rewards
        )

        states = full_sequences[:, :-1]
        actions = full_sequences[:, 1:]
        pi_logprobs = self._compute_action_logprobs(policy_model, states, actions).cpu()
        torch.cuda.empty_cache()
        if self.config.kl_loss_coef > 0 and reference_model:
            ref_logprobs = self._compute_action_logprobs(reference_model, states, actions).cpu()
            torch.cuda.empty_cache()
        else:
            ref_logprobs = torch.full_like(pi_logprobs, 1e-6).cpu()  # use a place holder to make sure code is compatible

        # TODO can we improve this code of post-processing sample creation??

        # Do not include the prompt or pad tokens in the loss
        # for example, if we have a sequence token ids: [1, 2, 3, 4, 5, 6, 7, -1, -1]
        # where [1, 2, 3, 4] are the prompt tokens
        # and [5, 6, 7] are the completion tokens
        # -1 is the pad token
        # the, the loss mask will be [0, 0, 0, 1, 1, 1, 0, 0, 0]
        loss_mask = (actions != self.pad_token_id).bool()
        loss_mask[:, : prompt_length - 1] = False

        samples = []

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
            assert completion_lengths[i] > 0
            assert loss_mask[i, ...].sum().item() == completion_lengths[i]
            assert loss_mask[i, :cut_position].sum().item() == completion_lengths[i]

            seq_rewards = torch.zeros_like(actions[i, :cut_position], dtype=self.torch_dtype)
            seq_rewards[-1] = normalized_rewards[i]  # important to use normalized rewards here

            gamma = self.compute_dynamic_discount(completion_lengths[i]) if self.config.dynamic_discount else self.config.gamma

            returns = self.compute_masked_monte_carlo_returns(
                rewards=seq_rewards, mask=loss_mask[i, :cut_position], gamma=gamma
            )

            samples.append(
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

        return samples

    def _process_generation_common_outputs(
        self,
        questions: List[str],
        ground_truths: List[str],
        task_types: List[str],
        input_ids: torch.Tensor,
        full_sequences: torch.Tensor,
    ) -> Dict:
        """Common processing logic for both evaluation and training outputs.

        Args:
            questions (List[str]): List of questions
            ground_truths (List[str]): List of ground truth answers
            task_types (List[str]): List of task types
            input_ids (torch.Tensor): Input token IDs
            full_sequences (torch.Tensor): Generated sequences including prompts

        Returns:
            dict: Dictionary containing processed outputs including completions and rewards
        """
        # Validate inputs
        batch_size = full_sequences.size(0)
        assert len(questions) == len(ground_truths) == len(task_types) == batch_size

        prompt_lengths = (input_ids != self.pad_token_id).sum(dim=1).cpu()
        completion_ids = full_sequences[:, input_ids.size(1) :]
        completion_lengths = (completion_ids != self.pad_token_id).sum(dim=1).cpu()
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards
        reward_output = self._compute_rewards(completion_texts, ground_truths)

        return {
            'prompt_lengths': prompt_lengths,
            'completion_ids': completion_ids,
            'completion_lengths': completion_lengths,
            'completion_texts': completion_texts,
            **reward_output,
        }

    def _compute_rewards(self, completion_texts: List[str], ground_truths: List[str]) -> Dict[str, torch.Tensor]:
        """Compute rewards for completions against ground truth(s)

        Args:
            completion_texts: List of generated completion texts
            ground_truths: A list of ground truths

        Returns:
            Dict: containing accuracy, format and total rewards

        """
        assert len(completion_texts) == len(ground_truths)

        accuracy_rewards = []
        format_rewards = []
        for idx in range(len(completion_texts)):
            out_dict = self._compute_reward_single_sample(
                completion_texts[idx],
                ground_truths[idx],
                self.config.xml_format,
            )
            accuracy_rewards.append(out_dict['accuracy_reward'])
            format_rewards.append(out_dict['format_reward'])

        accuracy_rewards = torch.tensor(accuracy_rewards, dtype=self.torch_dtype)
        format_rewards = torch.tensor(format_rewards, dtype=self.torch_dtype)
        total_rewards = accuracy_rewards + format_rewards

        return {
            'accuracy_rewards': accuracy_rewards,
            'format_rewards': format_rewards,
            'total_rewards': total_rewards,
        }

    @staticmethod
    def _compute_reward_single_sample(
        completion: str,
        ground_truth: str,
        xml_format: bool = False,
    ) -> Dict[str, float]:
        """Compute rewards for a single completion in a separate process."""
        accuracy_score = math_problem_grader(completion, ground_truth)
        # scale format reward since we have binary reward for the accuracy
        format_score = 0.5 * format_structure_grader(completion) if xml_format else 0.0
        return {'accuracy_reward': accuracy_score, 'format_reward': format_score}

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
        loss_mask = batch.loss_mask.to(self.device)

        if self.config.normalize_advantages:
            advantages = masked_whiten(advantages, loss_mask)

        # PPO clipped surrogate PG loss
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
        pg_losses = -torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

        # First average over the sequence length, then average over the batch
        pg_loss = masked_mean(pg_losses, loss_mask, dim=1).mean()

        # Initialize metrics with common values
        metrics = {
            'pg_loss': pg_loss.detach().item(),
        }

        # Compute KL divergence if coefficient is positive
        if self.config.kl_loss_coef > 0:
            # Compute the KL divergence between the model and the reference model
            # per_token_kl = torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1

            # Clamp log differences for stability
            per_token_log_ratio = torch.clamp(ref_logprobs - pi_logprobs, min=-10, max=10)
            per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1.0
            # per_token_kl = torch.clamp(per_token_kl, min=-100.0, max=100.0)  # Prevent extreme large values

            kl = masked_mean(per_token_kl, loss_mask, dim=1).mean()
            kl_loss = self.config.kl_loss_coef * kl

            loss = pg_loss + kl_loss
            metrics.update(
                {
                    'total_loss': loss.detach().item(),
                    'kl_loss': kl_loss.detach().item(),
                    'kl': kl.detach().item(),
                }
            )
        else:
            loss = pg_loss

        return loss, metrics

    def _get_average_completion_length(self, window_size: int = 10) -> int:
        """Compute the moving average of completion lengths.

        Returns:
            float: Moving average length, defaulting to 200.0 if no data.
        """
        if len(self._completion_lengths) < 10:
            return 400.0

        values = np.array(list(self._completion_lengths))

        # Calculate moving average using numpy's convolve
        weights = np.ones(window_size) / window_size
        moving_averages = np.convolve(values, weights, mode='valid')

        # Get the last moving average value
        last_ma_value = moving_averages[-1]
        return last_ma_value

    def _get_explore_start_steps(self) -> int:
        """Compute exploration start steps based on moving average length.

        Returns:
            int: Number of steps to start exploration.
        """
        moving_average_length = self._get_average_completion_length()
        explore_start_steps = max(int(moving_average_length * self.config.explore_start_ratio), 10)
        return explore_start_steps

    def _create_reference_model(self, policy_model: PreTrainedModel) -> PreTrainedModel:
        """Create a reference model from the policy model"""
        ref_model = deepcopy(policy_model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = ref_model.eval()
        return ref_model

    def _eval_collate_function(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader during training"""
        pad_token_id = self.pad_token_id

        # Extract input_ids and attention_mask as lists of tensors
        input_ids_list = [sample['input_ids'] for sample in batch]
        attention_mask_list = [sample['attention_mask'] for sample in batch]

        # Dynamically pad to the longest sequence in the batch
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id, padding_side='left')
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0, padding_side='left')

        # Collect other fields
        questions = [sample['question'] for sample in batch]
        ground_truths = [sample['ground_truth'] for sample in batch]
        task_types = [sample['task_type'] for sample in batch]

        return {
            'input_ids': input_ids,  # Shape: [batch_size, max_seq_len_in_batch]
            'attention_mask': attention_mask,
            'questions': questions,
            'ground_truths': ground_truths,
            'task_types': task_types,
        }

    def _train_collate_function(self, batch: List[GRPOSample]) -> GRPOSample:
        """Collate function for DataLoader during training"""
        pad_token_id = self.pad_token_id
        torch_dtype = self.torch_dtype

        # Pad states and actions (long tensors)
        batch_state_ids = pad_sequence([item.states for item in batch], batch_first=True, padding_value=pad_token_id)
        batch_action_ids = pad_sequence([item.actions for item in batch], batch_first=True, padding_value=pad_token_id)

        # Pad loss_mask (boolean tensor)
        batch_loss_mask = pad_sequence([item.loss_mask for item in batch], batch_first=True, padding_value=False)

        # Pad advantages, pi_logprobs, and ref_logprobs (float tensors)
        batch_advantages = pad_sequence([item.advantages for item in batch], batch_first=True, padding_value=0.0).to(
            torch_dtype
        )
        batch_pi_logprobs = pad_sequence([item.pi_logprobs for item in batch], batch_first=True, padding_value=0.0).to(
            torch_dtype
        )
        batch_ref_logprobs = pad_sequence([item.ref_logprobs for item in batch], batch_first=True, padding_value=0.0).to(
            torch_dtype
        )

        return GRPOSample(
            states=batch_state_ids,
            actions=batch_action_ids,
            loss_mask=batch_loss_mask,
            pi_logprobs=batch_pi_logprobs,
            ref_logprobs=batch_ref_logprobs,
            advantages=batch_advantages,
        )

    def _log_sample_metrics(
        self,
        is_training: bool,
        task_types: List[str],
        questions: List[str],
        ground_truths: List[str],
        accuracy_rewards: torch.Tensor,
        format_rewards: torch.Tensor,
        total_rewards: torch.Tensor,
        prompt_lengths: torch.Tensor,
        completion_lengths: torch.Tensor,
        completion_texts: List[str],
    ) -> None:
        """Log sample metrics to tensorboard and metrics collector."""
        phase = 'training' if is_training else 'evaluation'
        obj_prefix = f'objective/{phase}'
        tok_prefix = f'tokens/{phase}'

        # Slightly condensed metrics batch
        metrics_batch = {
            f'{obj_prefix}/accuracy_reward': accuracy_rewards.tolist(),
            f'{obj_prefix}/format_reward': format_rewards.tolist(),
            f'{obj_prefix}/total_reward': total_rewards.tolist(),
            f'{tok_prefix}/prompt_length': prompt_lengths.tolist(),
            f'{tok_prefix}/completion_length': completion_lengths.tolist(),
        }
        for name, values in metrics_batch.items():
            self._metrics.add_metrics_batch(name, values)

        # Minor cleanup of sample logging
        handler = self._train_sample_handler if is_training else self._eval_sample_handler
        tb_tag = f'{phase}_samples'
        tb_indices = random.sample(range(len(completion_texts)), k=1) if random.random() > 0.75 else []

        for idx in range(len(completion_texts)):
            sample = SampleLog(
                question=questions[idx],
                task_type=task_types[idx],
                ground_truth=ground_truths[idx],
                completion=completion_texts[idx],
                accuracy_reward=accuracy_rewards[idx].item(),
                format_reward=accuracy_rewards[idx].item(),
                total_reward=total_rewards[idx].item(),
                completion_length=completion_lengths[idx].item(),
                step=self.iteration_count,
            )

            if is_training:
                self.train_episode_count += 1
                handler.log_entry(sample.model_dump())
            else:
                self.eval_episode_count += 1
                handler.log_entry(sample.model_dump())

            if idx in tb_indices:
                try:
                    self._log_sample_to_tensorboard(
                        tb_tag,
                        self._format_sample_text(sample),
                        self.train_episode_count if is_training else self.eval_episode_count,
                    )
                except Exception as e:
                    self.logger.error(f"Failed to log sample to TensorBoard: {e}")

        handler.flush()

    def _format_sample_text(self, sample: SampleLog) -> str:
        """Format sample text for TensorBoard logging."""
        return (
            f"**Question [{sample.task_type}]**: {sample.question}\n\n"
            f"**Ground Truth**: {sample.ground_truth}\n\n"
            f"**Accuracy Reward**: {sample.accuracy_reward:.2f}\n\n"
            f"**Format Reward**: {sample.format_reward:.2f}\n\n"
            f"**Generated Completion**:\n```json\n{sample.completion}\n```"
        )

    def _log_training_stats(self, stats: Dict[str, Any], step: int) -> None:
        """Log stats to external file and tensorboard"""
        self._train_stats_handler.log_entry({**stats, 'step': step})
        self._train_stats_handler.flush()
        self._log_stats_to_tensorboard(stats, step)

    def _log_sample_to_tensorboard(self, tag: str, formatted_text: str, step: int) -> None:
        """Log formatted text to TensorBoard.

        Args:
            tag (str): TensorBoard tag (e.g., 'training' or 'evaluation').
            formatted_text (str): Text to log.
            step (int): Episode number for logging.
        """
        if self._writer:
            try:
                self._writer.add_text(f'{tag}', formatted_text, step)
            except Exception as e:
                self.logger.warning(f"Failed to log sample to TensorBoard: {e}")

    def _log_hyper_params_to_tensorboard(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard.

        Args:
            config (Dict[str, Any]): Hyperparameters dictionary.
        """
        if self._writer and config:
            try:
                config_str = yaml.dump(config, sort_keys=False, indent=4)
                self._writer.add_text('config/parameters', f"```yaml\n{config_str}\n```", 0)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")

    def _log_stats_to_tensorboard(self, stats: Dict[str, Any], step: int) -> None:
        """Log stats to tensorboard"""
        if self._writer:
            try:
                for name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self._writer.add_scalar(f"{name}", value, step)
            except Exception as e:
                self.logger.warning(f"Failed to log stats to TensorBoard: {e}")
