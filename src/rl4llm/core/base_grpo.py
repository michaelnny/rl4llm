"""Base GRPO trainer with common functionality for both single and distributed training"""

import logging
import os
import random
import re
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.graders import FormatGrader, MathGrader
from rl4llm.utils import FileHandler, masked_mean, masked_sum, masked_whiten, save_yaml_config_file

from .base_trainer import BaseTrainer
from .data_types import GRPOConfig, GRPOSample, SampleLog


class BaseGRPOTrainer(BaseTrainer):
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
        coherent_model_config: Optional[dict] = None,
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
            coherent_model_config (dict): Coherent model config.
            logger (Optional[logging.Logger]): Logger for training events; defaults to DummyLogger if None.
        """

        super().__init__(config, tokenizer, device, torch_dtype, artifacts_path, logger, rank)

        # For custom exploring start where we skip do exploration for the <think> token
        self.think_token_len = len(self.tokenizer.encode('<think>')) if self.config.xml_format else 0

        self._initialize_file_handler()

        # Initialize counters
        self.train_episode_count = 0
        self.eval_episode_count = 0
        self.update_count = 0
        self.iteration_count = 0
        self.ref_update_count = 0
        self.explore_epsilon = 0
        self.generation_mode = False

        self.math_grader: MathGrader = MathGrader(coherent_model_config, torch_dtype, device)
        self.format_grader: FormatGrader = FormatGrader()

    def _initialize_file_handler(self):
        """Initialize training components"""

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

    def _create_custom_llm_generator(self, policy_model: PreTrainedModel) -> CustomLLMGenerator:
        """Create a custom generator wrapped around the policy model, which supports group temperature and stochastic sampling"""
        # Try replacing the end token with "Wait" for some samples
        source_tokens = []
        # Determine which tokens should be replaced based on format
        if self.config.xml_format:
            source_tokens.append(self.tokenizer.encode('</think>')[0])
            source_tokens.append(self.tokenizer.encode(' </think>')[0])
            source_tokens.append(self.tokenizer.encode(':</think>')[0])
            source_tokens.append(self.tokenizer.encode('.</think>')[0])
        else:
            source_tokens.append(self.eos_token_id)

        self.special_tokens = ['Wait']
        target_tokens = [self.tokenizer.encode(f' {kwd}')[0] for kwd in self.special_tokens]

        # we should only make the replacement for reasoning tokens
        prevent_patterns = [
            self.tokenizer.encode('</think>'),
            self.tokenizer.encode(' </think>'),
            self.tokenizer.encode('<answer>'),
        ]

        return CustomLLMGenerator(
            model=policy_model,
            tokenizer=self.tokenizer,
            device=self.device,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            prevent_patterns=prevent_patterns,
        )

    def on_exit(self):
        super().on_exit()
        self._train_sample_handler.close()
        self._eval_sample_handler.close()
        self._train_stats_handler.close()

    def _train(self):
        """Train the model"""
        for _ in tqdm(range(self.config.max_steps), desc='Training steps', disable=not self.is_master):
            self.step()

    def step(self):
        """Run a single training iteration. Must be implemented by subclasses."""
        raise NotImplementedError('Subclasses must implement this method')

    @torch.inference_mode()
    def _evaluate_policy(self, policy_model: PreTrainedModel, test_loader: DataLoader) -> None:
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
            'repetition_penalty': 1.0,
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

    @torch.inference_mode()
    def _generate_group_samples(
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
        input_ids = input_ids.to(self.device).repeat(self.config.group_size, 1)
        attention_mask = attention_mask.to(self.device).repeat(self.config.group_size, 1)

        use_custom_generator = (
            generator is not None
            and hasattr(generator, 'generate')
            and (self.config.group_temperature or self.config.explore_init_epsilon > 0)
        )
        if not use_custom_generator:
            generator = policy_model

        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'repetition_penalty': self.config.repetition_penalty,
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
                    self.config.min_temperature,
                    self.config.max_temperature,
                    steps=self.config.group_size,
                    dtype=self.torch_dtype,
                    device=self.device,
                )
                # Round to 2 decimal places
                gen_kwargs['temperature'] = torch.round(temperature, decimals=2)

            check_correctness = partial(self.math_grader.__call__, ground_truth=ground_truth)
            explore_prob = self._get_exploration_epsilon()

            enable_exploring = (self.config.explore_start_steps > 0) and (explore_prob > 0) and (random.random() < explore_prob)

            # exploring start
            if enable_exploring:
                gen_kwargs['explore_start_steps'] = self.config.explore_start_steps
                gen_kwargs['explore_skip_n'] = self.think_token_len
                gen_kwargs['explore_top_k'] = self.config.explore_top_k

                # swaps special tokens like "</think>" with "Wait"
                if self.config.explore_max_replacements > 0:
                    gen_kwargs['correctness_callback'] = check_correctness
                    gen_kwargs['explore_replace_prob'] = explore_prob
                    gen_kwargs['explore_max_replacements'] = self.config.explore_max_replacements

        outputs = generator.generate(**gen_kwargs)

        self._clean_up()

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

    def normalize_group_rewards(self, rewards: torch.Tensor, zero_mean_only: bool = True, eps: float = 1e-8) -> torch.Tensor:
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
        std_reward = rewards.std()
        if zero_mean_only:
            return rewards - mean_reward
        if std_reward == 0.0:
            (rewards - mean_reward) / eps

        return (rewards - mean_reward) / std_reward

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

    @torch.inference_mode()
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
            **outputs,
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

        # # discard invalid samples
        # if torch.sum(outputs['accuracy_rewards']) == 0:
        #     self.logger.warning('Skipping samples with all zero rewards')
        #     return []

        self._log_sample_metrics(
            is_training=True,
            task_types=task_types,
            questions=questions,
            ground_truths=ground_truths,
            **outputs,
        )

        completion_lengths = outputs['completion_lengths']

        # Training specific processing
        rewards = outputs['accuracy_rewards']
        rewards = self.normalize_group_rewards(rewards) if self.config.normalize_group_rewards else rewards

        states = full_sequences[:, :-1]
        actions = full_sequences[:, 1:]
        pi_logprobs = self._compute_action_logprobs(policy_model, states, actions).cpu()
        self._clean_up()
        if self.config.kl_loss_coef > 0 and reference_model:
            ref_logprobs = self._compute_action_logprobs(reference_model, states, actions).cpu()
            self._clean_up()
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
            seq_rewards[-1] = rewards[i]  # important to use normalized rewards here

            gamma = self.compute_dynamic_discount(completion_lengths[i]) if self.config.dynamic_discount else self.config.gamma

            returns = self.compute_masked_monte_carlo_returns(
                rewards=seq_rewards, mask=loss_mask[i, :cut_position], gamma=gamma
            )

            samples.append(
                GRPOSample(
                    states=states[i, :cut_position].cpu(),
                    actions=actions[i, :cut_position].cpu(),
                    loss_mask=loss_mask[i, :cut_position].cpu(),
                    reward=rewards[i].cpu(),
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

        accuracy_rewards = self.math_grader(completion_texts, ground_truths)  # 0 or 1
        # format_rewards = self.format_grader(completion_texts, ground_truths, **{'xml_format': self.config.xml_format})  # 0 or 1

        accuracy_rewards = torch.tensor(accuracy_rewards, dtype=self.torch_dtype)
        # format_rewards = torch.tensor(format_rewards, dtype=self.torch_dtype)

        # alpha = 0.8  # Higher weight for accuracy
        # beta = 0.2  # Lower weight for format
        # total_rewards = alpha * accuracy_rewards + beta * format_rewards

        return {
            'accuracy_rewards': accuracy_rewards,
            # 'format_rewards': format_rewards,
        }

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
            sample_logits = logits[i, ...].float()
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
        loss_mask = batch.loss_mask.to(self.device)

        if self.config.normalize_advantages:
            advantages = masked_whiten(advantages, loss_mask)

        # PPO clipped surrogate PG loss
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
        pg_losses = -torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

        # First average over the sequence length, then average over the batch
        pg_loss = masked_mean(pg_losses, loss_mask, dim=1).mean()

        # Compute entropy for the policy
        # Convert log probabilities to probabilities first
        probs = torch.exp(pi_logprobs)
        entropies = -torch.sum(probs * pi_logprobs * loss_mask, dim=-1)
        entropy = entropies.mean()
        entropy_loss = self.config.entropy_loss_coef * entropy

        # Initialize metrics with common values
        metrics = {
            'loss/pg': pg_loss.detach().item(),
            'loss/entropy': entropy_loss.detach().item(),
            'entropy': entropy.detach().item(),
        }

        # Compute KL divergence if coefficient is positive
        if self.config.kl_loss_coef > 0:
            # Compute the KL divergence between the model and the reference model
            per_token_kl = torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1

            # # Clamp log differences for stability
            # per_token_log_ratio = torch.clamp(ref_logprobs - pi_logprobs, min=-20, max=20)
            # per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1.0
            # # per_token_kl = torch.clamp(per_token_kl, min=-100.0, max=100.0)  # Prevent extreme large values

            kl = masked_mean(per_token_kl, loss_mask, dim=1).mean()
            kl_loss = self.config.kl_loss_coef * kl

            loss = pg_loss + kl_loss + entropy_loss
            metrics.update(
                {
                    'loss/total': loss.detach().item(),
                    'loss/kl': kl_loss.detach().item(),
                    'kl': kl.detach().item(),
                }
            )
        else:
            loss = pg_loss + entropy_loss

        return loss, metrics

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
            sample_logits = logits[i, ...].float()
            sample_logprobs_all = torch.log_softmax(sample_logits, dim=-1)
            sample_actions = actions[i, ...].unsqueeze(1)
            sample_logprob = torch.gather(sample_logprobs_all, dim=1, index=sample_actions).squeeze(1)
            sample_logprobs.append(sample_logprob)

        # Concatenate results
        return torch.stack(sample_logprobs, dim=0)

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
        prompt_lengths: torch.Tensor,
        completion_lengths: torch.Tensor,
        completion_texts: List[str],
        **kwargs,
    ) -> None:
        """Log sample metrics to tensorboard and metrics collector."""
        phase = 'training' if is_training else 'evaluation'
        obj_prefix = f'objective/{phase}'
        tok_prefix = f'tokens/{phase}'

        # Slightly condensed metrics batch
        metrics_batch = {
            f'{obj_prefix}/accuracy_reward': accuracy_rewards.tolist(),
            f'{tok_prefix}/prompt_length': prompt_lengths.tolist(),
            f'{tok_prefix}/completion_length': completion_lengths.tolist(),
        }

        # checking for occurrence of special token
        for tok in self.special_tokens:
            # Create a regex pattern that matches the full word with word boundaries
            pattern = r'\b' + re.escape(tok.lower()) + r'\b'

            # Count occurrences of the pattern in each text
            counts = [len(re.findall(pattern, text.lower())) for text in completion_texts]
            metrics_batch[f'{tok_prefix}/{tok}_count'] = counts

        for name, values in metrics_batch.items():
            self._metrics.add_metrics_batch(name, values)

        # Sample logging
        handler = self._train_sample_handler if is_training else self._eval_sample_handler
        tb_tag = f'{phase}_samples'
        # Randomly select 1 sample for regular logging
        tb_indices = set(random.sample(range(len(completion_texts)), k=1))
        # # Add indices of samples with negative format rewards
        # negative_format_indices = {i for i, reward in enumerate(format_rewards) if reward < 0}
        # tb_indices.update(negative_format_indices)

        for idx in range(len(completion_texts)):
            sample = SampleLog(
                question=questions[idx],
                task_type=task_types[idx],
                ground_truth=ground_truths[idx],
                completion=completion_texts[idx],
                completion_length=completion_lengths[idx].item(),
                accuracy_reward=accuracy_rewards[idx].item(),
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
            # f"**Format Reward**: {sample.format_reward:.2f}\n\n"
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
