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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.graders import format_structure_grader, math_problem_grader
from rl4llm.utils import DummyLogger, MetricsCollector, masked_mean, masked_sum, masked_whiten, save_yaml_config_file

from .data_types import GRPOConfig, GRPOSample


class GRPOTrainer:
    """RL GRPO for training LLMs for reasoning on math tasks on a single GPU.

    This implementation uses GRPO policy optimization with KL-divergence regularization to fine-tune
    language models through reinforcement learning. Key features include:

    - Batched generation with variable exploration temperatures
    - Dynamic discount factor based on sequence length
    - Memory-efficient computation of action probabilities
    - Tensorboard logging and checkpoint management
    - Reward normalization and advantage estimation

    Args:
        config (GRPOConfig): Configuration object containing training parameters
        policy_model (PreTrainedModel): The policy model to be trained
        tokenizer (PreTrainedTokenizer): Tokenizer for the model
        optimizer (torch.optim.AdamW): Optimizer for policy updates
        scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler
        train_ds (Dataset): Training dataset
        device (torch.device): Device to run training on
        torch_dtype (torch.dtype): Data type for model parameters
        artifacts_path (str): Path to save checkpoints and logs
        logger (logging.Logger): Logger object for logging

    Attributes:
        episode_count (int): Number of episodes processed
        update_count (int): Number of policy updates performed
        iteration_count (int): Number of training iterations completed
        explore_epsilon (float): Exploration epsilon for epsilon-greedy exploration
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_ds: Dataset,
        device: torch.device,
        torch_dtype: torch.dtype,
        artifacts_path: str,
        logger: logging.Logger = None,
    ):
        self.config = config
        self.device = device
        self.torch_dtype = torch_dtype
        self.policy_model = policy_model.to(device)
        self.reference_model = self._create_reference_model()
        self.reference_model.to(device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        # Special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.llm_generator = CustomLLMGenerator(self.policy_model)

        self._train_collate_fn = partial(self._collate_function, pad_token_id=self.pad_token_id, torch_dtype=self.torch_dtype)

        self.train_ds = train_ds.shuffle(seed=None)
        self.train_iter = iter(self.train_ds)

        self.artifacts_path = artifacts_path

        self.logger = logger if logger else DummyLogger()  # use a dummy logger to make code easier

        self._initialize()

    def _initialize(self):
        """Initialize trainer by setup logging, directories, and counters"""
        self.tb_log_dir = os.path.join(self.artifacts_path, 'tb_logs')
        self.checkpoint_dir = os.path.join(self.artifacts_path, 'checkpoints')

        for path in [self.tb_log_dir, self.checkpoint_dir]:
            os.makedirs(path, exist_ok=True)

        self.logger.info(f"Artifacts will be saved at: {self.artifacts_path!r}")

        self.writer = SummaryWriter(self.tb_log_dir)
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
        if hyper_params:
            save_yaml_config_file(hyper_params, os.path.join(self.artifacts_path, 'hyper_params.yaml'))
            self._log_hyper_params_to_tensorboard(hyper_params)

        for _ in tqdm(range(self.config.max_steps), desc='Training steps'):
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
            samples = self.generate_samples()
            with torch.autograd.set_detect_anomaly(True):
                self.train_policy(samples)

        self.iteration_count += 1

        # Log all metrics
        metrics = self._get_metrics_summary()
        self._log_stats_to_tensorboard(metrics, self.iteration_count)

        self._handle_post_train()

    @torch.no_grad()
    def generate_group_samples(
        self,
    ) -> List[GRPOSample]:
        """Generate responses for a batch of questions and ground truth answers

        Args:
            None
        Returns:
            List[Dict]: List of samples for all groups in the batch
        """

        sample: Dict[str, str] = self._get_next_data_item()
        # Prepare messages for the entire batch
        question = sample['question']
        ground_truth = sample['ground_truth']
        task_type = sample['task_type'].upper()

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
            padding=False,
            padding_side='left',
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

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

        generation_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'max_new_tokens': self.config.max_new_tokens,
            'temperature': temperature,
            'enable_exploration': enable_exploration,
            'explore_start_steps': random.choice(range(10, self.config.explore_start_steps)),
            'explore_uncertainty': self.config.explore_uncertainty,
            'explore_top_k': self.config.explore_top_k,
            'explore_top_k_beta': self.config.explore_top_k_beta,
            'do_sample': True,
            'use_cache': True,
            'output_scores': False,
            'output_logits': False,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
        }

        outputs = self.llm_generator.generate(**generation_kwargs)

        return self._process_generation_outputs(question, ground_truth, task_type, inputs.input_ids, outputs.sequences)

    def generate_samples(
        self,
    ) -> List[GRPOSample]:
        """Generates samples using the current policy."""

        with self.generation_context():
            assert not self.policy_model.training
            assert not self.reference_model.training
            collected_samples: List[GRPOSample] = []

            with self.metrics.timer('generation'):
                while len(collected_samples) < self.config.rollout_size:
                    samples = self.generate_group_samples()
                    collected_samples.extend(samples)

            self.metrics.add_metric('elapsed/generation_episodes', self.episode_count)
            self.metrics.add_metric('elapsed/explore_epsilon', self.explore_epsilon)

            return collected_samples

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
    def generation_context(self):
        """Context manager for handling model and optimizer states during generation"""
        try:
            self._prepare_for_generation()
            yield
        finally:
            self._prepare_for_training()

    def save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        self.policy_model.save_pretrained(save_dir)

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return self.metrics.get_summary()

    def _prepare_for_generation(self):
        """Move unnecessary components to CPU during generation"""
        if self.generation_mode:
            return

        self.policy_model = self.policy_model.eval()
        self.reference_model = self.reference_model.eval()
        # Ensure both models are on GPU for generation
        self.policy_model = self.policy_model.to(self.device)
        self.reference_model = self.reference_model.to(self.device)

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
        # scaled_loss = loss / self.config.gradient_accumulate_steps
        # scaled_loss.backward()
        loss.backward()

        # These metrics will later be accumulated over mini batches
        self.metrics.add_metric('train/total_loss', loss.detach().item())
        self.metrics.add_metric('train/pg_loss', pg_loss.detach().item())
        self.metrics.add_metric('train/kl_loss', kl_loss.detach().item())
        self.metrics.add_metric('train/kl', kl.detach().item())

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
        self, question: str, ground_truth: str, task_type: str, input_ids: torch.Tensor, full_sequences: torch.Tensor
    ):
        """Turn generated sample sequences into training sample

        Args:
            question (str): The question text
            ground_truth (str): The ground truth answer
            task_type (str): The task type
            input_ids (torch.Tensor): The prompt only input token ids, shape [batch_size, prompt_seq_len]
            full_sequences (torch.Tensor): The full (prompt + generated) sequences token ids, shape [batch_size, seq_len]

        """
        prompt_length = input_ids.size(1)
        completion_ids = full_sequences[:, prompt_length:]
        completion_tokens_count = (completion_ids != self.pad_token_id).sum(dim=1).cpu()
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards for group outcomes
        accuracy_rewards = []
        format_rewards = []
        for completion, completion_len in zip(completion_texts, completion_tokens_count.tolist()):
            # Check for correctness
            accuracy_rewards.append(math_problem_grader(completion, ground_truth))

            # Check for general rules like format, sequence length etc
            format_rewards.append(format_structure_grader(completion, self.config.xml_format))

        accuracy_rewards = torch.tensor(accuracy_rewards, dtype=self.torch_dtype)
        format_rewards = torch.tensor(format_rewards, dtype=self.torch_dtype)

        rewards = accuracy_rewards + format_rewards

        # Normalize rewards
        if self.config.normalize_group_rewards:
            normalized_rewards = self.normalize_group_rewards(rewards)
        else:
            normalized_rewards = rewards

        states = full_sequences[:, :-1]
        actions = full_sequences[:, 1:]
        pi_logprobs = self._compute_action_logprobs(self.policy_model, states, actions).cpu()
        ref_logprobs = self._compute_action_logprobs(self.reference_model, states, actions).cpu()

        # Do not include the prompt or pad tokens in the loss
        # for example, if we have a sequence token ids: [1, 2, 3, 4, 5, 6, 7, -1, -1]
        # where [1, 2, 3, 4] are the prompt tokens
        # and [5, 6, 7] are the completion tokens
        # -1 is the pad token
        # the, the loss mask will be [0, 0, 0, 1, 1, 1, 0, 0, 0]
        loss_mask = (actions != self.pad_token_id).bool()
        loss_mask[:, : prompt_length - 1] = 0  # this will exclude prompt tokens up until the first completion token

        # construct a list of samples by trim the sequence to the first EOS token
        results = []
        eos_mask = actions == self.eos_token_id
        eos_mask[:, :prompt_length] = False  # Ignore EOS tokens in the prompt

        # Calculate cut positions starting from completion
        cut_positions = torch.where(
            eos_mask.any(dim=1),
            eos_mask.float().argmax(dim=1) + 1,
            actions.size(1) + 1,  # use full sequence length if no EOS token found
        )

        # Cut sequences to the first eos token in completion
        for i, cut_position in enumerate(cut_positions):
            assert completion_tokens_count[i] > 0
            assert loss_mask[i, ...].sum().item() == completion_tokens_count[i]
            assert loss_mask[i, :cut_position].sum().item() == completion_tokens_count[i]

            # the GRPO advantages is essentially non-discounted monte carlo returns
            # with normalized reward at terminal time step, and all zero for non-terminal time step
            seq_rewards = torch.zeros_like(actions[i, :cut_position], dtype=self.torch_dtype)
            seq_rewards[-1] = normalized_rewards[i]

            if self.config.dynamic_discount:
                gamma = self.compute_dynamic_discount(
                    completion_tokens_count[i].item(), min_gamma=self.config.min_gamma, max_gamma=self.config.max_gamma
                )
            else:
                gamma = self.config.gamma
            returns = self.compute_masked_monte_carlo_returns(
                rewards=seq_rewards, mask=loss_mask[i, :cut_position], gamma=gamma
            )

            sample = GRPOSample(
                states=states[i, :cut_position].cpu(),
                actions=actions[i, :cut_position].cpu(),
                loss_mask=loss_mask[i, :cut_position].cpu(),
                reward=rewards[i].cpu(),
                advantages=returns.cpu(),
                pi_logprobs=pi_logprobs[i, :cut_position].cpu(),
                ref_logprobs=ref_logprobs[i, :cut_position].cpu(),
            )

            results.append(sample)
            self.episode_count += 1

            self.metrics.add_metric('objective/accuracy_reward', accuracy_rewards[i].item())
            self.metrics.add_metric('objective/format_reward', format_rewards[i].item())
            self.metrics.add_metric('objective/total_rewards', rewards[i].item())
            self.metrics.add_metric('objective/completion_length', completion_tokens_count[i].item())

        # Randomly sample 1 items for logging
        sampled_indices = random.sample(range(len(completion_texts)), 1)
        for i in sampled_indices:
            self._log_sample_to_tensorboard(task_type, question, ground_truth, completion_texts[i], rewards[i].item())

        return results

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

    @staticmethod
    def compute_masked_monte_carlo_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.FloatTensor:
        """
        Computes monte carlo returns considering only assistant turns.

        Args:
            rewards (torch.Tensor): Float tensor with rewards (0 for user), shape [seq_len]
            mask (torch.Tensor): Binary mask (0 for user, 1 for assistant), shape [seq_len]

        Returns:
            torch.Tensor: Tensor of the original shape, with discounted returns
                for assistant turns and zeros for user turns
        """
        # Input validation
        assert rewards.dim() == mask.dim() == 1, 'Inputs must be 1-dimensional'
        assert rewards.size(0) == mask.size(0), 'Rewards and mask must have same length'

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
        assert rewards.dim() == 1, 'Rewards must be 1-dimensional'
        if len(rewards) <= 1:
            return rewards

        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)

        if std_reward < eps:
            return torch.zeros_like(rewards)

        return (rewards - mean_reward) / (std_reward + eps)

    @staticmethod
    def _collate_function(batch: List[GRPOSample], pad_token_id: int, torch_dtype: torch.dtype) -> GRPOSample:
        """Collate function for DataLoader during training"""
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

    @staticmethod
    def compute_dynamic_discount(
        episode_length: int, min_gamma: float = 0.999, max_gamma: float = 0.9998, max_length: int = 8000
    ) -> float:
        """Compute dynamic discount factor."""
        assert episode_length > 0, 'Episode length must be greater than 0.'
        assert max_length > 1000, 'Max length must be greater than 1000.'
        assert 0.0 < min_gamma and min_gamma < 1.0, 'Min discount must be in the range (0, 1).'
        assert 0.0 < max_gamma and max_gamma < 1.0, 'Max discount must be in the range (0, 1).'
        assert min_gamma < max_gamma, 'Min discount must be less than max discount.'
        scaled_length = min(episode_length / max_length, 1.0)
        gamma = min_gamma + (max_gamma - min_gamma) * scaled_length
        return gamma

    def _log_hyper_params_to_tensorboard(self, config: Dict[str, Any]):
        """Log hyper parameters used for the job"""
        if self.writer:
            config_str = yaml.dump(config, sort_keys=False, indent=4)

            self.writer.add_text('config/parameters', f"```yaml\n{config_str}\n```", 0)

    def _log_sample_to_tensorboard(self, task: str, question: str, ground_truth: str, completion_text: str, reward: float):
        """Log a sample text to tensorboard"""
        if self.writer:
            formatted_text = (
                f"**Question [{task}]**: {question}\n\n"
                f"**Ground Truth**: {ground_truth}\n\n"
                f"**Graded Reward**: {reward}\n\n"
                f"**Full Answer**:\n```json\n{completion_text}\n```"
            )
            self.writer.add_text('sample', formatted_text, self.episode_count)

    def _log_stats_to_tensorboard(self, stats: Dict[str, Any], step: int):
        """Log stats to tensorboard"""
        if self.writer:
            for name, value in stats.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{name}", value, step)
