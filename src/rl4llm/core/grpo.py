"""Implements RL GRPO algorithm to train LLM"""

import logging
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.generations import CustomLLMGenerator
from rl4llm.graders import math_problem_grader
from rl4llm.utils import MetricsCollector

from .data_types import GRPOConfig, GRPOSample

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """RL GRPO for training LLMs"""

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
    ):
        self.config = config
        self._setup_directories()
        self.device = device
        self.torch_dtype = torch_dtype
        self.policy_model = policy_model.to(device)
        self.reference_model = self._create_reference_model()
        self.reference_model.to(device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        self.train_ds = train_ds.shuffle(seed=None)
        self.train_iter = iter(self.train_ds)

        self.llm_generator = CustomLLMGenerator(self.policy_model)

        # Special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.writer = SummaryWriter(self.tb_log_dir)
        self.metrics = MetricsCollector()
        self._initialize_counters()

    def _setup_directories(self):
        """Setup logging and checkpoint directories"""
        self.tb_log_dir = os.path.join(self.config.artifacts_path, 'tb_logs')
        self.checkpoint_dir = os.path.join(self.config.artifacts_path, 'checkpoints')

        for path in [self.tb_log_dir, self.checkpoint_dir]:
            os.makedirs(path, exist_ok=True)

        logger.info(f"Artifacts will be saved at: {self.config.artifacts_path}")

    def _initialize_counters(self):
        """Initialize training progress counters"""
        self.generation_mode = False
        self.explore_epsilon = 0
        self.episode_count = 0
        self.update_count = 0
        self.iteration_count = 0
        self.ref_update_count = 0

    def _get_exploration_epsilon(self) -> float:
        """Computes exploration epsilon based on the current iteration step count."""
        if self.config.explore_decay_steps == 0 or self.config.explore_init_epsilon == 0:
            return 0.0
        else:
            # Calculate the decay rate
            decay_rate = (self.config.explore_init_epsilon - self.config.explore_min_epsilon) / self.config.explore_decay_steps

            # Apply decay, ensuring epsilon doesn't go below the minimum
            self.explore_epsilon = max(
                self.config.explore_min_epsilon, self.config.explore_init_epsilon - decay_rate * self.iteration_count
            )
        return self.explore_epsilon

    def _create_reference_model(self) -> PreTrainedModel:
        """Create a reference model from the policy model"""
        ref_model = deepcopy(self.policy_model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = ref_model.eval()
        return ref_model

    def _get_next_data_item(self) -> Dict:
        """Fetches the next data item from the iterator, handles epoch reset."""
        try:
            item = next(self.train_iter)
            return item
        except StopIteration:
            # Epoch finished! Reshuffle and recreate the iterator
            self.train_ds = self.train_ds.shuffle(seed=None)
            self.train_iter = iter(self.train_ds)
            item = next(self.train_iter)  # Get the first item of the new epoch
            return item

    def train(self, hyper_params: Optional[Dict] = None):
        """Train the model using RL GRPO"""

        # log the params we use for this training run
        if hyper_params:
            self._log_hyper_params_to_tensorboard(hyper_params)

        for _ in tqdm(range(self.config.max_iterations), desc='Training iterations'):
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
        metrics_summary = self.metrics.get_summary()
        self._log_stats_to_tensorboard(metrics_summary, self.iteration_count)

        self._handle_post_train()

    @contextmanager
    def generation_context(self):
        """Context manager for handling model and optimizer states during generation"""
        try:
            self._prepare_for_generation()
            yield
        finally:
            self._prepare_for_training()

    def _optimizer_to(self, device: str):
        """Move pytorch optimizer to some device

        Code copied from
        https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
        """
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def _prepare_for_generation(self):
        """Move unnecessary components to CPU during generation"""
        if self.generation_mode:
            return

        # Move optimizer states to CPU
        self._optimizer_to('cpu')
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

        # Move optimizer states back to original devices
        self._optimizer_to(self.policy_model.device)

        self.policy_model = self.policy_model.train()

        torch.cuda.empty_cache()
        self.generation_mode = False

    def _compute_action_logprobs(
        self, model: PreTrainedModel, input_ids: torch.LongTensor, actions: torch.LongTensor
    ) -> torch.Tensor:
        """Compute log probabilities of actions given the input states"""

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
            sample_logits = logits[i : i + 1]  # Keep dim for proper broadcasting
            sample_logprobs_all = torch.log_softmax(sample_logits, dim=-1)
            sample_actions = actions[i : i + 1].unsqueeze(2)
            sample_logprob = torch.gather(sample_logprobs_all, dim=2, index=sample_actions).squeeze(2)
            sample_logprobs.append(sample_logprob)

        # Concatenate results
        return torch.cat(sample_logprobs, dim=0)

    @torch.no_grad()
    def generate_group_samples(self, question: str, ground_truth: str) -> List[GRPOSample]:
        """Generate responses for given question and ground truth answer

        Args:
            question (str): Question prompt
            ground_truth (str): Ground truth answer

        Returns:
            List[Dict]: List of samples for the group
        """
        if not self.config.system_prompt:
            message = [{'role': 'user', 'content': question.strip()}]
        else:
            message = [
                {'role': 'system', 'content': self.config.system_prompt.strip()},
                {'role': 'user', 'content': question.strip()},
            ]

        # expand to have a batch dimension
        message = [message for _ in range(self.config.group_size)]

        message_prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(
            message_prompt,
            return_tensors='pt',
            truncation=True,
            padding=True,
            padding_side='left',
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        if self.config.group_temperature:
            # Spread temperature values according to self.config.group_size, where 0.0 means greedy sampling
            # this idea is similar how we do it in distributed RL training in classical RL
            # where we have multiple agents running in parallel, some agents are more exploratory than others
            temperature = torch.linspace(
                0.0, self.config.temperature, steps=self.config.group_size, dtype=self.torch_dtype, device=self.device
            )
        else:
            # make code compatible
            temperature = torch.tensor(
                [self.config.temperature] * self.config.group_size, dtype=self.torch_dtype, device=self.device
            )

        random_start_steps = 0
        explore_epsilon = self._get_exploration_epsilon()
        if explore_epsilon is not None and explore_epsilon > 0 and random.random() < explore_epsilon:
            random_start_steps = self.config.random_start_steps

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'temperature': temperature,
            'random_start_steps': random_start_steps,
            'random_start_top_k': self.config.random_start_top_k,
            'max_new_tokens': self.config.max_new_tokens,
            # 'top_p': self.config.top_p,
            # 'top_k': self.config.top_k,
            'use_cache': True,
            'output_scores': False,
            'output_logits': False,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
        }

        outputs = self.llm_generator.generate(**generation_kwargs)

        full_sequences = outputs.sequences
        prompt_length = input_ids.size(1)
        completion_ids = full_sequences[:, prompt_length:]
        completion_tokens_count = (completion_ids != self.pad_token_id).sum(dim=1).cpu()
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards for group outcomes
        rewards = torch.tensor(
            [math_problem_grader(completion, ground_truth) for completion in completion_texts], dtype=self.torch_dtype
        )

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
            assert loss_mask[i, ...].sum().item() == completion_tokens_count[i]
            assert loss_mask[i, :cut_position].sum().item() == completion_tokens_count[i]

            # the GRPO advantages is essentially non-discounted monte carlo returns
            # with normalized reward at terminal time step, and all zero for non-terminal time step
            seq_rewards = torch.zeros_like(actions[i, :cut_position], dtype=self.torch_dtype)
            seq_rewards[-1] = normalized_rewards[i]
            returns = self.compute_masked_monte_carlo_returns(
                rewards=seq_rewards, mask=loss_mask[i, :cut_position], gamma=self.config.gamma
            )

            # old_returns = loss_mask[i, :cut_position].float() * normalized_rewards[i]
            # torch.testing.assert_close(returns.float(), old_returns.float())

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

            self.metrics.add_metric('objective/reward', rewards[i].item())
            self.metrics.add_metric('objective/completion_length', completion_tokens_count[i].item())

        # Randomly sample 1 items for logging
        sampled_indices = random.sample(range(len(completion_texts)), 1)
        for i in sampled_indices:
            self._log_sample_to_tensorboard(question, ground_truth, completion_texts[i], rewards[i].item())

        return results

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
                    item = self._get_next_data_item()
                    assert 'question' in item and 'ground_truth' in item

                    samples = self.generate_group_samples(item['question'], item['ground_truth'])

                    collected_samples.extend(samples)

            self.metrics.add_metric('elapsed/generation_episodes', self.episode_count)
            self.metrics.add_metric('elapsed/explore_epsilon', self.explore_epsilon)

            return collected_samples

    def train_policy(self, samples: List[GRPOSample]) -> None:
        random.shuffle(samples)

        _collate_fn = partial(self._collate_function, pad_token_id=self.pad_token_id, torch_dtype=self.torch_dtype)

        data_loader = DataLoader(
            samples,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=_collate_fn,
            drop_last=True,
        )

        self.optimizer.zero_grad()

        assert self.policy_model.training

        mini_steps = 0
        mini_batch: GRPOSample = None

        with self.metrics.timer('train'):
            for _ in range(self.config.num_updates):
                for mini_batch in data_loader:
                    states = mini_batch.states.to(self.device)
                    actions = mini_batch.actions.to(self.device)
                    # correctness = mini_batch.reward.bool().to(self.device)

                    pi_logprobs = self._compute_action_logprobs(self.policy_model, states, actions)

                    behavior_logprobs = mini_batch.pi_logprobs.to(self.device)
                    advantages = mini_batch.advantages.to(self.device)
                    loss_mask = mini_batch.loss_mask.to(self.device)
                    ref_logprobs = mini_batch.ref_logprobs.to(self.device)
                    # Compute the KL divergence between the model and the reference model
                    per_token_kl = torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1

                    # PPO clipped surrogate PG loss
                    ratio = torch.exp(pi_logprobs - behavior_logprobs)
                    clipped_ratio = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
                    pg_losses = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

                    pg_loss = pg_losses[loss_mask].mean()
                    kl_loss = self.config.kl_loss_coef * per_token_kl[loss_mask].mean()
                    loss = -pg_loss + kl_loss

                    if self.config.gradient_accumulate_steps > 1:
                        loss /= self.config.gradient_accumulate_steps

                    loss.backward()

                    # Record training metrics
                    self.metrics.add_metric('train/total_loss', loss.detach().item())
                    self.metrics.add_metric('train/pg_loss', pg_loss.detach().item())
                    self.metrics.add_metric('train/kl_loss', kl_loss.detach().item())
                    self.metrics.add_metric('train/kl', per_token_kl[loss_mask].detach().sum(-1).mean().item())

                    mini_steps += 1

                    if mini_steps % self.config.gradient_accumulate_steps == 0:
                        self.optimizer.step()
                        # self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.update_count += 1
                        mini_steps = 0

        self.scheduler.step()

        self.metrics.add_metric('elapsed/policy_update', self.update_count)
        self.metrics.add_metric('elapsed/reference_update', self.ref_update_count)
        self.metrics.add_metric('train/learning_rate', self.optimizer.param_groups[0]['lr'])

    def save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        self.policy_model.save_pretrained(save_dir)

    def _handle_post_train(self):
        """Handle post-training operations"""
        if self.iteration_count < 1:
            return

        if self.iteration_count % self.config.sync_reference_interval == 0:
            logger.info('Updating reference model...')
            self._sync_reference_model()

        if self.iteration_count % self.config.checkpoint_interval == 0:
            logger.info('Saving policy model checkpoint...')
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
    def normalize_group_rewards(rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize group rewards by subtracting the mean and dividing by the standard deviation.

        Args:
            rewards (torch.Tensor): List of rewards for the group.

        Returns:
            torch.Tensor: Normalized rewards.
        """
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards)
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)  # Add small value to avoid division by zero
        return normalized_rewards

    @staticmethod
    def _collate_function(batch: List[GRPOSample], pad_token_id: int, torch_dtype: torch.dtype) -> GRPOSample:
        """Collate function for DataLoader during training"""
        batch_size = len(batch)
        max_seq_len = max([len(item.states) for item in batch])
        batch_state_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        batch_action_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        batch_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)

        batch_advantages = torch.full((batch_size, max_seq_len), 0, dtype=torch_dtype)
        batch_pi_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=torch_dtype)
        batch_ref_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=torch_dtype)
        batch_reward = torch.full((batch_size,), 0, dtype=torch_dtype)

        for i, item in enumerate(batch):
            seq_len = len(item.states)
            batch_state_ids[i, :seq_len] = item.states.to(dtype=torch.long)
            batch_action_ids[i, :seq_len] = item.actions.to(dtype=torch.long)
            batch_advantages[i, :seq_len] = item.advantages.to(dtype=torch_dtype)
            batch_pi_logprobs[i, :seq_len] = item.pi_logprobs.to(dtype=torch_dtype)
            batch_ref_logprobs[i, :seq_len] = item.ref_logprobs.to(dtype=torch_dtype)
            batch_loss_mask[i, :seq_len] = item.loss_mask.to(dtype=torch.bool)
            batch_reward[i] = item.reward

        return GRPOSample(
            states=batch_state_ids,
            actions=batch_action_ids,
            advantages=batch_advantages,
            pi_logprobs=batch_pi_logprobs,
            ref_logprobs=batch_ref_logprobs,
            loss_mask=batch_loss_mask,
            reward=batch_reward,
            completion_length=None,
        )

    def _log_hyper_params_to_tensorboard(self, config: Dict[str, Any]):
        """Log hyper parameters used for the job"""
        if self.writer:
            config_str = yaml.dump(config, sort_keys=False, indent=4)
            self.writer.add_text('config/parameters', f"```yaml\n{config_str}\n```", 0)

    def _log_sample_to_tensorboard(self, question: str, ground_truth: str, completion_text: str, reward: float):
        """Log a sample text to tensorboard"""
        if self.writer:
            formatted_text = (
                f"**Question**: {question}\n\n"
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
