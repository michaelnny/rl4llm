"""Implements RL GRPO algorithm to train LLM"""

import logging
import math
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from datasets import Dataset
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, set_seed

from rl4llm.generations import CustomLLMGenerator
from rl4llm.graders import math_problem_grader
from rl4llm.utils import Timer

logger = logging.getLogger(__name__)


class GRPOConfig(BaseModel):
    """GRPO training configuration"""

    """For RL sample generation"""
    system_prompt: Optional[str] = Field(None, description='System prompt for generation')
    max_new_tokens: Optional[int] = Field(4096, ge=100, description='Maximum number of new tokens to generate')
    temperature: Optional[float] = Field(0.9, gt=0.0, le=1.0, description='Sampling temperature for generation')
    top_k: Optional[int] = Field(0, ge=0, le=50000, description='Sampling top-k for generation')
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description='Sampling top-p for generation')
    do_sample: Optional[bool] = Field(True, description='Enable sampling for generation')
    group_size: int = Field(8, ge=4, le=256, description='Number of group outcomes for single question')

    # our enhancements to GRPO to encourage exploration
    use_group_temperature: Optional[bool] = Field(
        False, description='Use group temperatures instead of a single temperature to sample tokens during generation'
    )
    random_start_steps: int = Field(
        30,
        ge=0,
        le=128,
        description='Number of steps for random start by inject dirichlet noise to the probabilities distribution',
    )
    random_start_eps: float = Field(
        0.0, ge=0, le=0.25, description='Small eps to control weight of dirichlet noise vs original distribution'
    )
    random_start_alpha: float = Field(0.0, ge=0, le=0.1, description='Small alpha to control dirichlet noise distribution')
    explore_init_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Initial exploration epsilon')
    explore_min_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Minimum exploration epsilon after decay')

    """For RL GRPO training"""
    max_iterations: int = Field(10000, ge=1, description='How long to run the training')
    rollout_size: int = Field(1024, ge=1, le=5120, description='Number of samples to collect before update policy')
    num_updates: int = Field(1, ge=1, le=4, description='GRPO update epochs for a collection of samples')
    batch_size: int = Field(1, ge=1, le=256, description='Mini-batch size')
    gradient_accumulate_steps: int = Field(1, ge=1, le=32, description='Gradient accumulation steps')
    clip_eps: float = Field(0.2, ge=0.0, le=1.0, description='PPO policy loss clip epsilon')
    gamma: float = Field(1.0, ge=0.0, le=1.0, description='Fallback default discount factor for compute returns')
    normalize_group_rewards: bool = Field(True, description='Normalized group rewards')
    kl_loss_coef: float = Field(0.01, ge=0.0, le=1.0, description='KL penalty loss coefficient')
    sync_reference_interval: int = Field(
        0, ge=10, le=1000, description='Interval to update reference model using latest policy'
    )

    """Other configs"""
    seed: int = Field(167, ge=1, description='Runtime seed')
    checkpoint_interval: int = Field(0, ge=0, le=100, description='Interval to save policy model checkpoint')
    artifacts_path: str = Field(None, description='Path to save artifacts like checkpoints, tensorboard logs')


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

        self.tb_log_dir = os.path.join(self.config.artifacts_path, "tb_logs")
        self.checkpoint_dir = os.path.join(self.config.artifacts_path, "checkpoints")
        for _path in [self.tb_log_dir, self.checkpoint_dir]:
            if not os.path.exists(_path):
                os.makedirs(_path, exist_ok=True)

        set_seed(self.config.seed)

        logger.info(f"Artifacts will be saved at: {self.config.artifacts_path}")

        self.device = device
        self.torch_dtype = torch_dtype
        self.policy_model = policy_model
        self.reference_model = self._create_reference_model()
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.writer = SummaryWriter(self.tb_log_dir)
        self.train_ds = train_ds

        self.llm_generator = CustomLLMGenerator(self.policy_model)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = [self.tokenizer.eos_token, self.tokenizer.pad_token]

        self.generation_mode = False
        self.explore_epsilon = 0

        self.episode_count = 0
        self.update_count = 0
        self.iteration_count = 0
        self.ref_update_count = 0

    def _get_exploration_epsilon(self) -> float:
        """Computes epsilon value based on the current iteration step count."""
        if self.iteration_count <= 0:
            return self.config.explore_init_epsilon
        decay_rate = (self.config.explore_init_epsilon - self.config.explore_min_epsilon) / self.iteration_count
        self.explore_epsilon = max(
            self.config.explore_min_epsilon, self.config.explore_init_epsilon - decay_rate * self.episode_count
        )
        return self.explore_epsilon

    def _create_reference_model(self) -> PreTrainedModel:
        """Create a reference model from the policy model"""
        ref_model = deepcopy(self.policy_model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = ref_model.eval()
        return ref_model

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
        self._optimizer_to("cpu")
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
    def generate_group_samples(self, question: str, ground_truth: str) -> List[Dict]:
        """Generate responses for given question and ground truth answer

        Args:
            question (str): Question prompt
            ground_truth (str): Ground truth answer

        Returns:
            List[Dict]: List of samples for the group
        """
        if not self.config.system_prompt:
            message = [{"role": "user", "content": question.strip()}]
        else:
            message = [
                {"role": "system", "content": self.config.system_prompt.strip()},
                {"role": "user", "content": question.strip()},
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

        if self.config.use_group_temperature:
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
            'random_start_eps': self.config.random_start_eps,
            'random_start_alpha': self.config.random_start_alpha,
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
        completion_tokens_count = (completion_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards for group outcomes
        rewards = np.array([math_problem_grader(completion, ground_truth) for completion in completion_texts])

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

        # if random_start_steps > 0:
        #     loss_mask[:, : prompt_length - 1 + random_start_steps] = 0
        # else:
        #     loss_mask[:, : prompt_length - 1] = 0  # this will exclude prompt tokens up until the first completion token

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

            sample = {
                'states': states[i, :cut_position].cpu().tolist(),
                'actions': actions[i, :cut_position].cpu().tolist(),
                'loss_mask': loss_mask[i, :cut_position].cpu().tolist(),
                'reward': rewards[i],
                # this is essentially monte carlo return with no discount
                'advantages': (loss_mask[i, :cut_position].cpu() * normalized_rewards[i]).tolist(),
                'pi_logprobs': pi_logprobs[i, :cut_position].cpu().tolist(),
                'ref_logprobs': ref_logprobs[i, :cut_position].cpu().tolist(),
                'completion_text': completion_texts[i],
                'completion_length': completion_tokens_count[i],
            }

            assert (
                len(sample['states'])
                == len(sample['actions'])
                == len(sample['advantages'])
                == len(sample['pi_logprobs'])
                == len(sample['ref_logprobs'])
                == len(sample['loss_mask'])
            )
            results.append(sample)
            self.episode_count += 1

        sampled_items = random.choices(results, k=2)
        for sampled_item in sampled_items:
            self._log_sample_to_tensorboard(question, ground_truth, sampled_item['completion_text'], sampled_item['reward'])

        return results

    @staticmethod
    def normalize_group_rewards(rewards: np.ndarray) -> np.ndarray:
        """
        Normalize group rewards by subtracting the mean and dividing by the standard deviation.

        Args:
            rewards (np.ndarray): List of rewards for the group.

        Returns:
            np.ndarray: Normalized rewards.
        """
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)  # Add small value to avoid division by zero
        return normalized_rewards

    def train(self):
        """Train the model using RL GRPO"""
        for _ in tqdm(range(self.config.max_iterations), desc='Training iterations'):
            logger.info(f"Start iteration {self.iteration_count} ...")
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

        samples, generation_stats = self.generate_samples()
        with torch.autograd.set_detect_anomaly(True):
            train_stats = self.train_policy(samples)
        self.iteration_count += 1
        stats = {
            **generation_stats,
            **train_stats,
        }

        logger.info(f"Iteration stats:\n{stats}")

        self._log_stats_to_tensorboard(stats, self.iteration_count)

        self._handle_post_train()

    def generate_samples(
        self,
    ) -> Tuple[List[Dict], Dict]:
        """Generates samples using the current policy."""

        with self.generation_context():
            assert not self.policy_model.training
            assert not self.reference_model.training

            collected_samples = []
            with Timer() as timer:
                # Create the iterator once outside the loop
                data_iter = iter(self.train_ds)
                while len(collected_samples) < self.config.rollout_size:
                    try:
                        item = next(data_iter)  # Fetch the next batch
                    except StopIteration:
                        # Restart the iterator if all data is exhausted
                        self.train_ds = self.train_ds.shuffle(seed=None)
                        data_iter = iter(self.train_ds)
                        item = next(data_iter)

                    assert "question" in item and "ground_truth" in item

                    samples = self.generate_group_samples(item['question'], item['ground_truth'])

                    collected_samples.extend(samples)

            elapsed_time = timer.get_elapsed_time()

            stats = {
                'elapsed/generation_time': elapsed_time,
                'elapsed/time_per_episode': len(collected_samples) / elapsed_time,
                'elapsed/generation_episodes': self.episode_count,
                "objective/reward": np.mean([d['reward'] for d in collected_samples]).item(),
                "objective/reward_std": np.std([d['reward'] for d in collected_samples]).item(),
                "objective/completion_length": np.mean([d['completion_length'] for d in collected_samples]).item(),
                "objective/completion_length_std": np.std([d['completion_length'] for d in collected_samples]).item(),
                "other/explore_epsilon": self.explore_epsilon,
            }
        return collected_samples, stats

    def train_policy(self, samples: List[Dict]) -> Dict:
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

        total_steps = math.ceil(
            self.config.num_updates * len(samples) / (self.config.batch_size * self.config.gradient_accumulate_steps)
        )

        accumulated_stats = defaultdict(list)
        self.optimizer.zero_grad()

        assert self.policy_model.training

        mini_steps = 0
        mini_batch: Dict[str, torch.Tensor] = None

        with Timer() as timer:
            for _ in range(self.config.num_updates):
                for mini_batch in data_loader:
                    states = mini_batch['states'].to(self.device)
                    actions = mini_batch['actions'].to(self.device)

                    pi_logprobs = self._compute_action_logprobs(self.policy_model, states, actions)

                    behavior_logprobs = mini_batch["pi_logprobs"].to(self.device)
                    advantages = mini_batch["advantages"].to(self.device)
                    loss_mask = mini_batch["loss_mask"].to(self.device)
                    ref_logprobs = mini_batch["ref_logprobs"].to(self.device)
                    # Compute the KL divergence between the model and the reference model
                    per_token_kl = torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1

                    # PPO clipped surrogate PG loss
                    ratio = torch.exp(pi_logprobs - behavior_logprobs)
                    clipped_ratio = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
                    pg_losses = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

                    pg_loss = pg_losses[loss_mask].mean()
                    kl_penalties = self.config.kl_loss_coef * per_token_kl[loss_mask].mean()
                    loss = -pg_loss + kl_penalties

                    if self.config.gradient_accumulate_steps > 0:
                        loss /= self.config.gradient_accumulate_steps

                    loss.backward()

                    accumulated_stats['train/total_loss'].append(loss.detach().item())
                    accumulated_stats['train/pg_loss'].append(pg_loss.detach().item())
                    accumulated_stats['train/kl_penalty'].append(kl_penalties.detach().item())
                    accumulated_stats['train/kl'].append(per_token_kl[loss_mask].detach().sum(-1).mean().item())

                    mini_steps += 1

                    if mini_steps % self.config.gradient_accumulate_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.update_count += 1
                        mini_steps = 0

        elapsed_time = timer.get_elapsed_time()

        stats = {
            'elapsed/train_time': elapsed_time,
            'elapsed/train_updates': self.update_count,
            'elapsed/time_per_update': total_steps / elapsed_time,
            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
        }

        for k in accumulated_stats:
            stats[k] = np.mean(accumulated_stats[k]).item()

        return stats

    def save_checkpoint(self, save_dir: str):
        """Save policy model checkpoint following HF conventions"""
        self.policy_model.save_pretrained(save_dir)

    def _handle_post_train(self):
        """Handle post-training operations"""
        if self.iteration_count < 1:
            return

        if self.iteration_count % self.config.sync_reference_interval == 0:
            logger.info("Updating reference model...")
            self._sync_reference_model()

        if self.iteration_count % self.config.checkpoint_interval == 0:
            logger.info("Saving policy model checkpoint...")
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

    def _collate_function(self, batch: List[Dict], pad_token_id: int, torch_dtype: torch.dtype) -> Dict:
        """Collate function for DataLoader during training"""
        batch_size = len(batch)
        max_seq_len = max([len(item['states']) for item in batch])
        batch_state_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        batch_action_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        batch_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)

        batch_advantages = torch.full((batch_size, max_seq_len), 0, dtype=torch_dtype)
        batch_pi_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=torch_dtype)
        batch_ref_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=torch_dtype)

        for i, item in enumerate(batch):
            seq_len = len(item['states'])
            batch_state_ids[i, :seq_len] = torch.tensor(item['states'], dtype=torch.long)
            batch_action_ids[i, :seq_len] = torch.tensor(item['actions'], dtype=torch.long)
            batch_advantages[i, :seq_len] = torch.tensor(item["advantages"], dtype=self.torch_dtype)
            batch_pi_logprobs[i, :seq_len] = torch.tensor(item["pi_logprobs"], dtype=self.torch_dtype)
            batch_ref_logprobs[i, :seq_len] = torch.tensor(item["ref_logprobs"], dtype=self.torch_dtype)
            batch_loss_mask[i, :seq_len] = torch.tensor(item["loss_mask"], dtype=torch.bool)

        return {
            "states": batch_state_ids,
            "actions": batch_action_ids,
            "advantages": batch_advantages,
            "pi_logprobs": batch_pi_logprobs,
            "ref_logprobs": batch_ref_logprobs,
            "loss_mask": batch_loss_mask,
        }

    def _log_hyper_params_to_tensorboard(self, config: Dict[str, Any]):
        """Log hyper parameters used for the job"""
        if self.writer:
            config_str = yaml.dump(config, sort_keys=False, indent=4)
            self.writer.add_text("config/parameters", f"```yaml\n{config_str}\n```", 0)

    def _log_sample_to_tensorboard(self, question: str, ground_truth: str, completion_text: str, reward: float):
        """Log a sample text to tensorboard"""
        if self.writer:
            formatted_text = (
                f"**Question**: {question}\n\n"
                f"**Ground Truth**: {ground_truth}\n\n"
                f"**Graded Reward**: {reward}\n\n"
                f"**Full Answer**:\n```json\n{completion_text}\n```"
            )
            self.writer.add_text("sample", formatted_text, self.episode_count)

    def _log_stats_to_tensorboard(self, stats: Dict[str, Any], step: int):
        """Log stats to tensorboard"""
        if self.writer:
            for name, value in stats.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{name}", value, step)
