import math
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import torch
from deepspeed import DeepSpeedEngine
from datasets import Dataset
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.core.base_trainer import RLConfig, RLTrainer
from rl4llm.core.distributed import DistributedManager
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.logging import LoggingManager
from rl4llm.envs import LLMEnv


class GRPOConfig(RLConfig):
    """GRPO config instance for RL LLM"""

    ...


class TransitionData(BaseModel):
    """GPPO transition for training"""

    states: torch.Tensor = Field(
        ...,
        description="A long tensor for token sequences from t=0, 1, ..., T-1",
    )
    actions: torch.Tensor = Field(
        ...,
        description="A long tensor for token sequences from t=1, 2, ..., T-1, T",
    )
    loss_mask: torch.Tensor = Field(
        ...,
        description="A boolean tensor (0s user tokens, 1s assistant tokens) corresponding to token sequences from t=1, 2, ..., T-1, T",
    )
    pi_logprobs: torch.Tensor = Field(
        ...,
        description="A float tensor for action logprobs corresponding to token sequences from t=1, 2, ..., T-1, T",
    )
    ref_logprobs: torch.Tensor = Field(
        ...,
        description="A float tensor for action logprobs from reference model corresponding to token sequences from t=1, 2, ..., T-1, T",
    )
    advantages: torch.Tensor = Field(
        ...,
        description="A float tensor for GAE advantages estimate corresponding to token sequences from t=1, 2, ..., T-1, T",
    )

    @model_validator(mode="after")
    def check_tensor_shapes(cls, values):
        tensors = [
            values.states,
            values.actions,
            values.loss_mask,
            values.pi_logprobs,
            values.ref_logprobs,
            values.advantages,
        ]

        # Ensure all tensors are of the same shape
        tensor_shapes = [
            tensor.shape if isinstance(tensor, torch.Tensor) else None
            for tensor in tensors
        ]

        if len(set(tensor_shapes)) > 1:
            raise ValueError(f"Tensors have mismatched shapes: {tensor_shapes}")

        return values

    class Config:
        arbitrary_types_allowed = True


class GRPOTrainer(RLTrainer):
    """"""

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy_engine: DeepSpeedEngine,
        dist_manager: DistributedManager,
        logger: LoggingManager,
        artifacts_path: str,
        train_env: LLMEnv,
        eval_env: Optional[LLMEnv] = None,
        # train_dataset: Union[List[Dict] | Dataset],
        # eval_dataset: Optional[Union[List[Dict] | Dataset]] = None,
        seed: Optional[int] = 175,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            policy_engine=policy_engine,
            dist_manager=dist_manager,
            logger=logger,
            artifacts_path=artifacts_path,
            train_env=train_env,
            eval_env=eval_env,
            # train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            seed=seed,
        )

        self.math_grader = math_problem_grader

    def _initialize_trainer(self):
        """Initialize GRPO specific settings"""

        # avoid adding group of samples with almost identical outcomes
        _dummy_rewards = torch.tensor([0] * self.config.group_size, dtype=torch.float32)
        _idx = math.ceil(self.config.group_size * 0.05)
        _dummy_rewards[:_idx] = 1.0
        self.group_reward_std_threshold = torch.std(_dummy_rewards, unbiased=False)

    @torch.inference_mode()
    def _generate_group_samples(self, item: Dict[str, str]) -> List[TransitionData]:
        """Generate responses for a batch of questions and ground truth answers

        Args:
            item (Dict[str, str]): Dictionary with 'question', 'ground_truth', and 'task_type'.

        Returns:
            List[Dict]: List of samples for all groups in the batch
        """

        # # Prepare messages for the entire batch
        # task_type = item['task_type'].upper()
        # question = item['question']
        # ground_truth = item['ground_truth']
        # assert isinstance(question, str)
        # if task_type not in ['MATH', 'GSM']:
        #     raise ValueError(
        #         f"Invalid task type: {task_type}, only support 'MATH' or 'GSM'"
        #     )

        # input_ids = item['input_ids']
        # attention_mask = item['attention_mask']

        # if (
        #     self.config.max_prompt_tokens >= 512
        #     and input_ids.size(0) > self.config.max_prompt_tokens
        # ):
        #     self.logger.warning(
        #         f"Skip sample with prompt size grater than {self.config.max_prompt_tokens}"
        #     )
        #     return []

        # # Expand to have a "group" batch dimension
        # input_ids = input_ids.to(self.device).repeat(self.config.group_size, 1)
        # attention_mask = attention_mask.to(self.device).repeat(
        #     self.config.group_size, 1
        # )

        gen_kwargs = {
            # 'input_ids': input_ids,
            # 'attention_mask': attention_mask,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": self.config.max_completion_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": True,
            "use_cache": True,
            "output_scores": False,
            "output_logits": False,
            "return_dict_in_generate": True,
            "return_legacy_cache": False,
        }

        generator = self.policy_model
        outputs = self.train_env.rollout(self.policy_model, gen_kwargs)

        return outputs

        # return self._process_training_samples(
        #     question,
        #     ground_truth,
        #     task_type,
        #     input_ids,
        #     outputs.sequences,
        # )

    def _process_training_samples(
        self,
        question: str,
        ground_truth: str,
        task_type: str,
        input_ids: torch.Tensor,
        full_sequences: torch.Tensor,
    ) -> List[TransitionData]:
        """Process generated outputs after generation.

        Args:
            question: Single question
            ground_truth: Single ground truth
            task_type: Single task type
            input_ids: Prompt token ids [batch_size, prompt_seq_len]
            full_sequences: Full sequence token ids [batch_size, full_seq_len]

        Returns:
            List[GRPOSample] for training
        """
        # Standardize single inputs to lists
        batch_size = full_sequences.size(0)
        questions = [question] * batch_size
        ground_truths = [ground_truth] * batch_size
        task_types = [task_type] * batch_size

        prompt_length = input_ids.size(1)

        outputs, reward_dict = self._process_generation_common_outputs(
            questions=questions,
            ground_truths=ground_truths,
            task_types=task_types,
            input_ids=input_ids,
            full_sequences=full_sequences,
        )

        rewards = reward_dict["accuracy_reward"]

        # # discard samples as they leads to zero advantages -> zero gradients
        # if (
        #     torch.std(rewards, unbiased=False)
        #     <= self.group_reward_std_threshold
        # ):
        #     self.logger.warning(
        #         f'Skipping samples with identical rewards, \
        #             minimum group reward std: {self.group_reward_std_threshold}'
        #     )
        #     self.logger.log_scalar('generation/skipped_sample', len(rewards))
        #     return []

        completion_lengths = outputs["completion_lengths"]

        # Training specific processing
        rewards = (
            self.normalize_group_rewards(rewards)
            if self.config.normalize_rewards
            else rewards
        )

        states = full_sequences[:, :-1]
        actions = full_sequences[:, 1:]

        attention_mask = (states != self.tokenizer.pad_token_id).bool()
        pi_logits = self.policy_engine.forward(
            input_ids=states, attention_mask=attention_mask
        ).logits
        pi_logprobs = self.compute_logprobs_from_logits(pi_logits, actions).cpu()
        self.clean_up()
        if self.config.kl_loss_coef > 0 and self.reference_model:
            ref_pi_logits = self.reference_model.forward(
                input_ids=states, attention_mask=attention_mask
            ).logits
            ref_logprobs = self.compute_logprobs_from_logits(
                ref_pi_logits, actions
            ).cpu()
            self.clean_up()
        else:
            ref_logprobs = torch.full_like(
                pi_logprobs, 1e-6
            ).cpu()  # use a place holder to make sure code is compatible

        # TODO can we improve this code of post-processing sample creation??

        # Do not include the prompt or pad tokens in the loss
        # for example, if we have a sequence token ids: [1, 2, 3, 4, 5, 6, 7, -1, -1]
        # where [1, 2, 3, 4] are the prompt tokens
        # and [5, 6, 7] are the completion tokens
        # -1 is the pad token
        # the, the loss mask will be [0, 0, 0, 1, 1, 1, 0, 0, 0]
        loss_mask = (actions != self.tokenizer.pad_token_id).bool()
        loss_mask[:, : prompt_length - 1] = False

        samples = []

        # construct a list of samples by trim the sequence to the first EOS token and ignore EOS tokens in the prompt
        eos_mask = actions == self.tokenizer.eos_token_id
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

            # Compute discounted return
            # seq_rewards = torch.zeros_like(actions[i, :cut_position], dtype=self.torch_dtype).cpu()
            # seq_rewards[-1] = rewards[i]
            # gamma = self.compute_dynamic_discount(completion_lengths[i]) if self.config.dynamic_discount else self.config.gamma
            # returns = self.compute_masked_monte_carlo_returns(
            #     rewards=seq_rewards, mask=loss_mask[i, :cut_position].cpu(), gamma=gamma
            # )

            returns = rewards[i] * loss_mask[i, :cut_position].cpu()
            assert torch.nonzero(returns).sum() > 0

            # for sample in samples:
            #     # self.logger.log_scalar("perplexity", eval_perplexity, step)
            #     self.logger.log_sample("training", {"prompt": sample., "response": eval_response}, step)

            samples.append(
                TransitionData(
                    states=states[i, :cut_position].cpu(),
                    actions=actions[i, :cut_position].cpu(),
                    loss_mask=loss_mask[i, :cut_position].cpu(),
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
    ) -> Tuple[Dict, Dict]:
        """Common processing logic for both evaluation and training outputs.

        Args:
            questions (List[str]): List of questions
            ground_truths (List[str]): List of ground truth answers
            task_types (List[str]): List of task types
            input_ids (torch.Tensor): Input token IDs
            full_sequences (torch.Tensor): Generated sequences including prompts

        Returns:
            Tuple[Dict, Dict]: Dictionary containing processed outputs including completions and rewards
        """
        # Validate inputs
        batch_size = full_sequences.size(0)
        assert len(questions) == len(ground_truths) == len(task_types) == batch_size

        prompt_lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).cpu()
        completion_ids = full_sequences[:, input_ids.size(1) :]
        completion_lengths = (
            (completion_ids != self.tokenizer.pad_token_id).sum(dim=1).cpu()
        )
        completion_texts = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Compute rewards
        reward_dict = self.compute_rewards(completion_texts, ground_truths)

        return {
            "questions": questions,
            "prompt_lengths": prompt_lengths,
            "completion_ids": completion_ids,
            "completion_lengths": completion_lengths,
            "completion_texts": completion_texts,
        }, reward_dict

    def compute_rewards(
        self, completion_texts: List[str], ground_truths: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for completions against ground truth(s)

        Args:
            completion_texts: List of generated completion texts
            ground_truths: A list of ground truths

        Returns:
            Dict: containing accuracy, format and total rewards

        """
        assert len(completion_texts) == len(ground_truths)

        accuracy_rewards = [
            self.math_grader(full_answer=answer, **{"ground_truth": truth})
            for answer, truth in zip(completion_texts, ground_truths)
        ]  # 0 or 1
        accuracy_rewards = torch.tensor(accuracy_rewards, dtype=self.torch_dtype)

        # format_rewards = [0] * len(completion_texts) if not self.config.xml_format else self.format_grader(completion_texts, ground_truths)  # 0 or 1
        # format_rewards = torch.tensor(format_rewards, dtype=self.torch_dtype)

        # beta = 0.2  # Lower weight for format
        # total_rewards = accuracy_rewards + beta * format_rewards

        return {
            "accuracy_reward": accuracy_rewards,
            # 'format_reward': format_rewards,
        }

    def normalize_group_rewards(
        self,
        rewards: torch.Tensor,
        zero_mean_only: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize group rewards by subtracting the mean and dividing by the standard deviation.

        Args:
            rewards (torch.Tensor): List of rewards for the group.
            eps (float): Small value to prevent division by zero.

        Returns:
            torch.Tensor: Normalized rewards.
        """
        assert eps > 0.0, "Epsilon must be positive"
        assert rewards.dim() == 1, "Rewards must be 1-dimensional"
        if len(rewards) <= 1:
            return rewards

        mean_reward = rewards.mean()
        std_reward = rewards.std(unbiased=False)
        if zero_mean_only:
            return rewards - mean_reward

        return (rewards - mean_reward) / (std_reward + eps)

    def generate_experience(self) -> List[TransitionData]:
        """Generates samples using the current policy."""

        assert not self.policy_model.training
        # assert not self.reference_model.training
        collected_samples: List[TransitionData] = []

        local_rollout_size = self.config.rollout_size // self.dist_manager.world_size

        with self.logger.timer("generation"):
            while len(collected_samples) < local_rollout_size:
                data = self.get_next_train_data()  # this is the prompt-only loader
                samples = self._generate_group_samples(data)
                if samples:
                    collected_samples.extend(samples)

            if len(collected_samples) > local_rollout_size:
                collected_samples = collected_samples[:local_rollout_size]

        # TODO compute stats

        # self.logger.log_scalar('generation/episodes_total', self.train_episode_count)
        # self.logger.log_scalar('generation/explore_epsilon', self.explore_epsilon)

        return collected_samples

    def build_train_batch(self, experience: List[TransitionData]) -> DataLoader:
        data_loader = DataLoader(
            experience,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            collate_fn=self._train_collate_fn,
            drop_last=True,
        )

        return data_loader

    def compute_loss(
        self, pi_logits: torch.Tensor, experience_batch: TransitionData
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single training batch

        Args:
            pi_logits (torch.Tensor): Raw logits of actions computed using
                current policy, shape [batch_size, seq_len]
            experience_batch (TransitionData): A batch of samples collected
                during generation

        Returns:
            Tuple[torch.Tensor, Dict]: Tuple containing the total loss tensor
                and a dictionary of metrics
        """
        behavior_logprobs = experience_batch.pi_logprobs.to(self.device)
        actions = experience_batch.actions.to(self.device)
        advantages = experience_batch.advantages.to(self.device)
        loss_mask = experience_batch.loss_mask.to(self.device)
        ref_logprobs = experience_batch.ref_logprobs.to(self.device)
        loss_mask = experience_batch.loss_mask.to(self.device)

        if self.config.normalize_advantages:
            advantages = self.masked_whiten(advantages, loss_mask)

        # PPO clipped surrogate PG loss
        pi_logprobs = self.compute_logprobs_from_logits(pi_logits, actions)
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
        pg_losses1 = ratio * advantages.detach()
        pg_losses2 = clipped_ratio * advantages.detach()
        pg_losses = -torch.min(pg_losses1, pg_losses2)

        approxkl = 0.5 * self.masked_mean(
            torch.square(pi_logprobs - behavior_logprobs), loss_mask
        )
        clipfrac = self.masked_mean(torch.lt(pg_losses2, pg_losses1), loss_mask)

        # First average over the sequence length, then average over the batch
        pg_loss = self.masked_mean(pg_losses, loss_mask, dim=1).mean()

        # Compute entropy for the policy
        # Convert log probabilities to probabilities first
        probs = torch.exp(pi_logprobs)
        entropies = -torch.sum(probs * pi_logprobs * loss_mask, dim=-1)
        entropy = entropies.mean()
        entropy_loss = self.config.entropy_loss_coef * entropy

        # Initialize metrics with common values
        metrics = {
            "train/pg_loss": pg_loss.detach().item(),
            "train/entropy_loss": entropy_loss.detach().item(),
            "policy/entropy": entropy.detach().item(),
            "policy/approxkl": approxkl.detach().item(),
            "policy/clipfrac": clipfrac.detach().item(),
        }

        # Compute KL divergence if coefficient is positive
        if self.config.kl_loss_coef > 0:
            # Compute the KL divergence between the model and the reference model
            per_token_kl = (
                torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
            )

            # # Clamp for stability
            # per_token_log_ratio = torch.clamp(ref_logprobs - pi_logprobs, min=-20, max=20)
            # per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1.0

            kl = self.masked_mean(per_token_kl, loss_mask, dim=1).mean()
            kl_loss = self.config.kl_loss_coef * kl

            loss = pg_loss + kl_loss + entropy_loss
            metrics.update(
                {
                    "train/kl_loss": kl_loss.detach().item(),
                    "objective/kl": kl.detach().item(),
                }
            )
        else:
            loss = pg_loss + entropy_loss

        return loss, metrics

    def train_step(self, train_dataloader: DataLoader):
        """Performs the policy update phase."""

        self._prepare_for_training()

        for _ in range(self.config.num_updates):
            for i, micro_batch in enumerate(train_dataloader):
                input_ids = micro_batch.states
                attention_mask = (input_ids != self.tokenizer.pad_token_id).bool()
                pi_logits = self.policy_engine.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                loss, metrics = self.compute_loss(pi_logits, micro_batch)
                self.policy_engine.backward(loss)
                self.policy_engine.step()

                del input_ids, attention_mask, pi_logits
                torch.cuda.empty_cache()

                for k, v in metrics.items():
                    self.logger.log_scalar(k, v)

                if self.policy_engine.is_gradient_accumulation_boundary():
                    self.policy_update_count += 1
                    self.logger.log_scalar(
                        "train/policy_update", self.policy_update_count
                    )
                    self.logger.log_scalar(
                        "train/learning_rate",
                        self.policy_engine.get_lr()[0],
                    )

    def evaluate_step(self):
        pass

    def _eval_collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader during evaluation"""
        pad_token_id = self.tokenizer.pad_token_id

        # Extract input_ids and attention_mask as lists of tensors
        input_ids_list = [sample["input_ids"] for sample in batch]
        attention_mask_list = [sample["attention_mask"] for sample in batch]

        # Dynamically pad to the longest sequence in the batch
        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
            padding_side="left",
        )
        attention_mask = pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )

        # Collect other fields
        questions = [sample["question"] for sample in batch]
        ground_truths = [sample["ground_truth"] for sample in batch]
        task_types = [sample["task_type"] for sample in batch]

        return {
            "input_ids": input_ids.to(
                self.device
            ),  # Shape: [batch_size, max_seq_len_in_batch]
            "attention_mask": attention_mask.to(self.device),
            "questions": questions,
            "ground_truths": ground_truths,
            "task_types": task_types,
        }

    def _train_collate_fn(self, batch: List[TransitionData]) -> TransitionData:
        """Collate function for DataLoader during training"""
        pad_token_id = self.tokenizer.pad_token_id
        torch_dtype = self.torch_dtype

        # Pad states and actions (long tensors)
        batch_states = pad_sequence(
            [item.states for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ).long()
        batch_actions = pad_sequence(
            [item.actions for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ).long()

        # Pad loss_mask (boolean tensor)
        batch_loss_mask = pad_sequence(
            [item.loss_mask for item in batch],
            batch_first=True,
            padding_value=False,
        ).bool()

        # Pad advantages, pi_logprobs, and ref_logprobs (float tensors)
        batch_advantages = (
            pad_sequence(
                [item.advantages for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )
        batch_pi_logprobs = (
            pad_sequence(
                [item.pi_logprobs for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )
        batch_ref_logprobs = (
            pad_sequence(
                [item.ref_logprobs for item in batch],
                batch_first=True,
                padding_value=0.0,
            )
            .float()
            .to(torch_dtype)
        )

        return TransitionData(
            states=batch_states.to(self.device),
            actions=batch_actions.to(self.device),
            loss_mask=batch_loss_mask.to(self.device),
            pi_logprobs=batch_pi_logprobs.to(self.device),
            ref_logprobs=batch_ref_logprobs.to(self.device),
            advantages=batch_advantages.to(self.device),
        )
