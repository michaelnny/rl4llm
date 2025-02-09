import math
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)

from rl4llm.data import load_math_dataset
from rl4llm.graders import math_problem_grader


class GRPOTrainer:

    def __init__(
        self,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.AdamW,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_ds: Dataset,
        device: torch.device,
        dtype: torch.dtype,
        writer: SummaryWriter,
        seed: Optional[int] = 167,
    ):
        self.seed = seed

        set_seed(self.seed)

        self.device = device
        self.dtype = dtype
        self.policy_model = policy_model
        self.reference_model = deepcopy(policy_model)
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.writer = writer
        self.train_ds = train_ds

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = [self.tokenizer.eos_token, self.tokenizer.pad_token]

        self.episode_count = 0
        self.update_count = 0
        self.iteration_count = 0

    def _compute_action_logprobs(self, model, input_ids, attention_mask) -> torch.Tensor:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs

    @torch.no_grad()
    def generate_group_samples(self, item: Dict, decoding_args: Dict, group_size: int, system_prompt: str) -> List[Dict]:
        """Generate responses for given states, handling None states."""

        ground_truth = item["ground_truth"]

        if system_prompt:
            message = [{"role": "user", "content": item["question"]}]
        else:
            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": item["question"]}]

        # expand to have a batch dimension
        message = [message for _ in range(group_size)]

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

        generation_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'use_cache': True,
            'output_scores': True,
            'output_logits': True,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
            **decoding_args,
        }

        outputs = self.policy_model.generate(**generation_kwargs)
        full_sequences = outputs.sequences
        prompt_length = input_ids.size(1)
        completion_ids = full_sequences[:, prompt_length:]

        # Efficient trimming of sequences
        eos_mask = full_sequences == self.eos_token_id
        cut_positions = eos_mask.float().argmax(dim=1) + 1

        # prompt_tokens_count = (input_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        # completion_tokens_count = (completion_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards for group outcomes
        rewards = [math_problem_grader(completion, ground_truth) for completion in completion_texts]

        # Normalize rewards
        normalized_rewards = self.normalize_rewards(rewards)

        # TODO: this is
        attention_mask = (full_sequences != self.pad_token_id).bool()
        pi_logprobs = self._compute_action_logprobs(self.policy_model, full_sequences, attention_mask).cpu()
        ref_logprobs = self._compute_action_logprobs(self.reference_model, full_sequences, attention_mask).cpu()

        results = []

        for i, cut_position in enumerate(cut_positions):
            trimmed_sequence = full_sequences[i, :cut_position].cpu()
            trimmed_pi_logprobs = pi_logprobs[i, :cut_position].cpu()
            trimmed_ref_logprobs = ref_logprobs[i, :cut_position].cpu()

            # TODO: this is incorrect
            
            # Loss mask
            loss_mask = torch.zeros_like(trimmed_sequence, dtype=torch.bool)
            loss_mask[prompt_length:cut_position] = 1

            # Create dictionary with all information
            sample = {
                'state_ids': trimmed_sequence[:-1].tolist(),
                'action_ids': trimmed_sequence[1:].tolist(),
                'loss_mask': loss_mask[1:].tolist(),
                'reward': rewards[i],
                'advantages': loss_mask[1:] * normalized_rewards[i],
                'pi_logprobs': trimmed_pi_logprobs[1:].tolist(),
                'ref_logprobs': trimmed_ref_logprobs[1:].tolist(),
                'completion_text': completion_texts[i],
                'completion_length': loss_mask[1:].sum().item(),
            }

            assert (
                len(sample['state_ids'])
                == len(sample['action_ids'])
                == len(sample['advantages'])
                == len(sample['pi_logprobs'])
                == len(sample['ref_logprobs'])
                == len(sample['loss_mask'])
            )
            results.append(sample)
            self.episode_count += 1

            if self.writer:
                formatted_text = (
                    f"**Question**: {item["question"]}\n\n"
                    f"**Ground Truth**: {item["ground_truth"]}\n\n"
                    f"**Graded Reward**: {sample["reward"]}\n\n"
                    f"**Full Answer**:\n```json\n{sample["completion_text"]}\n```"
                )
                self.writer.add_text("sample", formatted_text, self.episode_count)
                self.writer.add_scalar("sample/reward", sample["reward"], self.episode_count)
                self.writer.add_scalar("sample/completion_length", sample['completion_length'], self.episode_count)

        return results

    @staticmethod
    def normalize_rewards(rewards: List[float]) -> List[float]:
        """
        Normalize rewards by subtracting the mean and dividing by the standard deviation.
        Args:
            rewards (list of float): List of rewards for the group.
        Returns:
            list of float: Normalized rewards.
        """
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)  # Add small value to avoid division by zero
        return normalized_rewards

    def generate_samples(
        self,
        decoding_args: Dict,
        system_prompt: str,
        group_size: int = 8,
        max_episodes: int = 1024,
    ) -> List[Dict]:
        """Generates samples using the inference engine."""
        assert group_size >= 4
        assert max_episodes >= 128

        self.reference_model.to(self.device)
        self.policy_model.eval()
        torch.cuda.empty_cache()
        collected_samples = []

        with tqdm(total=max_episodes, desc=f'Generating episodes', unit='episode') as pbar:
            data_iter = iter(self.train_ds)  # Create the iterator once outside the loop
            while len(collected_samples) < max_episodes:
                try:
                    item = next(data_iter)  # Fetch the next batch
                except StopIteration:
                    # Restart the iterator if all data is exhausted
                    self.train_ds = self.train_ds.shuffle(seed=None)
                    data_iter = iter(self.train_ds)
                    item = next(data_iter)

                assert "question" in item and "ground_truth" in item

                samples = self.generate_group_samples(item, decoding_args, group_size, system_prompt)

                collected_samples.extend(samples)
                pbar.update(len(samples))
        pbar.close()
        return collected_samples

    def train(
        self,
        samples: List[Dict],
        num_updates: int,
        batch_size: int,
        gradient_accumulate_steps: int,
        clip_eps: float = 0.2,
        kl_loss_coef: float = 0.1,
    ) -> None:

        def _collate_function(batch: List[Dict]) -> Dict:
            max_seq_len = max([len(item['state_ids']) for item in batch]) - 1
            batch_state_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            batch_action_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            batch_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)

            batch_advantages = torch.full((batch_size, max_seq_len), 0, dtype=self.dtype)
            batch_pi_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=self.dtype)
            batch_ref_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=self.dtype)
            batch_rewards = torch.full((batch_size,), 0, dtype=self.dtype)

            for i, item in enumerate(batch):
                seq_len = len(item['state_ids'])
                batch_state_ids[i, :seq_len] = torch.tensor(item['state_ids'], dtype=torch.long)
                batch_action_ids[i, :seq_len] = torch.tensor(item['action_ids'], dtype=torch.long)
                batch_advantages[i, :seq_len] = torch.tensor(item["advantages"], dtype=self.dtype)
                batch_pi_logprobs[i, :seq_len] = torch.tensor(item["pi_logprobs"], dtype=self.dtype)
                batch_ref_logprobs[i, :seq_len] = torch.tensor(item["ref_logprobs"], dtype=self.dtype)
                batch_loss_mask[i, :seq_len] = torch.tensor(item["loss_mask"], dtype=torch.bool)
                batch_rewards[i] = torch.tensor(item["reward"], dtype=self.dtype)

            return {
                "state_ids": batch_state_ids,
                "action_ids": batch_action_ids,
                "advantages": batch_advantages,
                "pi_logprobs": batch_pi_logprobs,
                "ref_logprobs": batch_ref_logprobs,
                "loss_mask": batch_loss_mask,
                "rewards": batch_rewards,
            }

        data_loader = DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=_collate_function,
            drop_last=False,
        )

        total_steps = math.ceil(num_updates * len(samples) / batch_size * gradient_accumulate_steps)

        pbar = tqdm(desc='Training steps', unit='batch', total=total_steps)
        # accumulated_iter_stats = defaultdict(list)
        accumulated_stats = defaultdict(list)

        self.reference_model.to("cpu")
        self.policy_model.train()

        mini_steps = 0
        for epoch in range(num_updates):
            for mini_batch in data_loader:
                state_ids = mini_batch['state_ids'].to(self.device)
                action_ids = mini_batch['action_ids'].to(self.device)
                attention_mask = (attention_mask != self.pad_token_id).bool()

                pi_logprobs = self._compute_action_logprobs(self.policy_model, state_ids, attention_mask, action_ids)

                behavior_logprobs = mini_batch["pi_logprobs"].to(self.device)
                advantages = mini_batch["advantages"].to(self.device)
                loss_mask = mini_batch["loss_mask"].to(self.device)
                ref_logprobs = mini_batch["ref_logprobs"].to(self.device)
                # Compute the KL divergence between the model and the reference model
                per_token_kl = torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1

                # PPO clipped surrogate PG loss
                ratio = torch.exp(pi_logprobs - behavior_logprobs)
                clipped_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)
                pg_losses = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())
                clipped = ratio.gt(1 + clip_eps) | ratio.lt(1 - clip_eps)
                pg_clipfrac = torch.as_tensor(clipped, dtype=ratio.dtype)
                approxkl = (pi_logprobs - behavior_logprobs).detach()

                pg_loss = pg_losses[loss_mask].mean()
                approxkl = approxkl[loss_mask].mean()
                pg_clipfrac = pg_clipfrac[loss_mask].mean()
                kl_penalties = kl_loss_coef * per_token_kl[loss_mask].mean()

                loss = -pg_loss + kl_penalties

                if gradient_accumulate_steps > 0:
                    loss /= gradient_accumulate_steps

                loss.backward()

                accumulated_stats['loss'].append(loss.detach().item())

                mini_steps += 1

                if mini_steps % gradient_accumulate_steps == 0:
                    pbar.update(1)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.update_count += 1

        elapsed_time = pbar.format_dict.get('elapsed', 0)
        pbar.close()

        self.iteration_count += 1
        stats = {
            'elapsed/time': round(elapsed_time, 4),
            'elapsed/step_time': round(elapsed_time / max(pbar.total, 1) if pbar.total else 0, 4),
            'elapsed/updates': self.update_count,
            'elapsed/episodes': self.episode_count,
            "reward": np.mean([d['reward'] for d in samples]),
            "completion_length": np.mean([d['completion_length'] for d in samples]),
            "loss": np.mean(accumulated_stats['loss']),
        }
        print(stats)

        if self.writer:
            for name, value in stats.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"leaner/{name}", value, self.iteration_count)

    def save_checkpoint(self, save_dir: str):
        self.policy_model.save_pretrained(save_dir)


class CosineDecayWithWarmupLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Follows the GPT-3 paper"""

    def __init__(self, optimizer, init_lr, max_lr, min_lr, warmup_steps, max_decay_steps, last_epoch=-1) -> None:
        """
        Args:
            init_lr: initial learning rate
            max_lr: maximum learning rate at the end of the linear warm up phase
            min_lr: minimum learning rate at the end of the cosine annealing phase phase
            warmup_steps: number of steps to linear warm the learning rate from init_lr to max_lr
            max_decay_steps: number of steps to apply cosine annealing to the learning rate from max_lr to min_lr
        """

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.max_decay_steps = max_decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm-up phase
            return [self.init_lr + (self.max_lr - self.init_lr) * self.last_epoch / self.warmup_steps] * len(
                self.optimizer.param_groups
            )
        elif self.last_epoch >= self.warmup_steps and self.last_epoch < self.max_decay_steps:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.max_decay_steps - self.warmup_steps)
            return [self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))] * len(
                self.optimizer.param_groups
            )
        else:
            return [self.min_lr] * len(self.optimizer.param_groups)


def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit = True
    optim_type = "AdamW_8bit"  # "AdamW"
    learning_rate = 1e-6
    weight_decay = 0.002
    eps = 1e-8
    betas = (0.9, 0.999)
    kl_loss_coef = 0.04
    num_epochs = 1000
    num_updates = 1
    batch_size = 4
    gradient_accumulation_steps = 16
    group_size = 8
    rollout_size = 128
    decoding_args = {"do_sample": True, "temperature": 0.6, "max_new_tokens": 800}
    system_prompt = """
Think first about the reasoning process in your mind and then provides the user with the answer.
"""
    checkpoint_interval = 20
    job_dir = "./runs/grpo_qwen_3b"
    tb_log_dir = f"{job_dir}/tb_logs"
    checkpoint_dir = f"{job_dir}/checkpoints"

    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir, exist_ok=True)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch_dtype = torch.bfloat16
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_args = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": torch_dtype,
        "use_cache": False,
        "attn_implementation": "flash_attention_2",
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if load_in_4bit:
        model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    policy_model = AutoModelForCausalLM.from_pretrained(**model_args)

    decay_params = []
    nodecay_params = []
    for name, param in policy_model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ["bias", "layer_norm.weight", "layernorm.weight", "norm.weight"]):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

    optim_groups = [
        {'params': nodecay_params, 'lr': learning_rate, 'weight_decay': 0.0, 'name': 'nodecay'},
        {'params': decay_params, 'lr': learning_rate, 'weight_decay': weight_decay, 'name': 'decay'},
    ]

    optim_kwargs = {'lr': learning_rate, 'eps': eps, 'betas': betas}

    if optim_type == "AdamW_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW(optim_groups, **optim_kwargs)
    else:
        optimizer = torch.optim.AdamW(optim_groups, **optim_kwargs)

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer,
        init_lr=0.1 * learning_rate,
        max_lr=learning_rate,
        min_lr=0.1 * learning_rate,
        warmup_steps=100,
        max_decay_steps=10000,
    )

    train_ds, _ = load_math_dataset()
    trainer = GRPOTrainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ds=train_ds,
        device=device,
        dtype=torch_dtype,
        writer=SummaryWriter(tb_log_dir),
    )

    for epoch in range(1, num_epochs + 1):
        samples = trainer.generate_samples(
            decoding_args=decoding_args, system_prompt=system_prompt, group_size=group_size, max_episodes=rollout_size
        )

        trainer.train(
            samples=samples,
            num_updates=num_updates,
            batch_size=batch_size,
            gradient_accumulate_steps=gradient_accumulation_steps,
            kl_loss_coef=kl_loss_coef,
        )

        if epoch > 1 and epoch & checkpoint_interval == 0:
            trainer.save_checkpoint(f"{checkpoint_dir}/epoch_{epoch}")


if __name__ == "__main__":
    main()
