import math
import os
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer, set_seed

from rl4llm.data import load_math_dataset, load_gsm_dataset
from rl4llm.graders import math_problem_grader


def create_scheduler(optimizer, max_lr, total_steps, warmup_fraction=0.1, initial_lr_fraction=0.1, final_lr_fraction=0.01):
    """
    Creates a OneCycleLR scheduler with warmup and cosine decay.

    Args:
        optimizer: The optimizer to use
        max_lr: Maximum learning rate after warmup
        total_steps: Total number of training steps
        warmup_fraction: Fraction of total steps used for warmup (default: 0.3)
        initial_lr_fraction: Fraction of max_lr to use as the initial learning rate (default: 0.1)
        final_lr_fraction: Fraction of max_lr to use as the final learning rate (default: 0.01)
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=warmup_fraction,
        div_factor=1 / initial_lr_fraction,
        final_div_factor=1 / (initial_lr_fraction * final_lr_fraction),
        anneal_strategy='cos',
    )


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
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

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

        self.generation_mode = False

        self.episode_count = 0
        self.update_count = 0
        self.iteration_count = 0

    @contextmanager
    def generation_context(self):
        """Context manager for handling model and optimizer states during generation"""
        try:
            self._prepare_for_generation()
            yield
        finally:
            self._prepare_for_training()

    def optimizer_to(self, device: str):
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
        self.optimizer_to("cpu")
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
        self.optimizer_to(self.policy_model.device)

        self.policy_model = self.policy_model.train()

        torch.cuda.empty_cache()
        self.generation_mode = False

    def _compute_action_logprobs(
        self, model: PreTrainedModel, input_ids: torch.LongTensor, actions: torch.LongTensor
    ) -> torch.Tensor:

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
    def generate_group_samples(
        self, question: str, ground_truth: str, decoding_args: Dict, group_size: int, system_prompt: str
    ) -> List[Dict]:
        """Generate responses for given states, handling None states."""
        if not system_prompt:
            message = [{"role": "user", "content": question.strip()}]
        else:
            message = [{"role": "system", "content": system_prompt.strip()}, {"role": "user", "content": question.strip()}]

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
        completion_tokens_count = (completion_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards for group outcomes
        rewards = [math_problem_grader(completion, ground_truth) for completion in completion_texts]

        # Normalize rewards
        normalized_rewards = self.normalize_rewards(rewards)

        states = full_sequences[:, :-1]
        actions = full_sequences[:, 1:]
        # attention_mask = (states != self.pad_token_id).bool()

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

            assert loss_mask[i, ...].sum() == completion_tokens_count[i]
            assert loss_mask[i, :cut_position].sum() == completion_tokens_count[i]

            sample = {
                'states': states[i, :cut_position].cpu().tolist(),
                'actions': actions[i, :cut_position].cpu().tolist(),
                'loss_mask': loss_mask[i, :cut_position].cpu().tolist(),
                'reward': rewards[i],
                'advantages': (
                    loss_mask[i, :cut_position].cpu() * normalized_rewards[i]
                ).tolist(),  # this is essentially monte carlo return with no discount
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

        if self.writer:
            sampled_item = random.choice(results)
            formatted_text = (
                f"**Question**: {question}\n\n"
                f"**Ground Truth**: {ground_truth}\n\n"
                f"**Graded Reward**: {sampled_item['reward']}\n\n"
                f"**Full Answer**:\n```json\n{sampled_item['completion_text']}\n```"
            )
            self.writer.add_text("sample", formatted_text, self.episode_count)

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

        with self.generation_context():
            assert not self.policy_model.training
            assert not self.reference_model.training
            collected_samples = []
            with tqdm(total=max_episodes, desc=f'Generating episodes', unit='episode') as pbar:
                # Create the iterator once outside the loop
                data_iter = iter(self.train_ds)
                while len(collected_samples) < max_episodes:
                    try:
                        item = next(data_iter)  # Fetch the next batch
                    except StopIteration:
                        # Restart the iterator if all data is exhausted
                        self.train_ds = self.train_ds.shuffle(seed=None)
                        data_iter = iter(self.train_ds)
                        item = next(data_iter)

                    assert "question" in item and "ground_truth" in item

                    samples = self.generate_group_samples(
                        item['question'], item['ground_truth'], decoding_args, group_size, system_prompt
                    )

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
        kl_loss_coef: float = 0.02,
    ) -> None:

        def _collate_function(batch: List[Dict]) -> Dict:
            max_seq_len = max([len(item['states']) for item in batch])
            batch_state_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            batch_action_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            batch_loss_mask = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)

            batch_advantages = torch.full((batch_size, max_seq_len), 0, dtype=self.dtype)
            batch_pi_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=self.dtype)
            batch_ref_logprobs = torch.full((batch_size, max_seq_len), 0, dtype=self.dtype)
            batch_rewards = torch.full((batch_size,), 0, dtype=self.dtype)

            for i, item in enumerate(batch):
                seq_len = len(item['states'])
                batch_state_ids[i, :seq_len] = torch.tensor(item['states'], dtype=torch.long)
                batch_action_ids[i, :seq_len] = torch.tensor(item['actions'], dtype=torch.long)
                batch_advantages[i, :seq_len] = torch.tensor(item["advantages"], dtype=self.dtype)
                batch_pi_logprobs[i, :seq_len] = torch.tensor(item["pi_logprobs"], dtype=self.dtype)
                batch_ref_logprobs[i, :seq_len] = torch.tensor(item["ref_logprobs"], dtype=self.dtype)
                batch_loss_mask[i, :seq_len] = torch.tensor(item["loss_mask"], dtype=torch.bool)
                batch_rewards[i] = torch.tensor(item["reward"], dtype=self.dtype)

            return {
                "states": batch_state_ids,
                "actions": batch_action_ids,
                "advantages": batch_advantages,
                "pi_logprobs": batch_pi_logprobs,
                "ref_logprobs": batch_ref_logprobs,
                "loss_mask": batch_loss_mask,
                "rewards": batch_rewards,
            }

        random.shuffle(samples)

        data_loader = DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            collate_fn=_collate_function,
            drop_last=False,
        )

        total_steps = math.ceil(num_updates * len(samples) / (batch_size * gradient_accumulate_steps))

        pbar = tqdm(desc='Training steps', unit='batch', total=total_steps)
        # accumulated_iter_stats = defaultdict(list)
        accumulated_stats = defaultdict(list)

        self.optimizer.zero_grad()

        assert self.policy_model.training

        mini_steps = 0
        for epoch in range(num_updates):
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
                clipped_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)
                pg_losses = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

                pg_loss = pg_losses[loss_mask].mean()
                kl_penalties = kl_loss_coef * per_token_kl[loss_mask].mean()
                loss = -pg_loss + kl_penalties

                if gradient_accumulate_steps > 0:
                    loss /= gradient_accumulate_steps

                loss.backward()

                accumulated_stats['loss/total'].append(loss.detach().item())
                accumulated_stats['loss/policy'].append(pg_loss.detach().item())
                accumulated_stats['loss/kl_penalty'].append(kl_penalties.detach().item())
                accumulated_stats['others/kl'].append(per_token_kl[loss_mask].detach().sum(-1).mean().item())

                mini_steps += 1

                if mini_steps % gradient_accumulate_steps == 0:
                    pbar.update(1)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.update_count += 1
                    mini_steps = 0

        # if mini_steps > 0:
        #     self.optimizer.step()
        #     self.scheduler.step()
        #     self.optimizer.zero_grad()
        #     self.update_count += 1

        elapsed_time = pbar.format_dict.get('elapsed', 0)
        pbar.close()

        self.iteration_count += 1
        stats = {
            'elapsed/time': round(elapsed_time, 4),
            'elapsed/updates': self.update_count,
            'elapsed/episodes': self.episode_count,
            "objective/reward": np.mean([d['reward'] for d in samples]).item(),
            "objective/reward_std": np.std([d['reward'] for d in samples]).item(),
            "objective/completion_length": np.mean([d['completion_length'] for d in samples]).item(),
            "others/learning_rate": self.optimizer.param_groups[0]['lr'],
        }

        for k in accumulated_stats:
            stats[k] = np.mean(accumulated_stats[k]).item()

        if self.writer:
            for name, value in stats.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{name}", value, self.iteration_count)

    def save_checkpoint(self, save_dir: str):
        self.policy_model.save_pretrained(save_dir)


def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    load_in_4bit = False
    optim_type = "AdamW8bit"  # "AdamW"
    learning_rate = 1e-6
    weight_decay = 0.002
    eps = 1e-8
    betas = (0.9, 0.999)
    lr_decay_steps = 10000
    kl_loss_coef = 0.05
    num_epochs = 1000
    num_updates = 1
    batch_size = 4
    gradient_accumulation_steps = 16
    group_size = 8
    rollout_size = 1024
    decoding_args = {"do_sample": True, "temperature": 0.7, "max_new_tokens": 1024}
    system_prompt = """
Think first about the reasoning process in your mind and then provides the user with the answer.
"""
    checkpoint_interval = 20
    job_dir = "./runs/grpo_qwen_1.5b"
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

    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    if optim_type == "AdamW8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(optim_groups, **optim_kwargs)
    else:
        optimizer = torch.optim.AdamW(optim_groups, **optim_kwargs)

    scheduler = create_scheduler(
        optimizer,
        max_lr=learning_rate,
        total_steps=lr_decay_steps,
        warmup_fraction=100 / lr_decay_steps,
        initial_lr_fraction=0.1,
        final_lr_fraction=0.01,
    )

    train_ds, _ = load_gsm_dataset()  # load_math_dataset()
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

    with torch.autograd.set_detect_anomaly(True):
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
