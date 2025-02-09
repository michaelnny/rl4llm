from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset
from tqdm import tqdm
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

# from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig

from rl4llm.graders import math_problem_grader
from rl4llm.types import DecodingConfig, EnvAction, EnvState, Episode, TokenUsage
from rl4llm.utils import TrainingTracker


class GRPOTrainer:

    def __init__(
        self,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_ds: Dataset,
        device: torch.device,
        tracker: TrainingTracker,
        seed: Optional[int] = 167,
    ):
        self.seed = seed

        set_seed(self.seed)

        self.device = device
        self.policy_model = policy_model
        self.reference_model = deepcopy(policy_model)
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        self.tokenizer = tokenizer
        self.tracker = tracker
        self.train_ds = train_ds

        sampler = DistributedSampler(self.train_ds, shuffle=True, seed=self.seed)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=1,  # always use one sample as we'll generate a group of outcomes
            sampler=sampler,
            # pin_memory=self.device.type == 'cuda',
            # collate_fn=self._train_collate_fn,
            drop_last=False,
        )

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = [self.tokenizer.eos_token, self.tokenizer.pad_token]

    @torch.inference_mode()
    def act(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, decoding: DecodingConfig
    ) -> Tuple[torch.Tensor, List[str], List[TokenUsage]]:
        """Generate responses for given states, handling None states."""

        # # batch_messages = [[{'role': t.role, 'content': t.content} for t in states] for states in valid_states]
        # message_prompt = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)

        # inputs = self.tokenizer(
        #     message_prompt,
        #     return_tensors='pt',
        #     truncation=True,
        #     padding=True,
        #     padding_side='left',
        #     max_length=self.tokenizer.model_max_length,
        # ).to(self.device)

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
            'max_new_tokens': decoding.max_new_tokens,
            'temperature': decoding.temperature if decoding.temperature is not None else 1.0,
            'top_p': decoding.top_p if decoding.top_p is not None else None,
            'top_k': decoding.top_k if decoding.top_k is not None else None,
            'do_sample': decoding.do_sample,
        }

        outputs = self.policy_model.generate(**generation_kwargs)

        sequences = outputs.sequences

        prompt_length = input_ids.size(1)
        batch_completion_ids = sequences[:, prompt_length:]
        prompt_tokens_count = (input_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        completion_tokens_count = (batch_completion_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        completion_texts = self.tokenizer.batch_decode(batch_completion_ids, skip_special_tokens=True)

        usages = []
        for i in range(sequences.size(0)):
            usages.append(
                TokenUsage(
                    prompt_tokens=prompt_tokens_count[i],
                    completion_tokens=completion_tokens_count[i],
                    total_tokens=prompt_tokens_count[i] + completion_tokens_count[i],
                ),
            )

        return sequences, completion_texts, usages

    def generate_samples(
        self,
        decoding: DecodingConfig,
        group_size: int = 8,
        max_episodes: int = 1024,
    ) -> Tuple[List[Episode], Dict]:
        """Generates samples using the inference engine."""
        assert group_size >= 4
        assert max_episodes >= 256
        torch.cuda.empty_cache()
        collected_episodes = []

        # states = vector_env.reset()

        with tqdm(total=max_episodes, desc=f'Generating episodes', unit='episode') as pbar:
            while len(collected_episodes) < max_episodes:
                sample = iter(self.train_loader)
                assert "question" in sample and "ground_truth" in sample

                ground_truth = sample["ground_truth"]

                message = [{"role": "user", "content": sample["question"]}]

                # expand to have a batch dimension
                message = [message for _ in range(group_size)]

                # batch_messages = [[{'role': t.role, 'content': t.content} for t in states] for states in valid_states]
                message_prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

                inputs = self.tokenizer(
                    message_prompt,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    padding_side='left',
                    max_length=self.tokenizer.model_max_length,
                ).to(self.device)

                sequences, completion_texts, usages = self.act(inputs.input_ids, inputs.attention_mask, decoding)

                rewards = [math_problem_grader(completion, ground_truth) for completion in completion_texts]

                



                # prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

                # inputs = self.tokenizer(
                #     prompt,
                #     return_tensors='pt',
                #     truncation=True,
                #     padding=True,
                #     padding_side='left',
                #     max_length=self.tokenizer.model_max_length,
                # ).to(self.device)

        #         active_env_indices = [i for i, state in enumerate(states) if state is not None]
        #         active_states = [states[i] for i in active_env_indices]

        #         actions = [None] * vector_env.num_envs  # Generate actions only for active environments
        #         if active_states:
        #             active_actions = self.act(active_states, decoding)
        #             for i, action in zip(active_env_indices, active_actions):
        #                 actions[i] = action

        #         new_states = vector_env.step(actions)

        #         for i in range(vector_env.num_envs):  # Process completed episodes and reset
        #             if states[i] is not None and vector_env.is_done(i):
        #                 episode = vector_env.get_episode(i)
        #                 processed_episodes = self._process_batch_episodes([episode], stats_tracker, for_evaluator)
        #                 if processed_episodes:
        #                     collected_episodes.extend(processed_episodes)
        #                     if len(collected_episodes) % 10 == 0:
        #                         pbar.update(10)
        #                 new_states[i] = vector_env.reset_one(i)

        #         states = new_states
        #         if self.tracker and len(collected_episodes) and len(collected_episodes) % 100 == 0:
        #             Thread(target=self.tracker.flush).start()

        # iter_stats = self._compute_iteration_stats(
        #     stats_tracker, collected_episodes, vector_env.max_reward, pbar.format_dict.get('elapsed', 0)
        # )
        # self._log_iteration_stats(iter_stats, for_evaluator)
        # return collected_episodes, iter_stats
