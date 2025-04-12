"""Implements MDP ENV for collect samples for RL"""

import functools
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from rl4llm.constants import LOGGER_NAME
from rl4llm.core.base_env import (
    BaseEnv,
    BaseRewardFunction,
    EnvState,
    EpisodeData,
)
from rl4llm.generation.explore_processor import ExploreLogitsProcessor

logger = logging.getLogger(LOGGER_NAME)


class LocalLLMEnv(BaseEnv):
    """
    Environment for generating training samples with LLM models from HuggingFace library.

    This environment handles the workflow of sampling prompts from a dataset,
    generating completions using a provided LLM, calculating rewards, and
    returning structured episode data.
    """

    def _generate_completions(
        self,
        llm: PreTrainedModel,
        sampling_params: Dict,
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Generates completions using the LLM for the current state.

        Args:
            llm: The pre-trained language model.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens, do_sample).
            state: The current EnvState containing input_ids and attention_mask.
            **kwargs: Additional custom arguments.

        Returns:
            A tuple containing:
            - completion_texts: List of decoded completion strings (up to, but not including, EOS).
            - completion_tokens: List of completion token tensors (unpadded, includes EOS if present).
            - completion_lengths: List of actual lengths for each completion (includes EOS, excludes PAD).
        """
        input_ids = state.input_ids.to(llm.device)
        attention_mask = state.attention_mask.to(llm.device)
        # Remove keys that conflict with inputs
        sampling_params = {
            k: v
            for k, v in sampling_params.items()
            if k not in ['input_ids', 'attention_mask', 'num_return_sequences']
        }

        output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **sampling_params,
        )

        start = state.input_ids.shape[1]
        texts, tokens_list, lengths = self._process_completions(
            output.sequences[:, start:]
        )

        return texts, tokens_list, lengths

    def _process_completions(
        self,
        sequences: torch.Tensor,
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Processes generated sequences to extract completions, decode text, and calculate lengths,
        stopping at the first EOS or PAD token.

        Args:
            sequences: Tensor of generated sequences (completion + padded) from the LLM.
                       Shape: (batch_size, sequence_length).

        Returns:
            A tuple containing:
            - completion_texts: List of decoded completion strings (up to, but not including, EOS).
            - completion_tokens: List of completion token tensors (unpadded, includes EOS if present).
            - completion_lengths: List of actual lengths for each completion (includes EOS, excludes PAD).
        """
        if sequences.ndim != 2:
            raise ValueError(
                f"Expected completion_sequences to be 2D, but got shape {sequences.shape}"
            )
        batch_size, seq_len = sequences.shape
        is_special = (sequences == self.eos_token_id) | (
            sequences == self.pad_token_id
        )
        found = is_special.any(dim=1)
        first_special = torch.argmax(is_special.int(), dim=1)
        default_length = torch.full(
            (batch_size,),
            seq_len,
            dtype=first_special.dtype,
            device=sequences.device,
        )
        actual_lengths = torch.where(found, first_special, default_length)
        first_special_ids = torch.gather(
            sequences, 1, first_special.unsqueeze(1)
        ).squeeze(1)
        actual_lengths = torch.where(
            found & (first_special_ids == self.eos_token_id),
            actual_lengths + 1,
            actual_lengths,
        )
        actual_lengths_cpu = actual_lengths.cpu()
        sequences_cpu = sequences.cpu()
        tokens_list = [
            sequences_cpu[i, : int(actual_lengths_cpu[i])]
            for i in range(batch_size)
        ]
        tokens_for_decoding = [
            (
                tokens[:-1]
                if (len(tokens) > 0 and tokens[-1].item() == self.eos_token_id)
                else tokens
            )
            for tokens in tokens_list
        ]
        texts = self.tokenizer.batch_decode(
            tokens_for_decoding,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return texts, tokens_list, actual_lengths_cpu.tolist()

    @torch.inference_mode()
    def rollout(
        self,
        llm: PreTrainedModel,
        sampling_params: Dict,
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[EpisodeData]:
        """
        Performs a rollout step: gets a batch, generates completions, calculates rewards,
        and returns structured episode data.

        Args:
            llm: The pre-trained language model to use for generation.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens).
            **kwargs: Additional custom arguments.

        Returns:
            A list of EpisodeData objects, one for each generated sample in the batch
            (batch_size * group_size samples). Returns an empty list if the dataset is exhausted.
        """
        if sampling_params.get('num_return_sequences', 1) > 1:
            raise ValueError(
                'Set group_size during initialization instead of using num_return_sequences.'
            )

        # Add more common generation arguments
        sampling_params.update(
            {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'use_cache': True,
                'output_scores': False,
                'output_logits': False,
                'return_dict_in_generate': True,
                'return_legacy_cache': False,
            }
        )
        state = self._reset()
        if state is None:
            logger.warning(
                f"Rank {self.rank}: Reset returned None; dataset exhausted."
            )
            return []
        texts, tokens_list, lengths = self._generate_completions(
            llm, sampling_params, state, **kwargs
        )

        rewards = self._calculate_rewards(texts, state.ground_truth)
        prompt_tokens = [
            state.input_ids[i][state.attention_mask[i] == 1].cpu()
            for i in range(len(texts))
        ]
        return [
            EpisodeData(
                prompt_text=state.prompt[i],
                prompt_tokens=prompt_tokens[i],
                prompt_length=len(prompt_tokens[i]),
                completion_text=texts[i],
                completion_tokens=tokens_list[i],
                completion_length=lengths[i],
                reward_dict={k: v[i] for k, v in rewards.items()},
                raw_data=state.raw_data[i],
            )
            for i in range(len(texts))
        ]


class ExploreLocalLLMEnv(LocalLLMEnv):
    """An extension of the standard LocalLLMEnv
    where we apply some custom logits processor to the generation process
    to encourage exploration."""

    def __init__(
        self,
        temperature: Union[List[float], torch.Tensor],
        explore_steps: int,
        explore_top_k: int,
        explore_skip_n: int,
        explore_decay_rate: float,
        replace_source_tokens: List[int],
        replace_target_tokens: List[int],
        replace_prevent_patterns: List[List[int]],
        replace_max_per_seq: int,
        replace_prob: float,
        **kwargs,
    ):

        super().__init__(**kwargs)

        assert len(temperature) >= 1

        self.temperature = temperature
        self.explore_steps = explore_steps
        self.explore_top_k = explore_top_k
        self.explore_skip_n = explore_skip_n
        self.explore_decay_rate = explore_decay_rate
        self.replace_source_tokens = replace_source_tokens
        self.replace_target_tokens = replace_target_tokens
        self.replace_prevent_patterns = replace_prevent_patterns
        self.replace_max_per_seq = replace_max_per_seq
        self.replace_prob = replace_prob

        self.accuracy_fn = None
        for fn in self.reward_functions:
            if fn.name == 'accuracy_reward':
                self.accuracy_fn = fn
                break

    def _generate_completions(
        self,
        llm: PreTrainedModel,
        sampling_params: Dict,
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Generates completions using the LLM for the current state.

        Args:
            llm: The pre-trained language model.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens, do_sample).
            state: The current EnvState containing input_ids and attention_mask.
            **kwargs: Additional custom arguments.

        Returns:
            A tuple containing:
            - completion_texts: List of decoded completion strings (up to, but not including, EOS).
            - completion_tokens: List of completion token tensors (unpadded, includes EOS if present).
            - completion_lengths: List of actual lengths for each completion (includes EOS, excludes PAD).
        """

        input_ids = state.input_ids.to(llm.device)
        attention_mask = state.attention_mask.to(llm.device)
        gen_args_copy = sampling_params.copy()
        gen_args_copy.pop('input_ids', None)
        gen_args_copy.pop('attention_mask', None)
        gen_args_copy.pop('num_return_sequences', None)
        gen_args_copy['return_dict_in_generate'] = True

        # add explore logits processor
        explore_prob = kwargs.get('explore_probability', 0.0)

        if explore_prob > 0 and (random.random() < explore_prob):
            correctness_callback = None
            # Checks for outcome correctness using the accuracy function,
            # if applied, will only apply token replacement to sequences with incorrect outcome
            if self.accuracy_fn:
                correctness_callback = functools.partial(
                    self.accuracy_fn.__call__,
                    ground_truths=(
                        state.ground_truth[0]
                        if self.batch_size == 1
                        else state.ground_truth
                    ),
                )

            explore_logits_processor = ExploreLogitsProcessor(
                initial_seq_len=input_ids.shape[1],
                tokenizer=self.tokenizer,
                group_size=self.group_size,
                temperature=self.temperature,
                explore_steps=self.explore_steps,
                explore_skip_n=self.explore_skip_n,
                explore_top_k=self.explore_top_k,
                explore_decay_rate=self.explore_decay_rate,
                replace_source_tokens=self.replace_source_tokens,
                replace_target_tokens=self.replace_target_tokens,
                replace_prevent_patterns=self.replace_prevent_patterns,
                replace_max_per_seq=self.replace_max_per_seq,
                replace_prob=self.replace_prob,
                correctness_callback=correctness_callback,
            )
            gen_args_copy['logits_processor'] = LogitsProcessorList(
                [explore_logits_processor]
            )

        output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_args_copy,
        )
        start = state.input_ids.shape[1]
        texts, tokens_list, lengths = self._process_completions(
            output.sequences[:, start:]
        )

        return texts, tokens_list, lengths
