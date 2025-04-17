"""Implements MDP ENV for collect samples for RL using vLLM engine

IMPORTANT:
This code is not used/supported anymore, as we have moved on to focusing on SGLang inference server.
But kept here for reference in case in the future we need to revisit it.
"""

import functools
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import vllm
from transformers import PreTrainedTokenizer

from rl4llm.core.base_env import (
    BaseEnv,
    BaseRewardFunction,
    EnvState,
    EpisodeData,
)

logger = logging.getLogger(__name__)


class VLLMEnv(BaseEnv):
    """
    Environment for generating training samples using vLLM engine.
    """

    def _generate_completions(
        self,
        llm: vllm.LLM,
        sampling_params: vllm.SamplingParams,
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Generates completions using the LLM for the current state.

        Args:
            llm: The vLLM engine.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens, do_sample).
            state: The current EnvState containing input_ids and attention_mask.
            **kwargs: Additional custom arguments.

        Returns:
            A tuple containing:
            - completion_texts: List of decoded completion strings (up to, but not including, EOS).
            - completion_tokens: List of completion token tensors (unpadded, includes EOS if present).
            - completion_lengths: List of actual lengths for each completion (includes EOS, excludes PAD).
        """
        output = llm.generate(
            prompts=state.prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Unpack completions
        completion_outputs = [item.outputs[0] for item in output]
        completion_ids = [
            torch.tensor(item.token_ids, dtype=torch.long)
            for item in completion_outputs
        ]
        completion_texts = [item.text for item in completion_outputs]
        actual_lengths = [len(item) for item in completion_ids]

        return completion_texts, completion_ids, actual_lengths

    @torch.inference_mode()
    def rollout(
        self,
        llm: vllm.LLM,
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
        if sampling_params.get('n', 1) > 1:
            raise ValueError(
                'Set group_size during initialization instead of using n.'
            )

        state = self._reset()
        if state is None:
            logger.warning(
                f"Rank {self.rank}: Reset returned None; dataset exhausted."
            )
            return []
        if not isinstance(sampling_params, vllm.SamplingParams):
            sampling_params = vllm.SamplingParams(**sampling_params)

        texts, tokens_list, lengths = self._generate_completions(
            llm, sampling_params, state, **kwargs
        )
        # post-processing completions
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


class vLLMExplorationLogitsProcessor:
    """
    Processes logits for vLLM engine generation, applying exploration sampling,
    and conditional token replacement based on sequence indices.

    IMPORTANT: vLLM calls the logits processor on a single sequence level, not batch level.
    So we can't reuse the same code from the HF logits processor.
    """

    def __init__(
        self,
        initial_seq_len: int,
        tokenizer: PreTrainedTokenizer,
        explore_steps: int = 0,
        explore_skip_n: int = 0,
        explore_top_k: int = 20,
        explore_decay_rate: float = 0.9,
        replace_source_tokens: Optional[List[int]] = None,
        replace_target_tokens: Optional[List[int]] = None,
        replace_prevent_patterns: Optional[List[List[int]]] = None,
        replace_prob: float = 0.0,
        replace_max_per_seq: int = 0,
        replace_boost_value: float = 100.0,
        replace_check_top_n: int = 3,
        correctness_callback: Optional[
            Callable[[List[str]], List[float]]
        ] = None,
    ):
        self.initial_seq_len: int = initial_seq_len
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.explore_steps: int = explore_steps
        self.explore_skip_n: int = explore_skip_n
        self.explore_top_k: int = explore_top_k
        self.explore_decay_rate: float = explore_decay_rate
        self.replace_source_tokens: List[int] = replace_source_tokens or []
        self.replace_target_tokens: List[int] = replace_target_tokens or []
        self.replace_prevent_patterns: List[List[int]] = (
            replace_prevent_patterns or []
        )
        self.replace_prob: float = replace_prob
        self.replace_max_per_seq: int = (
            replace_max_per_seq
            if self.replace_source_tokens
            and self.replace_target_tokens
            and replace_max_per_seq > 0
            else 0
        )
        self.replace_boost_value: float = replace_boost_value
        self.replace_check_top_n = replace_check_top_n
        self.correctness_callback: Optional[
            Callable[[List[str]], List[float]]
        ] = correctness_callback

        self.replacement_count = 0
        self.step_t = 0

    @torch.inference_mode()
    def __call__(
        self, past_token_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """Takes in past token ids and the logits for next token for a single sequence from the current batch."""

        generated_len = len(past_token_ids)

        # Handle exploration mode
        explore_start = (
            self.explore_steps > 0
            and (generated_len - self.explore_skip_n) < self.explore_steps
        )
        if self.explore_skip_n and generated_len < self.explore_skip_n:
            explore_start = False

        if explore_start and self.explore_top_k > 1:
            # print(f"EXPLORING random start for sequence {seq_idx}...")
            effective_steps = max(0, generated_len - self.explore_skip_n)
            current_explore_top_k = max(
                10,
                int(
                    self.explore_top_k
                    * (self.explore_decay_rate**effective_steps)
                ),
            )
            explore_k = min(10, current_explore_top_k)
            exp_top_k_values, exp_top_k_indices = torch.topk(
                logits, k=explore_k
            )
            logits.fill_(1e-8)
            logits.scatter_(
                0, exp_top_k_indices, torch.ones_like(exp_top_k_values) * 100.0
            )
            return logits

        # Check if next token is likely one of the  source token, like 'EOS' or '</think>'
        is_next_special = False
        _, top_k_indices = torch.topk(
            logits, k=self.replace_check_top_n, dim=-1
        )
        top_k_indices = top_k_indices.flatten().tolist()

        if any(
            tok in top_k_indices for tok in self.replace_source_tokens
        ) and all(
            tok not in past_token_ids[-20:]
            for tok in self.replace_target_tokens
        ):
            is_next_special = True

        # Handle token replacement
        should_replace = (
            is_next_special
            and self.replace_source_tokens
            and self.replace_target_tokens
            and self.replace_prob > 0
            and self.replace_max_per_seq > 0
            and self.replacement_count < self.replace_max_per_seq
            and generated_len > 50
        )

        if should_replace:
            generated_ids = past_token_ids  # past_token_ids[prompt_len:]
            if self._check_patterns(generated_ids):
                is_incorrect = True
                if self.correctness_callback is not None:
                    completion_text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    score = self.correctness_callback([completion_text])
                    is_incorrect = score[0] < 1.0

                if is_incorrect and random.random() < self.replace_prob:
                    # print(f"EXPLORING replace token for sequence {seq_idx}...")
                    if self.replacement_count < self.replace_max_per_seq:
                        self.replacement_count += 1
                        logits.fill_(1e-8)
                        logits[self.replace_target_tokens] = 100.0
                        return logits

        return logits

    def _check_patterns(self, token_ids: list[int]) -> bool:
        if not self.replace_prevent_patterns:
            return True
        for pattern in self.replace_prevent_patterns:
            pattern_len = len(pattern)
            for i in range(len(token_ids) - pattern_len + 1):
                if token_ids[i : i + pattern_len] == pattern:
                    return False
        return True


class ExploreVLLMEnv(VLLMEnv):
    """An extension of the standard VLLMEnv
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
        llm: vllm.LLM,
        sampling_params: vllm.SamplingParams,
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Generates completions using the LLM for the current state.

        Args:
            llm: The vLLM engine.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens, do_sample).
            state: The current EnvState containing input_ids and attention_mask.
            **kwargs: Additional custom arguments.

        Returns:
            A tuple containing:
            - completion_texts: List of decoded completion strings (up to, but not including, EOS).
            - completion_tokens: List of completion token tensors (unpadded, includes EOS if present).
            - completion_lengths: List of actual lengths for each completion (includes EOS, excludes PAD).
        """

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

            explore_logits_processor = vLLMExplorationLogitsProcessor(
                initial_seq_len=state.input_ids.shape[1],
                tokenizer=self.tokenizer,
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
            sampling_params.logits_processors = [explore_logits_processor]

        output = llm.generate(
            prompts=state.prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Unpack completions
        completion_outputs = [item.outputs[0] for item in output]
        completion_ids = [
            torch.tensor(item.token_ids, dtype=torch.long)
            for item in completion_outputs
        ]
        completion_texts = [item.text for item in completion_outputs]
        actual_lengths = [len(item) for item in completion_ids]

        return completion_texts, completion_ids, actual_lengths
