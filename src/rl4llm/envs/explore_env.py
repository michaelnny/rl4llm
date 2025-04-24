import functools
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LogitsProcessorList, PreTrainedModel

from rl4llm.core.base_env import BaseRewardFunction
from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.envs.llm_env import LocalLLMEnv
from rl4llm.envs.sgl_env import EnvState, InferenceEnv
from rl4llm.generation.hf_explore_processor import HfExploreLogitsProcessor
from rl4llm.generation.sgl_explore_procesor import SglExploreLogitProcessor

# from rl4llm.patches.sgl_patch_custom_sampler import apply_patch_explore_sampler


logger = logging.getLogger(__name__)


# --- Using SGLang inference client ---


class ExploreInferenceEnv(InferenceEnv):
    """An extension of the standard InferenceEnv
    where we apply some custom logits processor to the generation process
    to encourage exploration."""

    def __init__(
        self,
        group_temperature: torch.Tensor,
        group_top_p: torch.Tensor,
        explore_steps: int,
        explore_top_k: int,
        explore_skip_n: int,
        explore_decay: float,
        replace_source_tokens: Optional[List[int]] = None,
        replace_target_tokens: Optional[List[int]] = None,
        replace_check_top_k: int = 1,
        replace_max_count: int = 3,
        replace_prob: float = 0.5,
        **kwargs,
    ):

        super().__init__(**kwargs)
        if not isinstance(group_temperature, torch.Tensor):
            raise ValueError('group_temperature must be a tensor')
        if not isinstance(group_top_p, torch.Tensor):
            raise ValueError('group_top_p must be a tensor')
        if any(t < 0 for t in group_temperature):
            raise ValueError('group_temperature values cannot be negative')
        if any(p < 0 for p in group_top_p):
            raise ValueError('group_top_p values cannot be negative')
        assert len(group_top_p) == len(group_temperature) >= 1
        if not isinstance(replace_prob, float) or not (
            0.0 <= replace_prob < 1.0
        ):
            raise ValueError('replace_prob must be a float between (0.0, 1.0).')

        self.group_temperature = group_temperature
        self.group_top_p = group_top_p
        self.explore_steps = explore_steps
        self.explore_top_k = explore_top_k
        self.explore_skip_n = explore_skip_n
        self.explore_decay = explore_decay
        self.replace_source_tokens = replace_source_tokens
        self.replace_target_tokens = replace_target_tokens
        self.replace_check_top_k = replace_check_top_k
        self.replace_max_count = replace_max_count
        self.replace_prob = replace_prob

    def _prepare_logits_processor(self, explore_prob: float) -> Optional[str]:
        """Creates the explore logits processor string if conditions are met."""
        # This logic is self-contained setup, so keeping it separate is reasonable.
        if explore_prob > 0 and random.random() < explore_prob:
            explore_logit_processor = SglExploreLogitProcessor(
                explore_steps=self.explore_steps,
                explore_top_k=self.explore_top_k,
                explore_skip_n=self.explore_skip_n,
                explore_decay=self.explore_decay,
                replace_source_tokens=self.replace_source_tokens,
                replace_target_tokens=self.replace_target_tokens,
                replace_check_top_k=self.replace_check_top_k,
                replace_max_count=self.replace_max_count,
            )
            return explore_logit_processor.to_str()
        return None

    def _generate_completions(
        self,
        llm: InferenceClient,
        sampling_params: Dict[str, Any],
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Generates completions using the LLM with probabilistic retry logic for incorrect items.

        Args:
            llm: The custom inference client.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens).
            state: The current EnvState containing prompt and ground_truth.
            **kwargs: Additional arguments, including 'explore_probability'.

        Returns:
            - completion_texts: List of final decoded completion strings.
            - completion_tokens: List of final completion token tensors.
            - completion_lengths: List of final lengths for each completion.
        """
        explore_eps = kwargs.get('exploration_epsilon', 0.0)
        logit_processor = self._prepare_logits_processor(explore_eps)

        batch_size = state.input_ids.shape[0]
        batched_sampling_params = []

        for i in range(batch_size):
            sp = {
                'temperature': float(self.group_temperature[i]),
                'top_p': float(self.group_top_p[i]),
                'top_k': sampling_params.get('top_k', -1),
                'repetition_penalty': sampling_params.get(
                    'repetition_penalty', 1.0
                ),
                'max_new_tokens': sampling_params.get('max_new_tokens', 4096),
                'custom_params': {
                    'step': 0,
                    'replace_prob': self.replace_prob,
                    'replace_count': 0,
                },
            }
            batched_sampling_params.append(sp)

        output = llm.generate(
            prompts=state.prompt,
            sampling_params=batched_sampling_params,
            custom_logit_processor=logit_processor,
        )

        # Unpack completions
        completion_texts, completion_ids = self._process_llm_output(output)

        actual_lengths = [len(item) for item in completion_ids]
        return completion_texts, completion_ids, actual_lengths


# --- Using local HF LLM ---


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
        explore_decay: float,
        replace_source_tokens: List[int],
        replace_target_tokens: List[int],
        replace_prevent_patterns: List[List[int]],
        replace_max_count: int,
        replace_prob: float,
        **kwargs,
    ):

        super().__init__(**kwargs)

        assert len(temperature) >= 1

        self.temperature = temperature
        self.explore_steps = explore_steps
        self.explore_top_k = explore_top_k
        self.explore_skip_n = explore_skip_n
        self.explore_decay = explore_decay
        self.replace_source_tokens = replace_source_tokens
        self.replace_target_tokens = replace_target_tokens
        self.replace_prevent_patterns = replace_prevent_patterns
        self.replace_max_count = replace_max_count
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

            explore_logits_processor = HfExploreLogitsProcessor(
                initial_seq_len=input_ids.shape[1],
                tokenizer=self.tokenizer,
                group_size=self.group_size,
                temperature=self.temperature,
                explore_steps=self.explore_steps,
                explore_skip_n=self.explore_skip_n,
                explore_top_k=self.explore_top_k,
                explore_decay=self.explore_decay,
                replace_source_tokens=self.replace_source_tokens,
                replace_target_tokens=self.replace_target_tokens,
                replace_prevent_patterns=self.replace_prevent_patterns,
                replace_max_count=self.replace_max_count,
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
