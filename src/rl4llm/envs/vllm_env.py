"""Implements MDP ENV for collect samples for RL using vLLM engine"""

import functools
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import vllm

from rl4llm.constants import LOGGER_NAME
from rl4llm.envs.env import Env, EnvState, EpisodeData, ExploreEnv

logger = logging.getLogger(LOGGER_NAME)


class vLLMEnv(Env):
    """
    Environment for generating training samples using vLLM engine.
    """

    def __init__(self, **kwargs):
        """
        Initializes the vLLMEnv.
        """
        super().__init__(**kwargs)

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

        state = self.reset()
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


class vLLMExploreEnv(vLLMEnv, ExploreEnv):

    import vllm

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
            from rl4llm.generation.vllm_explore_processor import (
                vLLMExplorationLogitsProcessor,
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
