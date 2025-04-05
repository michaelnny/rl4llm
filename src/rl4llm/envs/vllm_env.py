"""Implements MDP ENV for collect samples for RL using vLLM engine"""

import logging
import random
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import vllm
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy

from rl4llm.constants import LOGGER_NAME
from rl4llm.envs.hf_llm_env import EnvState, EpisodeData, LLMEnv
from rl4llm.utils.dataset_utils import shard_dataset

logger = logging.getLogger(LOGGER_NAME)


class vLLMEnv(LLMEnv):
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
