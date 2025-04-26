"""Implements MDP ENV for collect samples using HF model"""

import functools
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LogitsProcessorList, PreTrainedModel

from rl4llm.core.base_env import BaseEnv, EnvState, EpisodeData

logger = logging.getLogger(__name__)


class HfMDPEnv(BaseEnv):
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

        return self._process_completions(output.sequences[:, start:])

    def _process_completions(
        self,
        sequences: torch.Tensor,
    ) -> Tuple[List[str], List[torch.Tensor]]:
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
        return texts, tokens_list

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

        completions, completion_tokens = self._generate_completions(
            llm, sampling_params, state, **kwargs
        )
        return self._to_episodes(state, completions, completion_tokens)
