"""Implements MDP ENV for collect samples using HF model"""

import functools
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LogitsProcessorList, PreTrainedModel

from rl4llm.core.base_env import (
    BaseMDPEnv,
    EnvState,
    ChatMessage,
    EpisodeData,
    BaseRewardFunction,
)

logger = logging.getLogger(__name__)


class HfMDPEnv(BaseMDPEnv):
    """
    Simple one-step MDP Environment for generating training samples with LLM models from HuggingFace library.

    This environment handles the workflow of sampling prompts from a dataset,
    generating completions using a provided LLM, calculating rewards, and
    returning structured episode data.
    """

    @torch.inference_mode()
    def _run_interaction_loop(
        self,
        initial_state: EnvState,
        llm: PreTrainedModel,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        """
        Default interaction loop: Performs a single generation step.

        Suitable for single-step MDPs where the full completion is generated at once.
        Subclasses for multi-step MDPs or tool use should override this method.

        Args:
            initial_state: The starting state from _prepare_initial_state.
            llm: The language model.
            generation_config: Configuration for generation (max_new_tokens, etc.).
            **kwargs: Additional arguments (unused in default).

        Returns:
            EnvState: The final state after generation, with updated batch_messages.
        """
        logger.debug(f"Rank {self.rank}: Running default single-step interaction loop.")

        # 1. Prepare inputs for the LLM
        # Convert initial messages to token IDs with padding
        batch_prompts = self._convert_batch_message_to_prompt(
            initial_state.batch_messages
        )

        # Tokenize the formatted prompts
        batch_inputs = self.tokenizer(
            batch_prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(llm.device)

        # 2. Generate completions
        outputs = llm.generate(
            **batch_inputs,
            **sampling_params,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 3. Decode and update state
        # We need to reconstruct the chat history including the assistant's reply.
        # This is tricky because batch_decode gives the *full* text including prompt.
        # A robust way is to decode the generated part *only*.
        prompt_lengths = batch_inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_lengths:]
        generated_responses = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        final_batch_messages = []
        for i, initial_msg_list in enumerate(initial_state.batch_messages):
            new_history = initial_msg_list + [
                ChatMessage(role="assistant", content=generated_responses[i].strip())
            ]
            final_batch_messages.append(new_history)

        # 4. Return the final state
        final_state = EnvState(
            batch_messages=final_batch_messages,  # Updated messages
            batch_ground_truth=initial_state.batch_ground_truth,
            batch_init_prompt_size=initial_state.batch_init_prompt_size,
            # Copy any other relevant fields from initial_state if needed
        )
        return final_state
