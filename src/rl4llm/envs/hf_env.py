"""Implements MDP ENV for collect samples using HF model"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LogitsProcessorList, PreTrainedModel

from rl4llm.constants import LOGGER_NAME
from rl4llm.core.base_env import (
    BaseMDPEnv,
    ChatMessage,
    EnvState,
    SampleState,
)

logger = logging.getLogger(LOGGER_NAME)


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
        env_state: EnvState,
        llm: PreTrainedModel,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        """
        Default interaction loop: Performs a single generation step.

        Args:
            env_state: The starting state from _prepare_initial_state.
            llm: The language model.
            generation_config: Configuration for generation (max_new_tokens, etc.).
            **kwargs: Additional arguments (unused in default).

        Returns:
            EnvState: The final state after generation, with updated SampleStates.
        """

        # 1. Prepare inputs for the LLM
        # Convert initial messages to token IDs with padding
        batch_prompts = self._convert_to_batch_prompts(env_state)

        # Tokenize the formatted prompts
        batch_inputs = self.tokenizer(
            batch_prompts,
            padding=True,
            padding_side='left',
            return_tensors='pt',
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
        prompt_lengths = batch_inputs['input_ids'].shape[1]
        generated_ids = outputs[:, prompt_lengths:]
        generated_responses = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # 3. Update each SampleState object *in place*
        for i, sample_state in enumerate(env_state.sample_states):
            # Safely get the generated text from the corresponding output
            generated_text = generated_responses[i].strip()

            # Append the new assistant message to the sample's history
            sample_state.messages.append(
                ChatMessage(role='assistant', content=generated_text)
            )

            # Mark this sample as done and record the step
            sample_state.done = True
            sample_state.current_step = 1  # It always takes exactly one step

        # 4. Return the *modified* EnvState object
        return env_state
