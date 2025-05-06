"""Implements MDP ENV for collect samples using SGLang inference server with a custom HTTP client"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rl4llm.core.base_env import (
    BaseMDPEnv,
    ChatMessage,
    EnvState,
    SampleState,
)
from rl4llm.core.base_inference_client import InferenceClient

logger = logging.getLogger(__name__)


class SglMDPEnv(BaseMDPEnv):
    """
    Simple one-step MDP Environment using SGLang inference server with a custom HTTP client.
    """

    @torch.inference_mode()
    def _run_interaction_loop(
        self,
        env_state: EnvState,
        llm: InferenceClient,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        """
        Performs a single generation step for all samples using the SampleState structure.

        Args:
            env_state: The starting state containing a list of SampleState objects.
            llm: The language model inference client.
            sampling_params: Configuration for generation.
            **kwargs: Additional arguments (unused in default).

        Returns:
            EnvState: The final state after one generation step, with updated SampleStates.
        """
        logger.debug(
            f"Rank {self.rank}: Running single-step interaction loop with SampleState design."
        )

        # 1. Prepare inputs for the LLM from the list of SampleStates
        # Convert message histories to prompt strings
        batch_prompts = self._convert_to_batch_prompts(env_state)

        # 2. Call the inference API for LLM generation
        try:
            outputs = llm.generate(
                prompts=batch_prompts,
                sampling_params=sampling_params,
            )
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error during LLM generation in single-step loop: {e}",
                exc_info=True,
            )
            # Return the state marked as done
            return env_state

        # 3. Update each SampleState object *in place*
        for i, sample_state in enumerate(env_state.sample_states):
            generated_text = outputs[i].get('text', '').strip()
            # Append the new assistant message to the sample's history
            sample_state.messages.append(
                ChatMessage(role='assistant', content=generated_text)
            )

            # Mark this sample as done and record the step
            sample_state.done = True
            sample_state.current_step = 1  # It always takes exactly one step

        # 4. Return the *modified* EnvState object
        return env_state
