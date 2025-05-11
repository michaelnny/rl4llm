"""Implements custom MDP ENV for collect samples using SGLang inference server with a custom HTTP client"""

import functools
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rl4llm.constants import LOGGER_NAME
from rl4llm.core.base_env import BaseMDPEnv, ChatMessage
from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.envs.sgl_env import EnvState
from rl4llm.generation.sgl_explore_procesor import SglExploreLogitProcessor

logger = logging.getLogger(LOGGER_NAME)


# --- Using SGLang inference client ---


class ExploreSglMDPEnv(BaseMDPEnv):
    """Simple one-step MDP Environment where we apply some custom logits processor
    to the generation process to encourage exploration."""

    def __init__(
        self,
        group_temperature: torch.Tensor,
        group_top_p: torch.Tensor,
        random_start_steps: int,
        random_start_top_k: int,
        random_start_skip_n: int,
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

        self.group_temperature = group_temperature
        self.group_top_p = group_top_p
        self.random_start_steps = random_start_steps
        self.random_start_top_k = random_start_top_k
        self.random_start_skip_n = random_start_skip_n

    def _prepare_logits_processor(self, explore_prob: float) -> Optional[str]:
        """Creates the explore logits processor string if conditions are met."""
        if explore_prob > 0 and random.random() < explore_prob:
            explore_logit_processor = SglExploreLogitProcessor(
                random_start_steps=self.random_start_steps,
                random_start_top_k=self.random_start_top_k,
                random_start_skip_n=self.random_start_skip_n,
            )
            return explore_logit_processor.to_str()
        return None

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

        # 1. Prepare inputs for the LLM from the list of SampleStates
        # Convert message histories to prompt strings
        batch_prompts = self._convert_to_batch_prompts(env_state)

        explore_eps = kwargs.get('explore_epsilon', 0.0)
        logit_processor = self._prepare_logits_processor(explore_eps)

        batch_size = len(batch_prompts)

        # Build sampling params for each sequence
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
                },
            }
            batched_sampling_params.append(sp)

        # 2. Call the inference API for LLM generation
        outputs = llm.generate(
            prompts=batch_prompts,
            sampling_params=batched_sampling_params,
            custom_logit_processor=logit_processor,
        )

        # 3. Update each SampleState object *in place*
        for i, sample_state in enumerate(env_state.sample_states):
            # Fallback to dummy text to ensure code works
            generated_text = (
                outputs[i].get('text', 'I can not answer this question').strip()
            )
            # Append the new assistant message to the sample's history
            sample_state.messages.append(
                ChatMessage(role='assistant', content=generated_text)
            )

            # Mark this sample as done and record the step
            sample_state.done = True
            sample_state.current_step = 1

        # 4. Return the *modified* EnvState object
        return env_state
