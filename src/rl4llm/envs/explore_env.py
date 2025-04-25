import functools
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.envs.sgl_env import EnvState, SglMDPEnv
from rl4llm.generation.sgl_explore_procesor import SglExploreLogitProcessor

logger = logging.getLogger(__name__)


# --- Using SGLang inference client ---


class ExploreSglMDPEnv(SglMDPEnv):
    """An extension of the standard SglMDPEnv
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

        return self._process_llm_output(output)
