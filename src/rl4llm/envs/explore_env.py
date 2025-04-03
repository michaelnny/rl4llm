import logging
import random
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import (
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from rl4llm.envs.llm_env import EnvState, LLMEnv
from rl4llm.generations.explore_processor import ExploreLogitsProcessor


class ExploreLLMEnv(LLMEnv):

    def __init__(
        self,
        temperatures: Union[List[float], torch.Tensor],
        explore_steps: int,
        explore_top_k: int,
        explore_skip: int,
        replace_source_tokens: List[int],
        replace_target_tokens: List[int],
        replace_prevent_patterns: List[List[int]],
        replace_max_per_seq: int,
        replace_prob: float,
        replace_threshold: float,
        **kwargs,
    ):

        super().__init__(**kwargs)

        assert len(temperatures) >= 1

        self.temperatures = temperatures
        self.explore_steps = explore_steps
        self.explore_top_k = explore_top_k
        self.explore_skip = explore_skip
        self.replace_source_tokens = replace_source_tokens
        self.replace_target_tokens = replace_target_tokens
        self.replace_prevent_patterns = replace_prevent_patterns  # if sequence has some pattern, do not perform replacement
        self.replace_max_per_seq = replace_max_per_seq
        self.replace_threshold = replace_threshold
        self.replace_prob = replace_prob

        self.explore_processor = ExploreLogitsProcessor(
            temperatures=self.temperatures,
            explore_steps=self.explore_steps,
            explore_skip=self.explore_skip,
            explore_top_k=self.explore_top_k,
            replace_source_tokens=self.replace_source_tokens,
            replace_target_tokens=self.replace_target_tokens,
            replace_prevent_patterns=self.replace_prevent_patterns,
            replace_max_per_seq=self.replace_max_per_seq,
            replace_prob=self.replace_prob,
            replace_threshold=self.replace_threshold,
        )

    def _generate_completions(
        self,
        llm: PreTrainedModel,
        gen_args: Dict,
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Generates completions using the LLM."""

        input_ids = state.input_ids.to(llm.device)
        attention_mask = state.attention_mask.to(llm.device)
        gen_args_copy = gen_args.copy()
        gen_args_copy.pop('input_ids', None)
        gen_args_copy.pop('attention_mask', None)
        gen_args_copy['return_dict_in_generate'] = True

        # add explore logits processor
        explore_prob = gen_args_copy.pop('explore_probability', 0.0)

        if explore_prob > 0 and (random.random() < explore_prob):
            self.explore_processor.reset()
            gen_args_copy['logits_processor'] = LogitsProcessorList(
                [self.explore_processor]
            )

        output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_args_copy,
        )
        return output.sequences
