import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from transformers import (
    LogitsProcessorList,
    PreTrainedModel,
)

from rl4llm.envs.llm_env import EnvState, LLMEnv
from rl4llm.generation.explore_processor import ExploreLogitsProcessor


class ExploreLLMEnv(LLMEnv):

    def __init__(
        self,
        temperature: Union[List[float], torch.Tensor],
        explore_steps: int,
        explore_top_k: int,
        explore_skip: int,
        explore_decay_rate: float,
        replace_source_tokens: List[int],
        replace_target_tokens: List[int],
        replace_prevent_patterns: List[List[int]],
        replace_max_per_seq: int,
        replace_prob: float,
        **kwargs,
    ):

        super().__init__(**kwargs)

        assert len(temperature) >= 1

        self.temperature = temperature
        self.explore_steps = explore_steps
        self.explore_top_k = explore_top_k
        self.explore_skip = explore_skip
        self.explore_decay_rate = explore_decay_rate
        self.replace_source_tokens = replace_source_tokens
        self.replace_target_tokens = replace_target_tokens
        self.replace_prevent_patterns = replace_prevent_patterns
        self.replace_max_per_seq = replace_max_per_seq
        self.replace_prob = replace_prob

        self.accuracy_fn = None
        for fn in self.reward_functions:
            if fn.name == 'accuracy_reward':
                self.accuracy_fn = fn
                break

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
        gen_args_copy.pop('num_return_sequences', None)
        gen_args_copy['return_dict_in_generate'] = True

        # add explore logits processor
        explore_prob = kwargs.get('explore_probability', 0.0)

        if explore_prob > 0 and (random.random() < explore_prob):
            correctness_callback = None
            # TODO handle recover the ground truth from state
            # if self.accuracy_fn:
            #     correctness_callback = functools.partial(self.accuracy_fn.__call__, ground_truths=ground_truths)

            explore_logits_processor = ExploreLogitsProcessor(
                initial_seq_len=input_ids.shape[1],
                tokenizer=self.tokenizer,
                temperature=self.temperature,
                explore_steps=self.explore_steps,
                explore_skip=self.explore_skip,
                explore_top_k=self.explore_top_k,
                explore_decay_rate=self.explore_decay_rate,
                replace_source_tokens=self.replace_source_tokens,
                replace_target_tokens=self.replace_target_tokens,
                replace_prevent_patterns=self.replace_prevent_patterns,
                replace_max_per_seq=self.replace_max_per_seq,
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
        return output.sequences
