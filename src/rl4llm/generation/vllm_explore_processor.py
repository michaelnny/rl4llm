import random
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer


class vLLMExplorationLogitsProcessor:
    """
    Processes logits for vLLM engine generation, applying  exploration sampling,
    and conditional token replacement based on sequence indices.

    IMPORTANT: vLLM calls the logits processor on a single sequence level, not batch level.
    So we can't reuse the same code from the HF logits processor.
    """

    def __init__(
        self,
        initial_seq_len: int,
        tokenizer: PreTrainedTokenizer,
        explore_steps: int = 0,
        explore_skip_n: int = 0,
        explore_top_k: int = 20,
        explore_decay_rate: float = 0.9,
        replace_source_tokens: Optional[List[int]] = None,
        replace_target_tokens: Optional[List[int]] = None,
        replace_prevent_patterns: Optional[List[List[int]]] = None,
        replace_prob: float = 0.0,
        replace_max_per_seq: int = 0,
        replace_boost_value: float = 100.0,
        replace_check_top_n: int = 3,
        correctness_callback: Optional[
            Callable[[List[str]], List[float]]
        ] = None,
    ):
        self.initial_seq_len: int = initial_seq_len
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.explore_steps: int = explore_steps
        self.explore_skip_n: int = explore_skip_n
        self.explore_top_k: int = explore_top_k
        self.explore_decay_rate: float = explore_decay_rate
        self.replace_source_tokens: List[int] = replace_source_tokens or []
        self.replace_target_tokens: List[int] = replace_target_tokens or []
        self.replace_prevent_patterns: List[List[int]] = (
            replace_prevent_patterns or []
        )
        self.replace_prob: float = replace_prob
        self.replace_max_per_seq: int = (
            replace_max_per_seq
            if self.replace_source_tokens
            and self.replace_target_tokens
            and replace_max_per_seq > 0
            else 0
        )
        self.replace_boost_value: float = replace_boost_value
        self.replace_check_top_n = replace_check_top_n
        self.correctness_callback: Optional[
            Callable[[List[str]], List[float]]
        ] = correctness_callback

        self.replacement_count = 0
        self.step_t = 0

    @torch.inference_mode()
    def __call__(
        self, past_token_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """Takes in past token ids and the logits for next token for a single sequence from the current batch."""

        generated_len = len(past_token_ids)

        # Handle exploration mode
        explore_start = (
            self.explore_steps > 0
            and (generated_len - self.explore_skip_n) < self.explore_steps
        )
        if self.explore_skip_n and generated_len < self.explore_skip_n:
            explore_start = False

        if explore_start and self.explore_top_k > 1:
            # print(f"EXPLORING random start for sequence {seq_idx}...")
            effective_steps = max(0, generated_len - self.explore_skip_n)
            current_explore_top_k = max(
                10,
                int(
                    self.explore_top_k
                    * (self.explore_decay_rate**effective_steps)
                ),
            )
            explore_k = min(10, current_explore_top_k)
            exp_top_k_values, exp_top_k_indices = torch.topk(
                logits, k=explore_k
            )
            logits.fill_(1e-8)
            logits.scatter_(
                0, exp_top_k_indices, torch.ones_like(exp_top_k_values) * 100.0
            )
            return logits

        # Check if next token is likely one of the  source token, like 'EOS' or '</think>'
        is_next_special = False
        _, top_k_indices = torch.topk(
            logits, k=self.replace_check_top_n, dim=-1
        )
        top_k_indices = top_k_indices.flatten().tolist()

        if any(
            tok in top_k_indices for tok in self.replace_source_tokens
        ) and all(
            tok not in past_token_ids[-20:]
            for tok in self.replace_target_tokens
        ):
            is_next_special = True

        # Handle token replacement
        should_replace = (
            is_next_special
            and self.replace_source_tokens
            and self.replace_target_tokens
            and self.replace_prob > 0
            and self.replace_max_per_seq > 0
            and self.replacement_count < self.replace_max_per_seq
            and generated_len > 50
        )

        if should_replace:
            generated_ids = past_token_ids  # past_token_ids[prompt_len:]
            if self._check_patterns(generated_ids):
                is_incorrect = True
                if self.correctness_callback is not None:
                    completion_text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    score = self.correctness_callback([completion_text])
                    is_incorrect = score[0] < 1.0

                if is_incorrect and random.random() < self.replace_prob:
                    # print(f"EXPLORING replace token for sequence {seq_idx}...")
                    if self.replacement_count < self.replace_max_per_seq:
                        self.replacement_count += 1
                        logits.fill_(1e-8)
                        logits[self.replace_target_tokens] = 100.0
                        return logits

        return logits

    def _check_patterns(self, token_ids: list[int]) -> bool:
        if not self.replace_prevent_patterns:
            return True
        for pattern in self.replace_prevent_patterns:
            pattern_len = len(pattern)
            for i in range(len(token_ids) - pattern_len + 1):
                if token_ids[i : i + pattern_len] == pattern:
                    return False
        return True
