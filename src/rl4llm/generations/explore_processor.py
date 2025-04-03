import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


class ExploreLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that implements the core features:
    - Exploration during initial generation steps
    - Group-specific temperature control
    - Special token replacement
    """

    def __init__(
        self,
        temperatures: Union[List[float], torch.Tensor],
        explore_steps: int,
        explore_skip: int,
        explore_top_k: int,
        replace_source_tokens: List[int],
        replace_target_tokens: List[int],
        replace_prevent_patterns: List[List[int]],
        replace_prob: float,
        replace_max_per_seq: int = 3,
        replace_threshold: float = 0.8,
    ):
        """
        Initialize the ExploreLogitsProcessor.

        Args:
            replace_source_tokens: List of token IDs to replace
            replace_target_tokens: List of token IDs to replace with
            replace_prevent_patterns: List of token sequences that prevent replacement
            temperatures: Temperature for sampling (can be per-batch)
            explore_steps: Number of initial steps to use uniform sampling
            explore_skip: Steps to skip before starting exploration
            explore_top_k: For exploration, how many top tokens to sample from uniformly
            replace_prob: Probability to replace a source token
            replace_max_per_seq: Maximum number of token replacements allowed per sequence
            replace_threshold: Threshold to detect high probability of source tokens
        """
        self.replace_source_tokens = replace_source_tokens or []
        self.replace_target_tokens = replace_target_tokens or []
        self.replace_prevent_patterns = replace_prevent_patterns or []
        self.explore_steps = explore_steps
        self.explore_skip = explore_skip
        self.explore_top_k = explore_top_k
        self.replace_prob = replace_prob
        self.replace_max_per_seq = replace_max_per_seq
        self.replace_threshold = replace_threshold

        # Track current generation step
        self.current_step = 0
        self.temperatures = temperatures

        # Convert source tokens to a set for faster lookup
        self.replace_source_tokens_set = set(self.replace_source_tokens)

        # Track replacements per sequence
        self.replacement_counts = None

    def reset(self) -> None:
        """
        Reset the internal state to allow reusing the processor.
        This resets the step counter and replacement counts.
        """
        self.current_step = 0
        self.replacement_counts = None

    def update_config(self, **kwargs):
        """
        Update configuration parameters of the processor.

        Args:
            **kwargs: Key-value pairs of parameters to update.
                Supported parameters: temperatures, replace_prob, max_replacements_per_seq,
                explore_steps, explore_skip, explore_top_k, replace_threshold,
                replace_source_tokens, replace_target_tokens, replace_prevent_patterns

        Returns:
            The processor instance for chaining
        """
        valid_params = {
            'temperatures',
            'explore_steps',
            'explore_skip',
            'explore_top_k',
            'replace_prob',
            'replace_max_per_seq',
            'replace_threshold',
            'replace_source_tokens',
            'replace_target_tokens',
            'replace_prevent_patterns',
        }

        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

                # Update the source tokens set if source tokens are updated
                if key == 'replace_source_tokens':
                    self.replace_source_tokens_set = set(value)
            else:
                print(
                    f"Warning: Parameter '{key}' is not supported for updating."
                )

    def _check_for_patterns(
        self, input_ids: torch.LongTensor
    ) -> torch.BoolTensor:
        """Check if any sequences contain patterns that prevent replacement."""
        batch_size = input_ids.shape[0]
        pattern_found = torch.zeros(
            batch_size, dtype=torch.bool, device=input_ids.device
        )

        for pattern in self.replace_prevent_patterns:
            pattern_tensor = torch.tensor(
                pattern, device=input_ids.device, dtype=torch.long
            )
            pattern_len = len(pattern)

            if input_ids.shape[1] >= pattern_len:
                # Check for pattern matches in each sequence
                for i in range(batch_size):
                    # Simple sliding window approach
                    for j in range(input_ids.shape[1] - pattern_len + 1):
                        window = input_ids[i, j : j + pattern_len]
                        if torch.all(window == pattern_tensor):
                            pattern_found[i] = True
                            break

        return ~pattern_found  # Return where patterns were NOT found

    def _apply_exploration(
        self, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply uniform exploration to logits."""
        batch_size, vocab_size = scores.shape

        # Get top-k token indices for each item in batch
        top_k = min(self.explore_top_k, vocab_size)
        top_k_values, top_k_indices = torch.topk(scores, k=top_k, dim=-1)

        # Create uniform distribution over top-k tokens
        uniform_logits = torch.full_like(scores, -1e8)

        for i in range(batch_size):
            uniform_logits[i, top_k_indices[i]] = 10.0

        return uniform_logits

    def _apply_temperature(
        self, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply temperature to logits."""

        # Per-sequence temperature
        batch_size = scores.shape[0]
        temp_tensor = self.temperatures

        if isinstance(temp_tensor, list):
            temp_tensor = torch.tensor(temp_tensor, device=scores.device)

        # Ensure temperature tensor has right size and is on correct device
        temp_tensor = temp_tensor.to(device=scores.device, dtype=torch.float)
        if temp_tensor.dim() == 0:
            temp_tensor = temp_tensor.repeat(batch_size)
        elif temp_tensor.size(0) != batch_size:
            # Fix potential size mismatch by broadcasting
            temp_tensor = temp_tensor.expand(batch_size)

        # Apply temperature differently to each sequence
        scaled_scores = torch.zeros_like(scores)
        for i in range(batch_size):
            if temp_tensor[i] == 0:
                scaled_scores[i] = scores[i]  # Keep original for greedy
            else:
                scaled_scores[i] = scores[i] / max(temp_tensor[i], 1e-8)

        return scaled_scores

    def _apply_token_replacement(
        self, scores: torch.FloatTensor, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Process token replacement by detecting when source tokens have high probability
        and modifying logits to favor target tokens instead.

        This approximates the token replacement feature of ExploreLLMGenerator.
        """
        if not self.replace_source_tokens or not self.replace_target_tokens:
            return scores

        batch_size, vocab_size = scores.shape
        modified_scores = scores.clone()

        # Initialize replacement counter if not yet initialized
        if self.replacement_counts is None:
            self.replacement_counts = torch.zeros(
                batch_size, dtype=torch.int, device=scores.device
            )

        # Check if any sequence has reached the max replacement limit
        can_still_replace = self.replacement_counts < self.replace_max_per_seq

        # 1. Check if patterns exist that would prevent replacement
        can_replace = self._check_for_patterns(input_ids)

        # Combine with replacement count check
        can_replace = can_replace & can_still_replace

        if not can_replace.any():
            return scores

        # 2. Check if source tokens have high probability (approximation of next token prediction)
        source_probs = torch.softmax(scores, dim=-1)

        # Check if any source token has probability above threshold
        has_high_source_token = torch.zeros(
            batch_size, dtype=torch.bool, device=scores.device
        )
        for idx, token_id in enumerate(self.replace_source_tokens):
            if token_id < vocab_size:  # Ensure token is within vocabulary
                has_high_source_token |= (
                    source_probs[:, token_id] > self.replace_threshold
                )

        # Combine conditions
        replacement_candidates = can_replace & has_high_source_token

        if not replacement_candidates.any():
            return scores

        # 3. Apply probabilistic replacement
        replacements_made = torch.zeros(
            batch_size, dtype=torch.bool, device=scores.device
        )

        for i in range(batch_size):
            if (
                replacement_candidates[i]
                and random.random() < self.replace_prob
            ):
                # Boost target tokens and reduce source tokens
                for source_id in self.replace_source_tokens:
                    if source_id < vocab_size:
                        modified_scores[i, source_id] = -1e8

                # Boost target tokens
                for target_id in self.replace_target_tokens:
                    if (
                        target_id < vocab_size
                    ):  # Ensure target token is in vocabulary
                        modified_scores[i, target_id] = 1e8

                # Track that replacement was made
                replacements_made[i] = True

        # Update replacement counts for sequences where replacements were made
        self.replacement_counts += replacements_made.to(dtype=torch.int)

        return modified_scores

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process the logits to implement:
        1. Exploration during initial steps
        2. Apply temperature (batch-specific if needed)
        3. Token replacement when appropriate
        """
        is_exploration_phase = (
            self.explore_steps > 0
            and self.current_step >= self.explore_skip
            and self.current_step < (self.explore_steps + self.explore_skip)
        )

        if is_exploration_phase:
            # Apply uniform sampling from top-k during exploration phase
            scores = self._apply_exploration(scores)
        else:
            # Apply temperature
            scores = self._apply_temperature(scores)

            # Apply token replacement (approximating)
            if self.replace_prob > 0 and self.current_step > (
                self.explore_steps + self.explore_skip
            ):
                scores = self._apply_token_replacement(scores, input_ids)

        # Increment step counter
        self.current_step += 1

        return scores
