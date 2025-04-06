import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedTokenizer


class ExploreLogitsProcessor(LogitsProcessor):
    """
    Processes logits for HF generation, applying temperature scaling, exploration sampling,
    and conditional token replacement based on sequence indices.

    Assumes a new instance is created for each generation request/batch.
    State like step count and replacement counts are tracked per sequence index.
    """

    def __init__(
        self,
        initial_seq_len: int,
        tokenizer: PreTrainedTokenizer,
        temperature: Union[torch.Tensor, float, List[float]],
        group_size: int,
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
        """
        Initializes the ExploreLogitsProcessor.

        Args:
            initial_seq_len: Length of the initial prompt sequence.
            tokenizer: Tokenizer for decoding sequences (used by correctness_callback).
            temperature: Temperature for logits scaling (float, list, or tensor).
            group_size: The number of sequences in the original batch request.
            device: The torch device where tensors should be placed.
            explore_steps: Number of steps for exploration sampling.
            explore_skip_n: Number of initial steps to skip before exploration.
            explore_top_k: Top-k value for exploration sampling.
            explore_decay_rate: Decay rate for explore_top_k during exploration.
            replace_source_tokens: Token IDs to penalize during replacement.
            replace_target_tokens: Token IDs to boost during replacement.
            replace_prevent_patterns: Token sequences that prevent replacement if found.
            replace_prob: Probability of applying replacement logic if conditions met.
            replace_max_per_seq: Max replacements per sequence (0 disables replacement).
            replace_boost_value: Logit boost value for target tokens.
            replace_check_top_n: Check if any of the top N original predicted tokens are in replace_source_tokens.
            correctness_callback: Function (List[str] -> List[float]) checking generated
                text correctness (score < 1.0 means incorrect, enabling replacement).
        """
        if not isinstance(initial_seq_len, int) or initial_seq_len < 0:
            raise ValueError('initial_seq_len must be a non-negative integer.')
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError('group_size must be a positive integer.')
        if (
            not isinstance(temperature, (list, torch.Tensor))
            or len(temperature) != group_size
        ):
            raise ValueError(
                'temperature must be list or tensor with same size as group_size'
            )
        if any(t < 0 for t in temperature):
            raise ValueError('temperature values cannot be negative')
        if not isinstance(explore_steps, int) or explore_steps < 0:
            raise ValueError('explore_steps must be a non-negative integer.')
        if not isinstance(explore_skip_n, int) or explore_skip_n < 0:
            raise ValueError('explore_skip_n must be a non-negative integer.')
        if not isinstance(explore_top_k, int) or explore_top_k <= 0:
            raise ValueError('explore_top_k must be a positive integer.')
        if not isinstance(explore_decay_rate, float) or not (
            0.0 < explore_decay_rate <= 1.0
        ):
            raise ValueError(
                'explore_decay_rate must be a float between 0.0 (exclusive) and 1.0 (inclusive).'
            )
        if replace_source_tokens is not None and not isinstance(
            replace_source_tokens, list
        ):
            raise TypeError(
                'replace_source_tokens must be a list of integers or None.'
            )
        if replace_target_tokens is not None and not isinstance(
            replace_target_tokens, list
        ):
            raise TypeError(
                'replace_target_tokens must be a list of integers or None.'
            )
        if replace_prevent_patterns is not None and (
            not isinstance(replace_prevent_patterns, list)
            or not all(isinstance(p, list) for p in replace_prevent_patterns)
        ):
            raise TypeError(
                'replace_prevent_patterns must be a list of lists of integers or None.'
            )
        if not isinstance(replace_prob, float) or not (
            0.0 <= replace_prob <= 1.0
        ):
            raise ValueError(
                'replace_prob must be a float between 0.0 and 1.0.'
            )
        if not isinstance(replace_max_per_seq, int) or replace_max_per_seq < 0:
            raise ValueError(
                'replace_max_per_seq must be a non-negative integer.'
            )
        if not isinstance(replace_boost_value, float):
            raise TypeError('replace_boost_value must be a float.')
        if not isinstance(replace_check_top_n, int) or replace_check_top_n <= 0:
            raise ValueError('replace_check_top_n must be a positive integer.')
        if correctness_callback is not None and not callable(
            correctness_callback
        ):
            raise TypeError('correctness_callback must be callable or None.')

        # --- Store Configuration ---
        self.initial_seq_len: int = initial_seq_len
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.group_size = group_size
        self.explore_steps: int = explore_steps
        self.explore_skip_n: int = explore_skip_n
        self.explore_top_k: int = explore_top_k
        self.explore_decay_rate: float = explore_decay_rate
        self.replace_source_tokens_list: List[int] = replace_source_tokens or []
        self.replace_target_tokens_list: List[int] = replace_target_tokens or []
        self.replace_prevent_patterns: List[List[int]] = (
            replace_prevent_patterns or []
        )
        self.replace_prob: float = replace_prob
        self.replace_max_per_seq: int = (
            replace_max_per_seq
            if self.replace_source_tokens_list
            and self.replace_target_tokens_list
            and replace_max_per_seq > 0
            else 0
        )
        self.replace_boost_value: float = replace_boost_value
        self.replace_check_top_n = replace_check_top_n
        self.correctness_callback: Optional[
            Callable[[List[str]], List[float]]
        ] = correctness_callback

        # --- Initialize State Tensors (for the full expected batch) ---
        self.temperature: torch.Tensor = self._initialize_temperature(
            temperature, group_size
        )
        self.replacement_counts: torch.Tensor = torch.zeros(
            group_size, dtype=torch.long
        )
        self.source_tokens_tensor: Optional[torch.Tensor] = (
            torch.tensor(self.replace_source_tokens_list, dtype=torch.long)
            if self.replace_source_tokens_list
            else None
        )
        self.target_tokens_tensor: Optional[torch.Tensor] = (
            torch.tensor(self.replace_target_tokens_list, dtype=torch.long)
            if self.replace_target_tokens_list
            else None
        )
        self.current_step: int = 0  # Tracks steps *after* the initial prompt
        self._current_device: Optional[torch.device] = None

    def _initialize_temperature(
        self,
        temperature: Union[torch.Tensor, float, List[float]],
        batch_size: int,
    ) -> torch.Tensor:
        """Creates the temperature tensor for the expected batch size."""
        if isinstance(temperature, (float, int)):
            temp_val = float(temperature)
            if temp_val < 0:
                raise ValueError('temperature cannot be negative')
            return torch.full((batch_size,), temp_val, dtype=torch.float32)
        else:
            if isinstance(temperature, list):
                temp_tensor = torch.tensor(temperature, dtype=torch.float32)
            elif isinstance(temperature, torch.Tensor):
                temp_tensor = temperature.to(dtype=torch.float32)
            else:  # Should be caught by initial validation, but defensive check
                raise TypeError('Unexpected temperature type')

            if temp_tensor.ndim == 0 or temp_tensor.shape[0] == 1:
                return temp_tensor.repeat(batch_size)
            elif temp_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch temperature length ({temp_tensor.shape[0]}) must match "
                    f"group_size ({batch_size})."
                )
            elif (temp_tensor < 0).any():
                raise ValueError('temperature values cannot be negative')
            else:
                return temp_tensor

    def _ensure_device(self, target_device: torch.device):
        """Moves internal tensors to the target device if they aren't already there."""
        if self._current_device == target_device:
            return  # Already on the correct device

        self.temperature = self.temperature.to(target_device)
        self.replacement_counts = self.replacement_counts.to(target_device)
        if self.source_tokens_tensor is not None:
            self.source_tokens_tensor = self.source_tokens_tensor.to(
                target_device
            )
        if self.target_tokens_tensor is not None:
            self.target_tokens_tensor = self.target_tokens_tensor.to(
                target_device
            )

        self._current_device = target_device

    def _check_replacement_patterns(
        self, generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """Checks if any 'prevent' patterns exist within the generated sequences."""
        batch_size, gen_seq_len = generated_ids.shape
        device = generated_ids.device

        if not self.replace_prevent_patterns or gen_seq_len == 0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)

        pattern_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for pattern in self.replace_prevent_patterns:
            pattern_len = len(pattern)
            if pattern_len == 0 or gen_seq_len < pattern_len:
                continue

            pattern_tensor = torch.tensor(
                pattern, device=device, dtype=torch.long
            )
            windows = generated_ids.unfold(
                dimension=1, size=pattern_len, step=1
            )
            matches = (windows == pattern_tensor.view(1, 1, pattern_len)).all(
                dim=2
            )
            pattern_found |= matches.any(dim=1)
            if pattern_found.all():
                break
        return ~pattern_found

    def _check_correctness(
        self,
        input_ids: torch.Tensor,  # Full input_ids for context
        pattern_allows_replacement: torch.Tensor,
    ) -> torch.Tensor:
        """Checks sequence correctness via callback for sequences not prevented by patterns."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # If no callback, assume all are "incorrect" (eligible for replacement if other conditions met)
        if self.correctness_callback is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)

        # Only decode and check sequences where patterns allow replacement
        indices_to_check = torch.where(pattern_allows_replacement)[0]
        is_incorrect_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )  # Default to correct

        if indices_to_check.numel() > 0:
            sequences_to_check = input_ids[indices_to_check]
            texts = self.tokenizer.batch_decode(
                sequences_to_check, skip_special_tokens=True
            )
            try:
                correctness_scores = self.correctness_callback(texts)
                if len(correctness_scores) != len(texts):
                    warnings.warn(
                        f"Correctness callback returned {len(correctness_scores)} scores, "
                        f"expected {len(texts)}. Assuming incorrect for affected sequences."
                    )
                    # Mark all checked sequences as incorrect in this ambiguous case
                    is_incorrect_mask[indices_to_check] = True
                else:
                    scores_tensor = torch.tensor(
                        correctness_scores, device=device
                    )
                    # Mark as incorrect only where score < 1.0 for the checked indices
                    is_incorrect_mask[indices_to_check] = scores_tensor < 1.0
            except Exception as e:
                warnings.warn(
                    f"Correctness callback failed: {e}. Assuming incorrect for affected sequences."
                )
                is_incorrect_mask[indices_to_check] = True

        # Return the mask indicating which sequences are considered incorrect
        return is_incorrect_mask

    def _determine_replacement_eligibility(
        self,
        input_ids: torch.Tensor,  # Full input_ids for context
    ) -> torch.Tensor:
        """Determines eligibility for replacement based on counts, patterns, and correctness."""
        # Check replacement counts against max limit for the specific sequences being processed
        current_replacement_counts = self.replacement_counts
        below_max_replacements = (
            current_replacement_counts < self.replace_max_per_seq
        )

        # Check patterns on the generated part of the sequences
        generated_ids = input_ids[:, self.initial_seq_len :]
        pattern_allows_replacement = self._check_replacement_patterns(
            generated_ids
        )

        # Check correctness only if below max replacements and patterns allow
        needs_correctness_check = (
            below_max_replacements & pattern_allows_replacement
        )
        is_incorrect = self._check_correctness(
            input_ids, needs_correctness_check
        )

        # Eligible if: below max count, pattern allows, AND (is incorrect OR no correctness check needed/performed)
        # The correctness check mask `is_incorrect` is False for sequences where `needs_correctness_check` was False,
        # so we only need to combine the three main conditions.
        eligible_for_replacement = (
            below_max_replacements & pattern_allows_replacement & is_incorrect
        )

        return eligible_for_replacement

    def _apply_replacement_logic(
        self,
        scores: torch.FloatTensor,
        eligible_mask: torch.Tensor,  # Mask relative to the current batch
        original_top_n_indices: torch.LongTensor,  # Top N indices relative to the current batch
    ) -> None:
        """Applies penalty/boost based on eligibility, original prediction, and probability."""
        if (
            not eligible_mask.any()
            or self.source_tokens_tensor is None
            or self.target_tokens_tensor is None
        ):
            return

        device = scores.device

        # Check if any of the original top N predictions were source tokens
        top_n_is_source_mask = torch.isin(
            original_top_n_indices, self.source_tokens_tensor
        )  # Shape: [current_batch_size, N]
        is_predicted_source_mask = top_n_is_source_mask.any(
            dim=1
        )  # Shape: [current_batch_size]

        # Combine eligibility mask (history, patterns, correctness) with prediction check
        combined_eligible_mask = eligible_mask & is_predicted_source_mask

        if not combined_eligible_mask.any():
            return

        # Apply probabilistic check
        prob_values = torch.rand_like(scores[:, 0], device=device)
        prob_mask = prob_values < self.replace_prob
        final_replace_mask = combined_eligible_mask & prob_mask

        num_to_replace = final_replace_mask.sum().item()
        if num_to_replace == 0:
            return

        # Get the indices within the *current batch* where replacement should happen
        replace_row_indices_in_batch = final_replace_mask.nonzero(
            as_tuple=True
        )[0]

        # Penalize source tokens for the selected rows
        scores[
            replace_row_indices_in_batch[:, None], self.source_tokens_tensor
        ] = float('-inf')

        # Boost a randomly chosen target token for the selected rows
        chosen_target_indices = torch.randint(
            0,
            len(self.replace_target_tokens_list),
            (num_to_replace,),
            device=scores.device,
        )
        chosen_target_tokens = self.target_tokens_tensor[chosen_target_indices]

        # Add boost value
        scores[
            replace_row_indices_in_batch, chosen_target_tokens
        ] += self.replace_boost_value

        # Update replacement counts using the *original* sequence indices
        self.replacement_counts[replace_row_indices_in_batch] += 1

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Dict[str, Any],
    ) -> torch.FloatTensor:
        """
        Processes logits for the next token generation step, considering sequence indices.

        Args:
            input_ids: Indices of input sequence tokens for the *current active batch*
                       ([batch_size, sequence_length]).
            scores: Raw prediction scores from the model for the *current active batch*
                    ([batch_size, vocab_size]).
            **kwargs: Must contain 'sequence_indices' (List[int] or torch.Tensor) mapping
                      rows in input_ids/scores to their original batch index.

        Returns:
            Processed prediction scores ([batch_size, vocab_size]).
        """
        batch_size, seq_len = input_ids.shape
        vocab_size = scores.shape[-1]
        device = scores.device

        self._ensure_device(device)

        # --- Update Step Count (only if generating new tokens) ---
        # We assume __call__ is invoked sequentially for generation steps
        # after the initial prompt processing.
        if seq_len > self.initial_seq_len:
            self.current_step += 1
        current_gen_step = max(0, seq_len - self.initial_seq_len - 1)

        # Needed for the replacement logic condition
        k_for_check = min(self.replace_check_top_n, vocab_size)
        _, original_top_n_indices = torch.topk(scores, k=k_for_check, dim=-1)

        # --- Temperature Scaling ---
        current_temperatures = self.temperature
        temp_expanded = current_temperatures.unsqueeze(
            1
        )  # Shape [current_batch_size, 1]

        # Handle greedy decoding (temp=0)
        zero_temp_mask = temp_expanded == 0
        if zero_temp_mask.any():
            greedy_tokens = scores.argmax(dim=-1, keepdim=True)
            # Create scores favoring only the greedy token immensely
            greedy_scores = torch.full_like(scores, float('-inf'))
            greedy_scores.scatter_(
                1, greedy_tokens, 100.0
            )  # Use a large positive value
            # Apply greedy scores where temp is 0, keep original scores otherwise
            scores = torch.where(zero_temp_mask, greedy_scores, scores)

        # Apply temperature scaling where temp > 0
        non_zero_temp_mask = temp_expanded > 0
        # Prevent division by zero or very small numbers; clamp temperature
        safe_temp = torch.clamp(temp_expanded, min=1e-8)
        scores = torch.where(non_zero_temp_mask, scores / safe_temp, scores)

        # --- Exploration Logic ---
        is_in_explore_phase = (
            self.explore_steps > 0
            and self.explore_skip_n
            <= current_gen_step
            < self.explore_skip_n + self.explore_steps
        )
        if is_in_explore_phase:
            effective_steps = current_gen_step - self.explore_skip_n
            current_explore_top_k = max(
                2,
                int(
                    self.explore_top_k
                    * (self.explore_decay_rate**effective_steps)
                ),
            )
            k = min(current_explore_top_k, vocab_size)

            # Apply uniform sampling within the top-k for exploration
            _, top_k_indices = torch.topk(scores, k=k, dim=-1)
            # Set all logits to -inf initially
            scores.fill_(float('-inf'))
            # Set logits for top-k indices to 0 (equal probability after softmax)
            scores.scatter_(dim=-1, index=top_k_indices, value=0.0)

        # --- Special Token Replacement Logic ---
        # Check if replacement is enabled and applicable at this step
        should_consider_replacement = (
            self.replace_max_per_seq > 0
            and self.replace_prob > 0.0
            and self.source_tokens_tensor
            is not None  # Ensure source/target tokens exist
            and self.target_tokens_tensor is not None
            and seq_len > self.initial_seq_len  # Only apply after prompt
            and current_gen_step
            >= (
                self.explore_steps + self.explore_skip_n
            )  # Only apply after exploration
        )

        if should_consider_replacement:
            # Determine which sequences in the current batch are eligible
            eligible_mask = self._determine_replacement_eligibility(input_ids)

            # Apply penalty/boost if eligible and other conditions met
            self._apply_replacement_logic(
                scores, eligible_mask, original_top_n_indices
            )

        return scores
