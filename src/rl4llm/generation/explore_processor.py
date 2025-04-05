import random
import warnings
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedTokenizer


class ExploreLogitsProcessor(LogitsProcessor):
    """
    Processes logits to apply custom generation techniques like temperature scaling,
    exploration sampling, and conditional token replacement.

    Features:
    - Batch-specific temperature scaling (including greedy decoding for temp=0).
    - Exploration phase with uniform top-k sampling.
    - Conditional replacement: Penalizes source tokens and boosts target tokens based
      on correctness, pattern checks, and probability.
    """

    def __init__(
        self,
        initial_seq_len: int,
        tokenizer: PreTrainedTokenizer,
        temperature: Union[torch.Tensor, float, List[float]],
        explore_steps: int = 0,
        explore_skip: int = 0,
        explore_top_k: int = 20,
        explore_decay_rate: float = 0.9,
        replace_source_tokens: Optional[List[int]] = None,
        replace_target_tokens: Optional[List[int]] = None,
        replace_prevent_patterns: Optional[List[List[int]]] = None,
        replace_prob: float = 0.0,
        replace_max_per_seq: int = 0,
        replace_boost_value: float = 100.0,
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
            explore_steps: Number of steps for exploration sampling.
            explore_skip: Number of initial steps to skip before exploration.
            explore_top_k: Top-k value for exploration sampling.
            explore_decay_rate: Decay rate for explore_top_k during exploration.
            replace_source_tokens: Token IDs to penalize during replacement.
            replace_target_tokens: Token IDs to boost during replacement.
            replace_prevent_patterns: Token sequences that prevent replacement if found.
            replace_prob: Probability of applying replacement logic if conditions met.
            replace_max_per_seq: Max replacements per sequence (0 disables replacement).
            replace_boost_value: Logit boost value for target tokens.
            correctness_callback: Function (List[str] -> List[float]) checking generated
                text correctness (score < 1.0 means incorrect, enabling replacement).

        Raises:
            TypeError: If input types are invalid.
            ValueError: If input values are out of expected ranges.
        """
        if not isinstance(initial_seq_len, int) or initial_seq_len < 0:
            raise ValueError('initial_seq_len must be a non-negative integer.')
        if not isinstance(temperature, (float, int, list, torch.Tensor)):
            raise TypeError('temperature must be float, list, or torch.Tensor.')
        if isinstance(temperature, (float, int)) and temperature < 0:
            raise ValueError('temperature must be non-negative.')
        if isinstance(temperature, list) and any(t < 0 for t in temperature):
            raise ValueError(
                'All temperatures in the list must be non-negative.'
            )
        if isinstance(temperature, torch.Tensor) and (temperature < 0).any():
            raise ValueError(
                'All temperatures in the tensor must be non-negative.'
            )

        if not isinstance(explore_steps, int) or explore_steps < 0:
            raise ValueError('explore_steps must be a non-negative integer.')
        if not isinstance(explore_skip, int) or explore_skip < 0:
            raise ValueError('explore_skip must be a non-negative integer.')
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
        if replace_prevent_patterns is not None:
            if not isinstance(replace_prevent_patterns, list) or not all(
                isinstance(p, list) for p in replace_prevent_patterns
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
        if correctness_callback is not None and not callable(
            correctness_callback
        ):
            raise TypeError('correctness_callback must be callable or None.')

        # --- Store Configuration ---
        self.initial_seq_len: int = initial_seq_len
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.explore_steps: int = explore_steps
        self.explore_skip: int = explore_skip
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
        self.correctness_callback: Optional[
            Callable[[List[str]], List[float]]
        ] = correctness_callback

        # --- Process Temperature ---
        self._is_batch_temp: bool
        self._initial_temperature_val: Union[float, torch.Tensor]
        if isinstance(temperature, (float, int)):
            self._initial_temperature_val = float(temperature)
            self._is_batch_temp = False
        elif isinstance(temperature, list):
            self._initial_temperature_val = torch.tensor(
                temperature, dtype=torch.float32
            )
            self._is_batch_temp = True
        elif isinstance(temperature, torch.Tensor):
            self._initial_temperature_val = temperature.clone().detach().float()
            self._is_batch_temp = True

        # --- Runtime State ---
        self.temperature: Optional[torch.Tensor] = None
        self.source_tokens_tensor: Optional[torch.Tensor] = None
        self.target_tokens_tensor: Optional[torch.Tensor] = None
        self.current_step: int = 0
        self.replacement_counts: Optional[torch.Tensor] = None
        self._last_batch_size: int = -1
        self._last_device: Optional[torch.device] = None

    def _initialize_state(self, batch_size: int, device: torch.device):
        """Initializes or updates state tensors based on batch size and device."""
        # Temperature Tensor
        if self._is_batch_temp:
            if not isinstance(self._initial_temperature_val, torch.Tensor):
                raise TypeError(
                    'Internal error: _initial_temperature_val is not a Tensor for batch temp.'
                )
            temp_tensor = self._initial_temperature_val.to(
                device=device, dtype=torch.float32
            )
            if temp_tensor.dim() == 0 or temp_tensor.shape[0] == 1:
                self.temperature = temp_tensor.repeat(batch_size)
            elif temp_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch temperature length ({temp_tensor.shape[0]}) != batch size ({batch_size})."
                )
            else:
                self.temperature = temp_tensor
        else:
            self.temperature = torch.full(
                (batch_size,),
                self._initial_temperature_val,
                device=device,
                dtype=torch.float32,
            )

        # Token Tensors
        if self.replace_source_tokens and (
            self.source_tokens_tensor is None
            or self.source_tokens_tensor.device != device
        ):
            self.source_tokens_tensor = torch.tensor(
                self.replace_source_tokens, device=device, dtype=torch.long
            )
        if self.replace_target_tokens and (
            self.target_tokens_tensor is None
            or self.target_tokens_tensor.device != device
        ):
            self.target_tokens_tensor = torch.tensor(
                self.replace_target_tokens, device=device, dtype=torch.long
            )

        # Counters
        self.replacement_counts = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self.current_step = 0
        self._last_batch_size = batch_size
        self._last_device = device

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
        generated_ids: torch.Tensor,
        pattern_allows_replacement: torch.Tensor,
    ) -> torch.Tensor:
        """Checks sequence correctness via callback for sequences not prevented by patterns."""
        batch_size = generated_ids.shape[0]
        device = generated_ids.device

        if self.correctness_callback is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)

        indices_to_check = torch.where(pattern_allows_replacement)[0]
        is_incorrect_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

        if indices_to_check.numel() > 0:
            # the check correctness callback expects the full batch data
            texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            try:
                correctness_scores = self.correctness_callback(texts)
                if len(correctness_scores) != len(texts):
                    warnings.warn(
                        f"Correctness callback returned {len(correctness_scores)} scores, expected {len(texts)}. Assuming incorrect."
                    )
                    is_incorrect_mask[indices_to_check] = True
                else:
                    scores_tensor = torch.tensor(
                        correctness_scores, device=device
                    )
                    is_incorrect_mask[indices_to_check] = scores_tensor < 1.0
            except Exception as e:
                warnings.warn(
                    f"Correctness callback failed: {e}. Assuming incorrect."
                )
                is_incorrect_mask[indices_to_check] = True

        return is_incorrect_mask

    def _determine_replacement_eligibility(
        self,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Determines eligibility for replacement based on counts, patterns, and correctness."""
        if self.replacement_counts is None:
            raise RuntimeError(
                'State not initialized before checking eligibility.'
            )

        below_max_replacements = (
            self.replacement_counts < self.replace_max_per_seq
        )
        pattern_allows_replacement = self._check_replacement_patterns(
            generated_ids
        )
        is_incorrect = self._check_correctness(
            generated_ids, pattern_allows_replacement
        )

        eligible_for_replacement = (
            below_max_replacements & pattern_allows_replacement & is_incorrect
        )
        return eligible_for_replacement

    def _apply_replacement_logic(
        self, scores: torch.FloatTensor, eligible_mask: torch.Tensor
    ) -> None:
        """Applies penalty to source tokens and boost to target tokens for eligible sequences."""
        if not eligible_mask.any():
            return

        # Apply probabilistic check using torch.rand which can be mocked
        # Use shape from eligible_mask to ensure correct size for the batch
        prob_values = torch.rand(
            eligible_mask.shape,
            device=eligible_mask.device,
            dtype=torch.float32,
        )
        prob_mask = prob_values < self.replace_prob
        final_replace_mask = eligible_mask & prob_mask

        num_to_replace = final_replace_mask.sum().item()
        if num_to_replace == 0:
            return

        replace_row_indices = final_replace_mask.nonzero(as_tuple=True)[0]

        # Penalize source tokens
        if (
            self.source_tokens_tensor is not None
            and self.source_tokens_tensor.numel() > 0
        ):
            # Use advanced indexing to modify only specific rows and columns
            scores[replace_row_indices[:, None], self.source_tokens_tensor] = (
                float('-inf')
            )

        # Boost a randomly chosen target token
        if (
            self.target_tokens_tensor is not None
            and self.target_tokens_tensor.numel() > 0
        ):
            chosen_target_indices = torch.randint(
                0,
                len(self.replace_target_tokens),
                (num_to_replace,),
                device=scores.device,
            )
            chosen_target_tokens = self.target_tokens_tensor[
                chosen_target_indices
            ]

            # Use advanced indexing with scatter_add_ or direct addition
            # Ensure not boosting already penalized tokens if source/target overlap
            current_values = scores[replace_row_indices, chosen_target_tokens]
            boost_mask = ~torch.isneginf(current_values)

            # Apply boost only where mask is True and indices match
            # Need to map the boost_mask back to the original row indices
            rows_to_boost = replace_row_indices[boost_mask]
            cols_to_boost = chosen_target_tokens[boost_mask]

            if rows_to_boost.numel() > 0:
                scores[rows_to_boost, cols_to_boost] += self.replace_boost_value

        # Update replacement counts
        if self.replacement_counts is not None:
            self.replacement_counts[final_replace_mask] += 1

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Processes logits for the next token generation step.

        Args:
            input_ids: Indices of input sequence tokens ([batch_size, sequence_length]).
            scores: Raw prediction scores from the model ([batch_size, vocab_size]).

        Returns:
            Processed prediction scores ([batch_size, vocab_size]).
        """
        batch_size, seq_len = input_ids.shape
        device = scores.device

        # --- State Management ---
        if (
            self._last_batch_size != batch_size
            or self._last_device != device
            or seq_len == self.initial_seq_len
        ):
            self._initialize_state(batch_size, device)
        elif seq_len > self.initial_seq_len:
            self.current_step += 1

        if self.temperature is None or self.replacement_counts is None:
            raise RuntimeError('Processor state not initialized.')

        # --- Temperature Scaling ---
        zero_temp_mask = self.temperature == 0
        if zero_temp_mask.any():
            greedy_tokens = scores.argmax(dim=-1)
            greedy_scores_mask = torch.full_like(scores, float('-inf'))
            # Scatter 100.0 (as expected by test) to the argmax positions
            greedy_scores_mask.scatter_(1, greedy_tokens.unsqueeze(1), 100.0)
            scores = torch.where(
                zero_temp_mask.unsqueeze(1), greedy_scores_mask, scores
            )

        non_zero_temp = self.temperature.unsqueeze(1).clamp(min=1e-8)
        scores = torch.where(
            (self.temperature > 0).unsqueeze(1), scores / non_zero_temp, scores
        )

        # --- Exploration Logic ---
        is_in_explore_phase = (
            self.explore_steps > 0
            and self.explore_skip
            <= self.current_step
            < self.explore_skip + self.explore_steps
        )
        if is_in_explore_phase:
            # Decay explore top k
            effective_steps = self.current_step - self.explore_skip
            current_explore_top_k = max(
                2,
                int(
                    self.explore_top_k
                    * (self.explore_decay_rate**effective_steps)
                ),
            )
            k = min(current_explore_top_k, scores.shape[-1])
            # Boost logits for explore top-k while set the rest to -inf
            top_k_values, top_k_indices = torch.topk(scores, k=k, dim=-1)
            min_vals = torch.full_like(scores, float('-inf'))
            uniform_score = torch.zeros_like(top_k_values)
            scores = torch.scatter(
                min_vals, dim=-1, index=top_k_indices, src=uniform_score
            )

        # --- Special Token Replacement Logic ---
        should_consider_replacement = (
            self.replace_max_per_seq > 0
            and self.replace_prob > 0.0
            and seq_len > self.initial_seq_len
            and (
                self.replacement_counts.max() < self.replace_max_per_seq
                if self.replacement_counts is not None
                else False
            )
            and self.current_step >= (self.explore_steps + self.explore_skip)
        )

        if should_consider_replacement:
            generated_ids = input_ids[:, self.initial_seq_len :]
            eligible_mask = self._determine_replacement_eligibility(
                generated_ids
            )
            self._apply_replacement_logic(
                scores, eligible_mask
            )  # Modifies scores in-place

        return scores
