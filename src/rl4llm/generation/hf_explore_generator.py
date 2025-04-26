import logging
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput

logger = logging.getLogger(__name__)


class HfExploreLLMGenerator:
    """
    A custom class for text generation using a HF language model (LLM).
    Supports batch-specific temperatures, exploring start, and special token replacement.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        source_tokens: Optional[List[int]] = None,
        target_tokens: Optional[List[int]] = None,
        prevent_patterns: Optional[List[List[int]]] = None,
    ):
        """
        Initialize the StochasticLLMGenerator.

        Args:
            model (PreTrainedModel): A pre-trained transformer-based model for text generation.
            tokenizer (PreTrainedTokenizer): A pre-trained tokenizer for the model.
            device: Torch device.
            source_tokens (List[int]): List of token IDs to replace, e.g., "EOS" or "</think>".
            target_tokens (List[int]): List of token IDs to replace with, e.g., "Wait", "Hmm".
            prevent_patterns (List[List[int]]): List of token sequences that, if present, prevent replacement,
                                               e.g., [[27, 9217, 29]].
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.source_tokens = source_tokens or []
        self.target_tokens = target_tokens or []
        self.prevent_patterns = prevent_patterns or []
        self.source_tokens_tensor = (
            torch.tensor(
                self.source_tokens, device=self.device, dtype=torch.long
            )
            if self.source_tokens
            else None
        )

    def _has_pattern(
        self, input_ids: torch.Tensor, pattern: List[int]
    ) -> torch.Tensor:
        """
        Check if the given pattern exists in the input sequences for each batch.

        Args:
            input_ids (torch.Tensor): Current token IDs of shape [batch_size, seq_len].
            pattern (List[int]): The pattern to check for, e.g., [27, 9217, 29].

        Returns:
            torch.Tensor: Boolean tensor of shape [batch_size] indicating if the pattern is present.
        """
        batch_size, seq_len = input_ids.shape
        # Ensure pattern is on the correct device
        pattern_tensor = torch.tensor(
            pattern, device=input_ids.device, dtype=torch.long
        )
        pattern_len = pattern_tensor.size(0)

        if seq_len < pattern_len:
            return torch.zeros(
                batch_size, dtype=torch.bool, device=input_ids.device
            )

        # Use unfold for potentially better performance than as_strided directly
        windows = input_ids.unfold(dimension=1, size=pattern_len, step=1)
        # Check for matches against the pattern tensor
        matches = (windows == pattern_tensor.view(1, 1, pattern_len)).all(dim=2)
        return matches.any(dim=1)

    def _check_replacement_patterns(
        self, generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if the generated sequences contain any patterns that would prevent replacement.

        Args:
            generated_ids (torch.Tensor): Generated token IDs of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Boolean tensor of shape [batch_size] indicating if replacement is allowed.
        """
        batch_size = generated_ids.shape[0]
        if not self.prevent_patterns:
            return torch.ones(
                batch_size, dtype=torch.bool, device=generated_ids.device
            )

        pattern_found = torch.zeros(
            batch_size, dtype=torch.bool, device=generated_ids.device
        )
        for pattern in self.prevent_patterns:
            # Pass input_ids directly now assuming _has_pattern handles batching
            pattern_found |= self._has_pattern(generated_ids, pattern)

        return ~pattern_found

    def _check_correctness(
        self,
        generated_ids: torch.Tensor,
        can_replace: torch.Tensor,
        correctness_callback: Optional[Callable],
    ) -> torch.Tensor:
        """
        Check if the generated sequences are correct using the provided callback.

        Args:
            generated_ids (torch.Tensor): Generated token IDs of shape [batch_size, seq_len].
            can_replace (torch.Tensor): Boolean tensor of shape [batch_size] indicating if replacement is allowed.
            correctness_callback (Optional[Callable]): Function to evaluate correctness.

        Returns:
            torch.Tensor: Boolean tensor of shape [batch_size] indicating incorrectness.
        """
        batch_size = generated_ids.shape[0]
        incorrect_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=generated_ids.device
        )

        if correctness_callback is None:
            return ~incorrect_mask  # Assume all incorrect if no callback

        indices_to_check = torch.where(can_replace)[0]
        if indices_to_check.numel() == 0:
            return incorrect_mask

        # Consider batch decoding if tokenizer supports it well and it's faster
        sequences_to_check = generated_ids[indices_to_check]
        texts = self.tokenizer.batch_decode(
            sequences_to_check, skip_special_tokens=True
        )

        for i, text in enumerate(texts):
            original_index = indices_to_check[i]
            # Callback should return a value convertible to float, < 1.0 means incorrect
            try:
                is_incorrect = float(correctness_callback(text)) < 1.0
                incorrect_mask[original_index] = is_incorrect
            except Exception as e:
                # Handle potential errors in the callback gracefully
                print(
                    f"Warning: Correctness callback failed for sequence {original_index}: {e}"
                )
                incorrect_mask[original_index] = True

        return incorrect_mask

    def _replace_special_tokens(
        self,
        next_tokens: torch.Tensor,
        can_replace: torch.Tensor,
        replace_prob: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly replace source special tokens with the target token if allowed.

        Args:
            next_tokens (torch.Tensor): Tokens to potentially replace, shape [batch_size].
            can_replace (torch.Tensor): Boolean tensor of shape [batch_size] indicating if replacement is allowed.
            replace_prob (float): Probability of replacement, between 0.0 and 1.0.

        Returns:
            torch.Tensor: Tokens with some possibly replaced, shape [batch_size].
            torch.Tensor: The replacement mask.
        """
        if (
            self.source_tokens_tensor is None  # Check tensor presence
            or not self.target_tokens  # Check target tokens list
            or can_replace.sum() == 0
            or replace_prob <= 0.0
        ):
            return next_tokens, torch.zeros_like(can_replace)

        batch_size = next_tokens.size(0)
        device = next_tokens.device
        modified_tokens = next_tokens.clone()

        # Determine which tokens *are* source tokens (needed if replacement depends on it)
        # is_source_token = (next_tokens.unsqueeze(-1) == self.source_tokens_tensor).any(dim=-1)
        # If replacement should ONLY happen if the predicted token is a source_token:
        # replace_eligible = can_replace & is_source_token
        # Using original logic: replacement happens based on `can_replace` flag, independent of what next_token is
        replace_eligible = can_replace

        # Apply probability check only to eligible items
        prob_mask = torch.rand(batch_size, device=device) < replace_prob
        final_replace_mask = replace_eligible & prob_mask

        if final_replace_mask.sum() > 0:
            # Randomly choose one target token *per item* that needs replacement
            num_to_replace = final_replace_mask.sum().item()
            chosen_target_tokens = torch.tensor(
                random.choices(self.target_tokens, k=num_to_replace),
                device=device,
                dtype=next_tokens.dtype,  # Ensure dtype match
            )
            # Assign chosen tokens only where final_replace_mask is True
            modified_tokens[final_replace_mask] = chosen_target_tokens

        return (
            modified_tokens,
            final_replace_mask,
        )

    def _determine_replacement_eligibility(
        self,
        generated_ids: torch.Tensor,  # Use generated_ids (excluding prompt)
        next_tokens: torch.Tensor,
        replacement_counts: torch.Tensor,
        replace_max_per_seq: int,
        correctness_callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Determine which sequences are eligible for token replacement.

        Args:
            generated_ids (torch.Tensor): Generated token IDs of shape [batch_size, seq_len].
            next_tokens (torch.Tensor): Batch of next token IDs of shape [batch_size].
            replacement_counts (torch.Tensor): Count of replacements already made.
            replace_max_per_seq (int): Maximum number of replacements allowed.
            correctness_callback (Optional[Callable]): Function to evaluate correctness.

        Returns:
            torch.Tensor: Boolean tensor of shape [batch_size] indicating eligibility for replacement.
        """
        # Check if the *predicted* next_tokens are in source_tokens
        if self.source_tokens_tensor is None:
            return torch.zeros_like(
                next_tokens, dtype=torch.bool
            )  # Ensure bool type

        is_source_token = (
            next_tokens.unsqueeze(-1) == self.source_tokens_tensor
        ).any(dim=-1)

        # If no predicted tokens are source tokens, no replacement needed based on this condition
        if not is_source_token.any():
            return torch.zeros_like(next_tokens, dtype=torch.bool)

        # Base eligibility mask: is a source token AND hasn't reached max replacements
        below_max_replacements = replacement_counts < replace_max_per_seq
        can_replace_base = is_source_token & below_max_replacements

        # Get indices where base conditions are met to perform further checks
        eligible_indices = torch.where(can_replace_base)[0]
        final_can_replace = torch.zeros_like(next_tokens, dtype=torch.bool)

        if eligible_indices.numel() > 0:
            # Subset the generated sequences for pattern/correctness checks
            generated_ids_subset = generated_ids[eligible_indices]
            # Check patterns only for eligible sequences
            pattern_allowed_subset = self._check_replacement_patterns(
                generated_ids_subset
            )

            # Apply correctness check if callback exists, only where patterns allow
            if correctness_callback is not None:
                is_incorrect_subset = self._check_correctness(
                    generated_ids_subset,
                    pattern_allowed_subset,
                    correctness_callback,
                )
                # Final eligibility for the subset: pattern allows AND (is incorrect OR no callback)
                can_replace_subset = (
                    pattern_allowed_subset & is_incorrect_subset
                )
            else:
                # If no correctness callback, eligibility depends only on pattern check
                can_replace_subset = pattern_allowed_subset

            # Place results back into the full batch tensor
            final_can_replace[eligible_indices] = can_replace_subset

        return final_can_replace

    def _update_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        next_tokens: torch.Tensor,
        unfinished_sequences: torch.Tensor,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update the input sequences with newly generated tokens and handle finished sequences.

        Args:
            input_ids (torch.Tensor): Current token IDs of the generated sequence.
            attention_mask (torch.Tensor): Attention mask for the input sequence.
            next_tokens (torch.Tensor): Newly generated tokens to append.
            unfinished_sequences (torch.Tensor): Tensor indicating unfinished sequences.
            eos_token_id (Optional[int]): Token ID for end of sequence.
            pad_token_id (Optional[int]): Token ID for padding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated input_ids, attention_mask, unfinished_sequences.
        """
        assert input_ids.dim() == 2
        assert next_tokens.dim() == 1 and unfinished_sequences.dim() == 1
        assert (
            input_ids.size(0)
            == next_tokens.size(0)
            == unfinished_sequences.size(0)
        )

        this_token_is_eos = torch.zeros_like(next_tokens, dtype=torch.bool)
        if eos_token_id is not None:
            # Check for EOS before potentially padding
            this_token_is_eos = next_tokens == eos_token_id

            # Apply padding token ONLY to sequences that are already finished
            next_tokens = torch.where(
                unfinished_sequences.bool(),
                next_tokens,
                torch.tensor(
                    pad_token_id,
                    device=next_tokens.device,
                    dtype=next_tokens.dtype,
                ),
            )

        # Append the token (original or PAD)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        # Update attention mask: extend by 1 for sequences that are still unfinished
        new_mask_value = (
            unfinished_sequences.clone().long()
        )  # 1 if unfinished, 0 if finished
        attention_mask = torch.cat(
            [attention_mask, new_mask_value.unsqueeze(-1)], dim=-1
        )

        # Update unfinished sequences: Mark as finished if EOS was generated *this step*
        if eos_token_id is not None:
            unfinished_sequences = (
                unfinished_sequences.bool() & ~this_token_is_eos
            )

        return input_ids, attention_mask, unfinished_sequences

    def _uniform_sampling(
        self, logits: torch.Tensor, top_k: int
    ) -> torch.Tensor:
        """
        Uniformly sample from the top-k tokens.

        Args:
            logits (torch.Tensor): Logits of shape [batch_size, vocab_size].
            top_k (int): Number of top tokens to sample from.

        Returns:
            torch.Tensor: Sampled token indices of shape [batch_size].
        """
        batch_size, vocab_size = logits.shape
        # Ensure k is valid
        k = min(top_k, vocab_size)
        if k == 0:  # Handle edge case where top_k might be 0 after calculation
            k = 1

        top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)

        # If k=1, multinomial fails, just return the top index
        if k == 1:
            return top_k_indices.squeeze(-1)

        # Create uniform probabilities for the top-k tokens
        uniform_probs = torch.ones_like(top_k_values) / k

        # Sample indices from the uniform distribution over top-k
        sampled_relative_indices = torch.multinomial(
            uniform_probs, num_samples=1
        )  # Shape: [batch_size, 1]

        # Gather the actual token IDs using the sampled relative indices
        next_tokens = torch.gather(
            top_k_indices, dim=1, index=sampled_relative_indices
        ).squeeze(-1)

        return next_tokens

    def _sampling(
        self,
        token_logits: torch.Tensor,
        temperature: torch.Tensor,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Sample the next token from logits using temperature, top-k, and top-p filtering.

        Args:
            token_logits (torch.Tensor): Logits for next token, shape [batch_size, vocab_size].
            temperature (torch.Tensor): Temperature for scaling, shape [batch_size].
            top_p (float): Nucleus sampling threshold.
            top_k (int): Top-k sampling parameter.

        Returns:
            torch.Tensor: Sampled token IDs, shape [batch_size].
        """
        assert token_logits.dim() == 2
        assert token_logits.size(0) == temperature.size(0)

        batch_size, vocab_size = token_logits.shape

        # Handle zero temperature (greedy decoding) separately for clarity and efficiency
        zero_temp_mask = temperature == 0
        if zero_temp_mask.all():
            return token_logits.argmax(dim=-1)

        # Use unsqueezed temperature for broadcasting. Clamp to avoid division by zero.
        scaled_logits = torch.where(
            temperature.unsqueeze(1) > 0,
            token_logits / temperature.unsqueeze(1).clamp(min=1e-8),
            token_logits,  # Keep original logits if temp is zero (already handled, but safe)
        )

        if top_k > 0:
            k = min(top_k, vocab_size)
            # Get the values and indices of the top-k logits
            top_k_values, top_k_indices = torch.topk(scaled_logits, k=k, dim=-1)
            # Create a mask setting all logits to -inf initially
            min_vals = torch.full_like(scaled_logits, float('-inf'))
            # Use scatter to place the top-k values back into the correct positions
            scaled_logits = torch.scatter(
                min_vals, dim=-1, index=top_k_indices, src=top_k_values
            )

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                scaled_logits, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Create a mask for tokens to remove (those exceeding cumulative P)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to ensure the first element is always kept
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter the removal mask back to the original logit positions
            indices_to_remove = torch.scatter(
                sorted_indices_to_remove,
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove,
            )
            # Set logits of removed tokens to -inf
            scaled_logits = scaled_logits.masked_fill(
                indices_to_remove, float('-inf')
            )

        # Convert logits to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # If some sequences had temp=0, override their sampled token with the greedy choice
        if zero_temp_mask.any():
            greedy_tokens = token_logits.argmax(dim=-1)
            next_tokens = torch.where(
                zero_temp_mask, greedy_tokens, next_tokens
            )

        assert next_tokens.dim() == 1 and next_tokens.size(
            0
        ) == token_logits.size(0)
        return next_tokens

    def _apply_token_replacement(
        self,
        next_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        initial_seq_len: int,
        replacement_counts: torch.Tensor,
        replace_max_per_seq: int,
        explore_replace_prob: float,
        correctness_callback: Optional[Callable],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply token replacement if conditions are met.

        Args:
            next_tokens (torch.Tensor): Tokens to potentially replace.
            input_ids (torch.Tensor): Current token IDs of the generated sequence.
            initial_seq_len (int): Initial sequence length.
            replacement_counts (torch.Tensor): Count of replacements already made.
            replace_max_per_seq (int): Maximum number of replacements allowed.
            explore_replace_prob (float): Probability of token replacement.
            correctness_callback (Optional[Callable]): Function to evaluate correctness.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Modified tokens and updated replacement counts.
        """
        # Get the part of the sequence *actually generated* in this run
        # This is important for correctness/pattern checks on the generated part only
        if input_ids.shape[1] > initial_seq_len:
            generated_ids = input_ids[:, initial_seq_len:]
        else:
            # Handle case where no tokens have been generated yet (e.g., first step)
            generated_ids = torch.empty(
                (input_ids.shape[0], 0),
                dtype=torch.long,
                device=input_ids.device,
            )

        # Determine which sequences are eligible for replacement based on the *predicted* next_tokens
        can_replace = self._determine_replacement_eligibility(
            generated_ids,  # Pass only the generated part
            next_tokens,
            replacement_counts,
            replace_max_per_seq,
            correctness_callback,
        )

        modified_next_tokens = next_tokens  # Start with original tokens
        replace_mask = torch.zeros_like(can_replace)  # Initialize mask

        if can_replace.sum() > 0:
            # Apply probabilistic replacement where eligible
            modified_next_tokens, replace_mask = self._replace_special_tokens(
                next_tokens, can_replace, explore_replace_prob
            )
            # Update replacement counts where replacement actually occurred
            replacement_counts += replace_mask.long()

        return modified_next_tokens, replacement_counts

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """
        Applies repetition penalty to the logits in-place or returns modified logits.

        Args:
            logits (torch.Tensor): Logits to modify, shape [batch_size, vocab_size].
            input_ids (torch.Tensor): History of token ids, shape [batch_size, seq_len].
            penalty (float): The repetition penalty factor (penalizes logits of repeated tokens).

        Returns:
            torch.Tensor: Logits with repetition penalty applied.
        """
        if penalty == 1.0:
            return logits  # No penalty to apply

        batch_size = logits.shape[0]
        # Create a copy to avoid modifying the original tensor if passed by reference elsewhere,
        # although in this specific flow it might be okay to modify in-place.
        # Let's return a modified copy for safety/clarity.
        modified_logits = logits.clone()

        for i in range(batch_size):
            # Consider using only generated tokens for penalty if desired:
            # sequence_to_check = input_ids[i, self.initial_seq_len:] # Requires initial_seq_len access
            # Using the whole sequence is simpler here:
            sequence_to_check = input_ids[i]

            if sequence_to_check.numel() == 0:  # Skip if sequence is empty
                continue

            tokens_to_penalize = torch.unique(sequence_to_check)

            # Create masks for positive and negative logits for these tokens
            # Use the logits for the *current* batch item being processed
            current_logits_at_penalize_indices = modified_logits[
                i, tokens_to_penalize
            ]

            positive_mask = current_logits_at_penalize_indices > 0
            negative_mask = (
                ~positive_mask
            )  # Same as current_logits_at_penalize_indices <= 0

            # Apply penalty: Divide positive logits, Multiply negative logits
            # Get the actual token IDs corresponding to the masks
            positive_tokens = tokens_to_penalize[positive_mask]
            negative_tokens = tokens_to_penalize[negative_mask]

            # Apply penalty directly to the selected logits for this batch item
            if positive_tokens.numel() > 0:
                modified_logits[i, positive_tokens] /= penalty
            if negative_tokens.numel() > 0:
                modified_logits[i, negative_tokens] *= penalty

        return modified_logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: Union[torch.Tensor, float],
        pad_token_id: int,
        eos_token_id: int,
        repetition_penalty: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        max_new_tokens: int = 50,
        random_start_steps: int = 0,
        random_start_skip_n: int = 0,
        random_start_top_k: int = 20,
        explore_replace_prob: float = 0.0,
        replace_max_per_seq: int = 0,
        correctness_callback: Optional[Callable] = None,
        **kwargs,  # Keep **kwargs for potential future use or compatibility
    ) -> GenerateDecoderOnlyOutput:
        """
        Generate text with customizable sampling, exploration, special token replacement, and repetition penalty.

        Args:
            input_ids (torch.Tensor): Initial token IDs, shape [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask, shape [batch_size, seq_len].
            temperature (Union[torch.Tensor, float]): Temperature for sampling.
            pad_token_id (int): Token ID for padding.
            eos_token_id (int): Token ID for end of sequence.
            repetition_penalty (float): Penalty for repeating tokens (1.0 means no penalty). <<< ADDED
            top_p (float): Nucleus sampling threshold (default: 1.0).
            top_k (int): Top-k sampling parameter (default: 0).
            max_new_tokens (int): Max new tokens to generate (default: 50).
            random_start_steps (int): Steps for exploration (default: 0).
            random_start_skip_n (int): Steps to skip exploration (default: 0).
            random_start_top_k (int): Top-k for exploration (default: 20).
            explore_replace_prob (float): Probability of token replacement (default: 0.0).
            replace_max_per_seq (int): Max replacements per sequence (default: 0).
            correctness_callback (Optional[callable]): Function evaluating correctness (float 0.0-1.0).
            **kwargs: Additional arguments (unused).

        Returns:
            GenerateDecoderOnlyOutput: Generated sequences and potentially other info.
        """

        if kwargs.get('num_return_sequences', 1) > 1:
            raise ValueError(
                'Does not support generate multiple sequences, \
                             repeat the prompts before apply tokenization.'
            )

        batch_size = input_ids.shape[0]
        initial_seq_len = input_ids.size(1)
        device = input_ids.device

        generated_tokens = 0
        # Ensure unfinished_sequences is on the correct device and dtype
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
        replacement_counts = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )

        past_key_values = None

        if isinstance(temperature, (float, int)):
            temperature = torch.full(
                (batch_size,),
                float(temperature),
                device=device,
                dtype=torch.float32,
            )
        elif isinstance(temperature, list):
            temperature = torch.tensor(
                temperature, device=device, dtype=torch.float32
            )
        else:
            temperature = temperature.to(device=device, dtype=torch.float32)
        # Ensure temperature has the correct shape
        if temperature.dim() == 0:
            temperature = temperature.repeat(batch_size)
        assert (
            len(temperature) == batch_size
        ), 'Temperature must be the same size of batch_size'

        while generated_tokens < max_new_tokens:
            # Prepare model inputs
            model_inputs = {
                'input_ids': (
                    input_ids if past_key_values is None else input_ids[:, -1:]
                ),
                'attention_mask': attention_mask,
                'past_key_values': past_key_values,
                'use_cache': True,
            }

            outputs = self.model(**model_inputs)
            next_token_logits = outputs.logits[
                :, -1, :
            ].float()  # Shape: [batch_size, vocab_size]
            past_key_values = outputs.past_key_values

            next_token_logits = self._apply_repetition_penalty(
                logits=next_token_logits,
                input_ids=input_ids,
                penalty=repetition_penalty,
            )

            explore_now = False
            if random_start_steps > 0:
                is_after_skip = generated_tokens >= random_start_skip_n
                is_within_explore_window = generated_tokens < (
                    random_start_steps + random_start_skip_n
                )
                explore_now = is_after_skip and is_within_explore_window

            if explore_now:
                # Apply exploration sampling (e.g., uniform from top-k)
                effective_steps = max(0, generated_tokens - random_start_skip_n)
                # Decay random_start_top_k (optional, example)
                current_random_start_top_k = max(
                    2, int(random_start_top_k * (0.8**effective_steps))
                )  # Ensure k >= 2 for sampling
                next_tokens_candidates = self._uniform_sampling(
                    next_token_logits,  # Use potentially penalty-adjusted logits
                    top_k=current_random_start_top_k,
                )
            else:
                # Apply standard sampling (temp, top-k, top-p)
                next_tokens_candidates = self._sampling(
                    token_logits=next_token_logits,  # Use potentially penalty-adjusted logits
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

            # Check if conditions for replacement are met *before* updating sequences
            should_consider_replacement = (
                explore_replace_prob > 0.0
                and replace_max_per_seq > 0
                and replacement_counts.max()
                < replace_max_per_seq  # Optimization: skip if all sequences maxed out
                and generated_tokens
                > random_start_skip_n  # Optionally skip replacement during skip phase
                # and generated_tokens
                # > (random_start_steps + random_start_skip_n)
                # * 2  # Only replace after exploration
            )

            if should_consider_replacement:
                next_tokens, replacement_counts = self._apply_token_replacement(
                    next_tokens_candidates,  # Pass the sampled candidates
                    input_ids,
                    initial_seq_len,
                    replacement_counts,
                    replace_max_per_seq,
                    explore_replace_prob,
                    correctness_callback,
                )
            else:
                next_tokens = (
                    next_tokens_candidates  # Use the originally sampled tokens
                )

            input_ids, attention_mask, unfinished_sequences = (
                self._update_sequences(
                    input_ids,
                    attention_mask,
                    next_tokens,  # Use the final next_tokens (potentially replaced)
                    unfinished_sequences,
                    eos_token_id,
                    pad_token_id,
                )
            )

            # Check if all sequences are finished
            if (
                unfinished_sequences.max() == 0
            ):  # .max() == 0 means all are False (0)
                break

            generated_tokens += 1

        return GenerateDecoderOnlyOutput(sequences=input_ids)


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = HfExploreLLMGenerator(model, tokenizer, device)

    # Example batch input
    message = [
        {
            'role': 'user',
            'content': 'Calen originally had 5 more pencils than does Caleb, and Caleb has 3 less than twice as many pencils as does Candy.  If Calen lost 10 pencils, which left him with 10 pencils, then how many pencils does Candy have?',
        },
    ]

    group_size = 8
    batch_messages = [message] * group_size

    message_prompt = tokenizer.apply_chat_template(
        batch_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(message_prompt, return_tensors='pt', padding=True).to(
        device
    )

    # Example batch-specific temperatures
    temperatures = torch.linspace(
        0.0, 0.9, steps=group_size, dtype=torch_dtype
    ).to(device)

    # Generate text
    output = generator.generate(
        inputs.input_ids,
        inputs.attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperatures,
        max_new_tokens=512,
        top_p=1.0,
        top_k=50,
        random_start_steps=2,
        random_start_top_k=100,
        explore_replace_prob=0.2,
        replace_max_per_seq=2,
    )

    # Decode the output tokens back to text
    input_len = inputs.input_ids.shape[1]
    generated_texts = tokenizer.batch_decode(
        output.sequences[:, input_len:], skip_special_tokens=True
    )

    for i, text in enumerate(generated_texts):
        print(f"Generated [{i}]: '{text}'\n")
        print('\n\n--\n\n')
