import functools
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

from rl4llm.constants import LOGGER_NAME
from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.envs.sgl_env import EnvState, InferenceEnv

logger = logging.getLogger(LOGGER_NAME)


class ExploreLogitProcessor(CustomLogitProcessor):
    """A simple logits processor to implement group temperature and exploring start for sampling."""

    def __init__(
        self,
        temperatures: torch.Tensor,
        explore_steps: int = 0,
        explore_skip_n: int = 0,
        explore_top_k: int = 20,
        explore_decay_rate: float = 0.9,
    ):
        """
        Initializes the ExploreLogitsProcessor.

        Args:
            temperatures: Temperature for logits scaling (tensor).
            group_size: The number of sequences in the original batch request.
            device: The torch device where tensors should be placed.
            explore_steps: Number of steps for exploration sampling.
            explore_skip_n: Number of initial steps to skip before exploration.
            explore_top_k: Top-k value for exploration sampling.
            explore_decay_rate: Decay rate for explore_top_k during exploration.
        """
        if not isinstance(temperatures, torch.Tensor):
            raise ValueError('temperature must be a tensor')
        if any(t < 0 for t in temperatures):
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

        self.temperatures = temperatures
        self.explore_steps: int = explore_steps
        self.explore_skip_n: int = explore_skip_n
        self.explore_top_k: int = explore_top_k
        self.explore_decay_rate: float = explore_decay_rate

        self.step_t: int = 0
        self._initialized = False

    def __call__(self, logits: torch.Tensor, custom_param_list) -> torch.Tensor:
        """Apply exploring random start and group temperature"""
        import torch

        assert logits.dim() == 2
        bsz, vocab_size = logits.shape

        device = logits.device

        # --- Exploration Logic ---
        is_explore = (
            self.explore_steps > 0
            and self.explore_skip_n
            <= self.step_t
            < self.explore_skip_n + self.explore_steps
        )
        if is_explore:
            effective_steps = self.step_t - self.explore_skip_n
            current_explore_top_k = max(
                2,
                int(
                    self.explore_top_k
                    * (self.explore_decay_rate**effective_steps)
                ),
            )
            k = min(current_explore_top_k, vocab_size)

            # Apply uniform sampling within the top-k for exploration
            _, top_k_indices = torch.topk(logits, k=k, dim=-1)
            logits.fill_(float('-inf'))
            logits.scatter_(dim=-1, index=top_k_indices, value=100.0)
        else:
            # --- Group Temperature Logic ---
            if not self._initialized:
                self.temperatures = self.temperatures.to(device)

            # Important, SGLang might dynamic batch the requests during the first few steps
            # so the input logits not necessary have the full batch
            curr_temps = self.temperatures[:bsz]  # Shape: (bsz,)

            # Apply temperature scaling where temp > 0
            non_zero_temp_mask = curr_temps > 0

            safe_temp = torch.clamp(curr_temps, min=1e-8)
            # Reshape temperature and mask for broadcasting with logits
            # (bsz,) -> (bsz, 1)
            safe_temp_reshaped = safe_temp.unsqueeze(1)
            non_zero_temp_mask_reshaped = non_zero_temp_mask.unsqueeze(1)
            scaled_logits = logits / safe_temp_reshaped

            logits = torch.where(
                non_zero_temp_mask_reshaped, scaled_logits, logits
            )

        self.step_t += 1
        return logits


class ExploreInferenceEnv(InferenceEnv):
    """An extension of the standard InferenceEnv
    where we apply some custom logits processor to the generation process
    to encourage exploration."""

    def __init__(
        self,
        temperatures: torch.Tensor,
        explore_steps: int,
        explore_top_k: int,
        explore_skip_n: int,
        explore_decay_rate: float,
        continue_special_tokens: List[str],
        continue_max_retry: int,
        continue_prob: float,
        **kwargs,
    ):

        super().__init__(**kwargs)
        if not isinstance(temperatures, torch.Tensor):
            raise ValueError('temperature must be a tensor')
        if any(t < 0 for t in temperatures):
            raise ValueError('temperature values cannot be negative')
        assert len(temperatures) >= 1
        if not isinstance(continue_prob, float) or not (
            0.0 <= continue_prob < 1.0
        ):
            raise ValueError(
                'continue_prob must be a float between (0.0, 1.0).'
            )

        self.temperatures = temperatures
        self.explore_steps = explore_steps
        self.explore_top_k = explore_top_k
        self.explore_skip_n = explore_skip_n
        self.explore_decay_rate = explore_decay_rate
        self.continue_special_tokens = continue_special_tokens
        self.continue_max_retry = continue_max_retry
        self.continue_prob = continue_prob

        self.accuracy_fn = None
        for fn in self.reward_functions:
            if fn.name == 'accuracy_reward':
                self.accuracy_fn = fn
                break

        if not self.continue_special_tokens:
            logger.warning('No special tokens provided for retry mechanism.')
        if not self.accuracy_fn:
            logger.warning(
                "No 'accuracy_reward' function found. Retry mechanism will not activate."
            )

    def _prepare_logits_processor(self, explore_prob: float) -> Optional[str]:
        """Creates the explore logits processor string if conditions are met."""
        # This logic is self-contained setup, so keeping it separate is reasonable.
        if explore_prob > 0 and random.random() < explore_prob:
            explore_logit_processor = ExploreLogitProcessor(
                temperatures=self.temperatures,
                explore_steps=self.explore_steps,
                explore_skip_n=self.explore_skip_n,
                explore_top_k=self.explore_top_k,
                explore_decay_rate=self.explore_decay_rate,
            )
            return explore_logit_processor.to_str()
        return None

    def _finalize_outputs(
        self,
        final_texts: List[str],
        last_outputs: List[Optional[Dict[str, Any]]],
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Tokenizes final texts, handles EOS token addition based on finish reason,
        and returns the required tuple format.

        Args:
            final_texts: The list of fully generated completion strings.
            last_outputs: List containing the last output dictionary from the LLM
                          for each item, or None if never generated (shouldn't happen here).

        Returns:
            Tuple containing:
            - List of final completion strings.
            - List of final completion token tensors (unpadded).
            - List of final completion lengths.
        """
        completion_ids_list = []
        completion_lengths = []
        batch_size = len(final_texts)

        for i in range(batch_size):
            text = final_texts[i]
            token_ids = self.tokenizer(
                text,
                padding=False,
                truncation=False,
                add_special_tokens=False,  # Usually False for completions
            )['input_ids']

            # Determine finish reason from the last relevant output for this item
            finish_reason_type = None
            if last_outputs[i]:  # Should always be populated by the end
                meta_info = last_outputs[i].get('meta_info', {})
                finish_reason = meta_info.get('finish_reason', {})
                finish_reason_type = finish_reason.get('type')

            # Append EOS if generation stopped naturally (not by length limit)
            # and EOS is not already the last token.
            if (
                finish_reason_type != 'length'
                and token_ids  # Check if list is not empty
                and token_ids[-1] != self.tokenizer.eos_token_id
            ):
                token_ids.append(self.tokenizer.eos_token_id)

            completion_ids_list.append(
                torch.tensor(token_ids, dtype=torch.long)
            )
            completion_lengths.append(len(token_ids))

        # The final texts are already assembled
        return final_texts, completion_ids_list, completion_lengths

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
            - completion_lengths: List of final lengths for each completion.
        """
        explore_prob = kwargs.get('explore_probability', 0.0)
        logit_processors = self._prepare_logits_processor(explore_prob)

        original_prompts = state.prompt
        ground_truths = state.ground_truth
        batch_size = len(original_prompts)

        # --- State Tracking ---
        # Stores the full accumulated text for each item
        current_completions = [''] * batch_size
        # Stores the last raw output dict from the LLM for each item
        last_llm_outputs: List[Optional[Dict[str, Any]]] = [None] * batch_size
        # Tracks remaining retry attempts for each item
        retry_attempts_left = [self.continue_max_retry] * batch_size
        # Indices of items that are still active (need generation or retry)
        active_indices = list(range(batch_size))
        # Whether an item is considered correct (stops retrying)
        is_correct = [False] * batch_size

        # --- Generation Loop ---
        current_pass = 0
        max_passes = 1 + self.continue_max_retry  # Initial pass + max retries

        while active_indices and current_pass < max_passes:
            prompts_for_pass: List[str] = []
            indices_in_pass: List[int] = []

            # --- 1. Prepare Prompts for the Current Pass ---
            for original_idx in active_indices:
                if current_pass == 0:
                    prompt_str = original_prompts[original_idx]
                else:
                    # Construct retry prompt: original_prompt + previous_completion + [special_token]
                    prompt_str = (
                        original_prompts[original_idx]
                        + current_completions[original_idx]
                    )
                    if self.continue_special_tokens:
                        special_token = random.choice(
                            self.continue_special_tokens
                        )
                        prompt_str += special_token
                        # Also append the special token to the stored completion
                        # so it's part of the text being evaluated for accuracy later
                        current_completions[original_idx] += special_token

                prompts_for_pass.append(prompt_str)
                indices_in_pass.append(original_idx)

            if (
                not prompts_for_pass
            ):  # Should not happen if active_indices is not empty
                logger.warning('No prompts generated for pass, breaking loop.')
                break

            # --- 2. Run Batched Generation ---
            logger.debug(
                f"Pass {current_pass}: Generating for {len(indices_in_pass)} items."
            )
            output_batch = llm.generate(
                prompts=prompts_for_pass,
                sampling_params=sampling_params,
                custom_logit_processor=(
                    logit_processors if current_pass == 0 else None
                ),
            )

            # --- 3. Process Results and Update State ---
            next_active_indices: List[int] = []
            needs_accuracy_check_indices: List[int] = []
            needs_accuracy_check_completions: List[str] = []
            needs_accuracy_check_gts: List[str] = []

            for i, original_idx in enumerate(indices_in_pass):
                output = output_batch[i]
                generated_text = output['text']

                # Append the *newly* generated text to the current completion
                current_completions[original_idx] += generated_text
                last_llm_outputs[original_idx] = output

                # If retry is disabled globally or for this item, it's done
                if self.continue_prob == 0.0 or not self.accuracy_fn:
                    is_correct[original_idx] = True
                    continue  # Move to next item in the batch

                # If retry is enabled, prepare for accuracy check
                needs_accuracy_check_indices.append(original_idx)
                needs_accuracy_check_completions.append(
                    current_completions[original_idx]
                )
                needs_accuracy_check_gts.append(ground_truths[original_idx])

            # --- 4. Perform Accuracy Check (Batched if possible) ---
            if needs_accuracy_check_indices:
                # Assume accuracy_fn takes lists and returns a list of floats where 1.0 means correct
                accuracy_results = self.accuracy_fn(
                    needs_accuracy_check_completions, needs_accuracy_check_gts
                )

                for i, original_idx in enumerate(needs_accuracy_check_indices):
                    if accuracy_results[i] == 1.0:
                        is_correct[original_idx] = True
                        logger.debug(f"Item {original_idx} marked as correct.")
                    else:
                        # Incorrect: Check if it should be retried
                        retry_attempts_left[original_idx] -= 1
                        should_retry = (
                            retry_attempts_left[original_idx] >= 0
                            and random.random() < self.continue_prob
                        )

                        if should_retry:
                            logger.debug(
                                f"Item {original_idx} incorrect, scheduling retry ({retry_attempts_left[original_idx]} left)."
                            )
                            next_active_indices.append(original_idx)
                        else:
                            logger.debug(
                                f"Item {original_idx} incorrect, but stopping (max retries or probability)."
                            )
                            # No more retries, it's finished (though incorrect)
                            is_correct[original_idx] = False

            # --- 5. Update Active Indices for Next Pass ---
            active_indices.clear()
            for idx in range(batch_size):
                was_checked = idx in needs_accuracy_check_indices
                is_finished = False
                if was_checked:
                    # If checked, it's finished if it's correct OR it wasn't added to next_active_indices
                    is_finished = is_correct[idx] or (
                        idx not in next_active_indices
                    )
                else:
                    is_finished = (
                        self.continue_prob == 0.0 or not self.accuracy_fn
                    ) or is_correct[idx]

                # If the item is NOT finished, add it to the list for the next pass
                if not is_finished:
                    if retry_attempts_left[idx] >= 0:
                        active_indices.append(idx)
                    else:
                        logger.warning(
                            f"Item {idx} detected as active but has no retries left. Forcing finish."
                        )

            current_pass += 1
            logger.debug(
                f"End of Pass {current_pass - 1}. Active items for next pass: {active_indices}"
            )

            # Safety break if state becomes inconsistent
            if current_pass >= max_passes and active_indices:
                logger.debug(
                    f"Reached max passes ({max_passes}) but items {active_indices} still active. Forcing termination."
                )
                break

        # --- 6. Finalize Outputs ---
        # Use the accumulated texts and the last recorded LLM output for each item
        return self._finalize_outputs(current_completions, last_llm_outputs)
