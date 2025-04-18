import functools
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import LogitsProcessorList, PreTrainedModel

from rl4llm.core.base_env import BaseRewardFunction
from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.envs.llm_env import LocalLLMEnv
from rl4llm.envs.sgl_env import EnvState, InferenceEnv
from rl4llm.generation.hf_explore_processor import HfExploreLogitsProcessor
from rl4llm.generation.sgl_explore_procesor import SglExploreLogitProcessor

# from rl4llm.patches.sgl_patch_custom_sampler import apply_patch_explore_sampler


logger = logging.getLogger(__name__)


# --- Using SGLang inference client ---


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
        explore_decay: float,
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
        self.explore_decay = explore_decay
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
            explore_logit_processor = SglExploreLogitProcessor(
                explore_steps=self.explore_steps,
                skip_n=self.explore_skip_n,
                explore_top_k=self.explore_top_k,
                decay=self.explore_decay,
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

    def _should_retry(self, is_correct: bool, retry_attempts_left: int) -> bool:
        """Determine if an item should retry based on accuracy, retries left, and probability."""
        # Retry if: not correct AND has attempts left AND retry mechanism enabled AND probability check passes
        return (
            not is_correct
            and retry_attempts_left
            >= 0  # Allows the last retry when attempts == 0
            and self.continue_prob > 0.0
            and self.accuracy_fn
            is not None  # Ensure accuracy check is possible
            and self.continue_special_tokens  # Ensure tokens are available
            and random.random() < self.continue_prob
        )

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

        custom_params = [
            {'temperature': float(self.temperatures[i]), 'step': 0}
            for i in range(batch_size)
        ]

        # State tracking for each item in the batch
        # Stores the full accumulated completion text, including special tokens
        current_full_completions = [''] * batch_size
        retry_attempts_left = [self.continue_max_retry] * batch_size
        is_correct = [False] * batch_size
        # Stores the last raw output dict from the LLM for finish_reason check
        last_llm_outputs: List[Optional[Dict[str, Any]]] = [None] * batch_size
        # Indices of items still needing generation/retry
        active_indices = list(range(batch_size))

        current_pass = 0
        # Max passes = initial pass + max retries
        max_passes = 1 + self.continue_max_retry

        while active_indices and current_pass < max_passes:
            logger.debug(
                f"Starting Pass {current_pass + 1}/{max_passes}. "
                f"Active indices: {active_indices}"
            )

            # Prepare prompts for the active items in this pass
            prompts_for_pass = []
            # Keep track of original batch indices corresponding to prompts_for_pass
            indices_in_pass = list(active_indices)

            for original_idx in indices_in_pass:
                # Prompt = Original Prompt + Accumulated Completion (incl. special tokens)
                prompt = (
                    original_prompts[original_idx]
                    + current_full_completions[original_idx]
                )
                prompts_for_pass.append(prompt)

            if not prompts_for_pass:
                logger.warning(
                    'No prompts generated for active indices. Breaking loop.'
                )
                break

            # Generate completions for the active prompts
            # Apply exploration only on the first pass
            current_logit_processor = (
                logit_processors if current_pass == 0 else None
            )
            output_batch = llm.generate(
                prompts=prompts_for_pass,
                sampling_params=sampling_params,
                custom_logit_processor=current_logit_processor,
                custom_params=custom_params,
            )

            if len(output_batch) != len(prompts_for_pass):
                # Handle potential errors if LLM output size doesn't match input size
                logger.error(
                    f"LLM output batch size ({len(output_batch)}) mismatch with "
                    f"input prompt batch size ({len(prompts_for_pass)}) in pass {current_pass + 1}. "
                    f"Indices in pass: {indices_in_pass}. Active indices: {active_indices}."
                    'Skipping update for this pass.'
                )
                # Decide how to handle this: break, continue, skip updates?
                # For now, let's break to avoid index errors.
                break  # Or implement more robust error handling

            # Process results for each item generated in this pass
            next_active_indices = []
            for i, original_idx in enumerate(indices_in_pass):
                output = output_batch[i]
                # Newly generated text segment from this pass
                generated_text_segment = output.get('text', '')
                last_llm_outputs[original_idx] = (
                    output  # Store latest output info
                )

                # Append the *newly* generated text segment
                current_full_completions[original_idx] += generated_text_segment

                # Check accuracy using the *full* current completion if retry is possible
                item_is_correct = False
                if self.accuracy_fn and self.continue_prob > 0.0:
                    try:
                        # Ensure ground truth exists for the index
                        if original_idx >= len(ground_truths):
                            raise IndexError(
                                f"Ground truth index {original_idx} out of bounds."
                            )

                        accuracy_result = self.accuracy_fn(
                            [
                                current_full_completions[original_idx]
                            ],  # Pass as list
                            [ground_truths[original_idx]],  # Pass as list
                        )
                        # Ensure result format is as expected (e.g., list of floats)
                        if (
                            isinstance(accuracy_result, list)
                            and len(accuracy_result) == 1
                        ):
                            accuracy = accuracy_result[0]
                            item_is_correct = accuracy == 1.0
                        else:
                            logger.warning(
                                f"Unexpected accuracy result format for index {original_idx}: {accuracy_result}. Treating as incorrect."
                            )
                            item_is_correct = False

                        logger.debug(
                            f" Index {original_idx}: Pass {current_pass + 1}. Accuracy Check. Correct: {item_is_correct}. "
                            f"Full Completion: '{current_full_completions[original_idx]}'"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error during accuracy check for index {original_idx} in pass {current_pass + 1}: {e}. Treating as incorrect."
                        )
                        item_is_correct = False
                else:
                    # If no accuracy function or retry disabled, consider it "correct" to stop retries
                    item_is_correct = True
                    logger.debug(
                        f" Index {original_idx}: Pass {current_pass + 1}. No accuracy check needed/possible."
                    )

                is_correct[original_idx] = (
                    item_is_correct  # Update final correctness state
                )

                # Decide whether to retry this item
                should_retry_item = False
                if not item_is_correct:
                    # Decrement attempts *before* checking if retry is possible
                    retry_attempts_left[original_idx] -= 1
                    logger.debug(
                        f" Index {original_idx}: Incorrect. Retries left: {retry_attempts_left[original_idx]}"
                    )

                    if self._should_retry(
                        item_is_correct, retry_attempts_left[original_idx]
                    ):
                        should_retry_item = True
                        logger.debug(f" Index {original_idx}: Will retry.")
                    else:
                        logger.debug(
                            f" Index {original_idx}: Will not retry (max attempts, probability, or config)."
                        )
                else:
                    logger.debug(
                        f" Index {original_idx}: Correct. No retry needed."
                    )

                # If retrying, append a special token and keep it active
                if should_retry_item:
                    # *** THE KEY FIX: Append the special token to the stored completion ***
                    special_token = random.choice(self.continue_special_tokens)
                    current_full_completions[original_idx] += special_token
                    logger.debug(
                        f" Index {original_idx}: Appended special token '{special_token}'."
                    )
                    next_active_indices.append(original_idx)
                # Otherwise, the item is finished for this generation cycle

            # Update the list of active indices for the next pass
            active_indices = next_active_indices
            current_pass += 1

        logger.info(f"Generation loop finished after {current_pass} passes.")
        logger.debug(f"Final full completions: {current_full_completions}")

        # Finalize outputs using the accumulated completions (which now include special tokens)
        return self._finalize_outputs(
            current_full_completions, last_llm_outputs
        )


# --- Using local HF LLM ---


class ExploreLocalLLMEnv(LocalLLMEnv):
    """An extension of the standard LocalLLMEnv
    where we apply some custom logits processor to the generation process
    to encourage exploration."""

    def __init__(
        self,
        temperature: Union[List[float], torch.Tensor],
        explore_steps: int,
        explore_top_k: int,
        explore_skip_n: int,
        explore_decay: float,
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
        self.explore_skip_n = explore_skip_n
        self.explore_decay = explore_decay
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
        sampling_params: Dict,
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Generates completions using the LLM for the current state.

        Args:
            llm: The pre-trained language model.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens, do_sample).
            state: The current EnvState containing input_ids and attention_mask.
            **kwargs: Additional custom arguments.

        Returns:
            A tuple containing:
            - completion_texts: List of decoded completion strings (up to, but not including, EOS).
            - completion_tokens: List of completion token tensors (unpadded, includes EOS if present).
            - completion_lengths: List of actual lengths for each completion (includes EOS, excludes PAD).
        """

        input_ids = state.input_ids.to(llm.device)
        attention_mask = state.attention_mask.to(llm.device)
        gen_args_copy = sampling_params.copy()
        gen_args_copy.pop('input_ids', None)
        gen_args_copy.pop('attention_mask', None)
        gen_args_copy.pop('num_return_sequences', None)
        gen_args_copy['return_dict_in_generate'] = True

        # add explore logits processor
        explore_prob = kwargs.get('explore_probability', 0.0)

        if explore_prob > 0 and (random.random() < explore_prob):
            correctness_callback = None
            # Checks for outcome correctness using the accuracy function,
            # if applied, will only apply token replacement to sequences with incorrect outcome
            if self.accuracy_fn:
                correctness_callback = functools.partial(
                    self.accuracy_fn.__call__,
                    ground_truths=(
                        state.ground_truth[0]
                        if self.batch_size == 1
                        else state.ground_truth
                    ),
                )

            explore_logits_processor = HfExploreLogitsProcessor(
                initial_seq_len=input_ids.shape[1],
                tokenizer=self.tokenizer,
                group_size=self.group_size,
                temperature=self.temperature,
                explore_steps=self.explore_steps,
                explore_skip_n=self.explore_skip_n,
                explore_top_k=self.explore_top_k,
                explore_decay=self.explore_decay,
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
        start = state.input_ids.shape[1]
        texts, tokens_list, lengths = self._process_completions(
            output.sequences[:, start:]
        )

        return texts, tokens_list, lengths
