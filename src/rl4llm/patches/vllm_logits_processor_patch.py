"""
Apply patch to the logits processor so we support:
1. process data on the batch level
2. passing additional data to the logits processor
"""

import inspect

import torch
import vllm.model_executor.layers.logits_processor as lp
from vllm.model_executor.sampling_metadata import SamplingMetadata


def new_apply_logits_processors_batch(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    Applies logits processors to a batch of logits.

    This version processes sequences in batches based on their group
    (which shares the same logits_processors list), assuming the
    processors themselves support batch inputs.

    Args:
        logits: Tensor of shape (total_sequences, vocab_size).
        sampling_metadata: Metadata containing sequence groups, sampling params,
                           and sequence data.

    Returns:
        The processed logits tensor.
    """
    processed_rows_count = 0
    for seq_idx, seq_group in enumerate(sampling_metadata.seq_groups):
        logits_processors = seq_group.sampling_params.logits_processors
        group_sample_indices = seq_group.sample_indices

        # Only process if there are processors AND sequences to sample in this group
        if logits_processors and group_sample_indices:
            # 1. Gather batch data for this group
            group_logits = logits[
                group_sample_indices
            ]  # Shape: (group_size, vocab_size)
            group_seq_ids = seq_group.seq_ids

            # Ensure seq_ids correspond to sample_indices if they can differ
            if len(group_seq_ids) != len(group_sample_indices):
                # This case needs clarification based on SamplingMetadata structure.
                # For now, assume direct correspondence:
                if len(group_seq_ids) < len(group_sample_indices):
                    raise ValueError(
                        'Mismatch between seq_ids and sample_indices count'
                    )
                # If seq_ids is longer, assume the first len(group_sample_indices) correspond
                relevant_seq_ids = group_seq_ids[: len(group_sample_indices)]
            else:
                relevant_seq_ids = group_seq_ids

            group_past_tokens_ids = [
                seq_group.seq_data[seq_id].output_token_ids
                for seq_id in relevant_seq_ids
            ]
            group_prompt_tokens_ids = [
                seq_group.seq_data[seq_id].prompt_token_ids
                for seq_id in relevant_seq_ids
            ]

            # 2. Apply processors sequentially to the batch
            # Processor MUST handle list of input_ids and batch of scores
            current_group_logits = group_logits
            for processor in logits_processors:
                if _ends_with_var_keyword(processor):
                    # Prepare batched kwargs
                    # Note: seq_idx is scalar (group index)
                    kwargs = {
                        'seq_idx': seq_idx,
                        'prompt_tokens_ids': group_prompt_tokens_ids,
                    }
                    current_group_logits = processor(
                        group_past_tokens_ids, current_group_logits, **kwargs
                    )
                else:
                    current_group_logits = processor(
                        group_past_tokens_ids, current_group_logits
                    )

            # 3. Update the original logits tensor
            logits[group_sample_indices] = current_group_logits

        # Keep track of all rows associated with this group (sampled & prompt)
        processed_rows_count += len(group_sample_indices) + len(
            seq_group.prompt_logprob_indices
        )

    # Verification: Check if all rows were accounted for (either processed or skipped)
    if processed_rows_count != logits.shape[0]:
        raise AssertionError(
            f"Mismatch in processed logit rows. Expected {logits.shape[0]}, "
            f"accounted for {processed_rows_count}."
        )

    return logits


def _ends_with_var_keyword(fn_to_check) -> bool:
    """Checks if a function has three parameters and the last one is '**kwargs'"""
    # Handle bound methods (like instance methods)
    if hasattr(fn_to_check, '__func__'):
        fn_to_check = fn_to_check.__func__
    # Handle callable objects
    elif not inspect.isfunction(fn_to_check) and hasattr(
        fn_to_check, '__call__'
    ):
        fn_to_check = fn_to_check.__call__
        if hasattr(fn_to_check, '__func__'):
            fn_to_check = fn_to_check.__func__

    try:
        signature = inspect.signature(fn_to_check)
        params = list(signature.parameters.values())

        # Adjust for 'self' or 'cls' if it's a method
        if params and params[0].name in ('self', 'cls'):
            params = params[1:]

        # Check if the remaining signature matches (input_ids, scores, **kwargs)
        return (
            len(params) == 3
            and params[-1].kind == inspect.Parameter.VAR_KEYWORD
        )
    except (
        ValueError
    ):  # Handle built-in functions or other non-introspectable callables
        return False


# patch the original code
lp._apply_logits_processors = new_apply_logits_processors_batch
