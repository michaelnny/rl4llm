"""Apply a group sampling technic by creating a linearly scaled temperature for the group"""

from array import array
from typing import List, Tuple

import torch
from vllm.model_executor.sampling_metadata import (
    _SAMPLING_EPS,
    VLLM_TOKEN_ID_ARRAY_TYPE,
    SamplingMetadata,
    SamplingTensors,
)

# Save original
original_from_sampling_metadata = SamplingTensors.from_sampling_metadata

# Gets the config from OS env variables
import os

MIN_TEMPERATURE = float(os.environ.get('VLLM_MIN_TEMPERATURE', 0.6))
MAX_TEMPERATURE = float(os.environ.get('VLLM_MAX_TEMPERATURE', 1.0))

assert MIN_TEMPERATURE >= 0.0
assert MIN_TEMPERATURE < MAX_TEMPERATURE <= 1.0


@classmethod
def grouped_sampling_from_sampling_metadata(
    cls,
    sampling_metadata: 'SamplingMetadata',
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple['SamplingTensors', bool, bool, bool]:
    prompt_tokens: List[array] = []
    output_tokens: List[array] = []
    top_ks: List[int] = []
    temperatures: List[float] = []
    top_ps: List[float] = []
    min_ps: List[float] = []
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    repetition_penalties: List[float] = []
    do_penalties = False
    do_top_p_top_k = False
    do_min_p = False

    assert sampling_metadata.seq_groups is not None
    num_seq_groups = len(sampling_metadata.seq_groups)

    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        r = sampling_params.repetition_penalty
        top_p = sampling_params.top_p
        min_p = sampling_params.min_p

        if MAX_TEMPERATURE == 0.0:
            temperature = 0.0  # for greedy sampling like during evaluation
        elif num_seq_groups > 1:
            # Group sampling: Linearly scale temperature from min_temperature to max_temperature over the sequence groups
            temperature = round(
                MIN_TEMPERATURE
                + (MAX_TEMPERATURE - MIN_TEMPERATURE)
                * (i / (num_seq_groups - 1)),
                4,
            )
        else:
            # If there's only one group, use the max temperature directly
            temperature = MAX_TEMPERATURE

        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        top_k = vocab_size if top_k == -1 else top_k
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0
        if not do_top_p_top_k and (
            top_p < 1.0 - _SAMPLING_EPS or top_k != vocab_size
        ):
            do_top_p_top_k = True
        if not do_min_p and min_p > _SAMPLING_EPS:
            do_min_p = True
        if not do_penalties and (
            abs(p) >= _SAMPLING_EPS
            or abs(f) >= _SAMPLING_EPS
            or abs(r - 1.0) >= _SAMPLING_EPS
        ):
            do_penalties = True

        is_prompt = seq_group.is_prompt
        if is_prompt and sampling_params.prompt_logprobs is not None:
            # For tokens in the prompt that we only need to get
            # their logprobs
            query_len = seq_group.query_len
            assert query_len is not None
            prefill_len = len(seq_group.prompt_logprob_indices)
            temperatures += [temperature] * prefill_len
            top_ps += [top_p] * prefill_len
            top_ks += [top_k] * prefill_len
            min_ps += [min_p] * prefill_len
            presence_penalties += [0] * prefill_len
            frequency_penalties += [0] * prefill_len
            repetition_penalties += [1] * prefill_len

        if seq_group.do_sample:
            sample_lens = len(seq_group.sample_indices)
            assert sample_lens >= len(seq_ids)
            temperatures += [temperature] * sample_lens
            top_ps += [top_p] * sample_lens
            top_ks += [top_k] * sample_lens
            min_ps += [min_p] * sample_lens
            presence_penalties += [p] * sample_lens
            frequency_penalties += [f] * sample_lens
            repetition_penalties += [r] * sample_lens

    if do_penalties:
        for seq_group in sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            sampling_params = seq_group.sampling_params
            if (
                seq_group.is_prompt
                and sampling_params.prompt_logprobs is not None
            ):
                prefill_len = len(seq_group.prompt_logprob_indices)
                prompt_tokens.extend(
                    array(VLLM_TOKEN_ID_ARRAY_TYPE) for _ in range(prefill_len)
                )
                output_tokens.extend(
                    array(VLLM_TOKEN_ID_ARRAY_TYPE) for _ in range(prefill_len)
                )
            if seq_group.do_sample:
                for seq_id in seq_ids:
                    seq_data = seq_group.seq_data[seq_id]
                    prompt_tokens.append(seq_data.prompt_token_ids_array)
                    output_tokens.append(seq_data.output_token_ids_array)

    sampling_tensors = SamplingTensors.from_lists(
        temperatures,
        top_ps,
        top_ks,
        min_ps,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        prompt_tokens,
        output_tokens,
        vocab_size,
        device,
        dtype,
    )
    return (sampling_tensors, do_penalties, do_top_p_top_k, do_min_p)


SamplingTensors.from_sampling_metadata = grouped_sampling_from_sampling_metadata
