import inspect

import torch
import vllm.model_executor.layers.logits_processor as lp
from vllm.model_executor.sampling_metadata import SamplingMetadata

# def new_apply_logits_processors_batch(
#     logits: torch.Tensor,
#     sampling_metadata: SamplingMetadata,
# ) -> torch.Tensor:
#     """
#     Applies logits processors to a batch of logits.

#     This version processes sequences in batches based on their group
#     (which shares the same logits_processors list), assuming the
#     processors themselves support batch inputs.

#     Args:
#         logits: Tensor of shape (total_sequences, vocab_size).
#         sampling_metadata: Metadata containing sequence groups, sampling params,
#                            and sequence data.

#     Returns:
#         The processed logits tensor.
#     """

#     # get logits processors from the very first sequence
#     logits_processors = sampling_metadata.seq_groups[
#         0
#     ].sampling_params.logits_processors

#     # construct input token ids consists prompt + past generation
#     if logits_processors:
#         batch_input_token_ids = []
#         for seq_group in sampling_metadata.seq_groups:
#             seq_ids = seq_group.seq_ids
#             for seq_id, logits_row_idx in zip(
#                 seq_ids, seq_group.sample_indices
#             ):
#                 data = seq_group.seq_data[seq_id]
#                 token_ids = list(data.prompt_token_ids) + list(
#                     data.output_token_ids
#                 )
#                 batch_input_token_ids.append(
#                     torch.tensor(token_ids, dtype=torch.long)
#                 )

#         input_ids = torch.stack(batch_input_token_ids, dim=0).to(logits.device)
#         assert input_ids.shape[0] == logits.shape[0]
#         for processor in logits_processors:
#             logits = processor(input_ids, logits)

#     return logits


# # patch the original code
# lp._apply_logits_processors = new_apply_logits_processors_batch
