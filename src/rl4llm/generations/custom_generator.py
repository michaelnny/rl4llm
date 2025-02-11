"""A simple LLM generation code with custom decoding and KV caching."""

import logging
from typing import Any, Dict, List, Union, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput


logger = logging.getLogger(__name__)


class CustomLLMGenerator:
    """Custom text generation for LLM"""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: torch.Tensor,
        max_new_tokens: Optional[int] = 4096,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 1.0,
        top_k: Optional[int] = 0,
    ) -> GenerateDecoderOnlyOutput:
        """Core generation method with manual sampling (group temperatures) and KV caching.

        Args:
            input_ids (torch.Tensor): a 2D tensor contains prompt tokens.
            attention_mask (torch.Tensor): a bool tensor for attention tokens.
            temperature (torch.Tensor): a 1D tensor contains a group of temperatures for the group generation, where each item could have different temperature.
            max_new_tokens (Optional[int]): maximum number of tokens to generate, default 4096.
            do_sample (Optional[bool]): sampling tokens, default on.
            top_p (Optional[float]): sampling top p, default 1.0.
            top_k (Optional[int]): sampling top p, default 0.

        Returns:
            GenerateDecoderOnlyOutput: contains the generation sequence (prompt + generated)

        """
        assert temperature.dim() == 1
        assert input_ids.size(0) == temperature.size(0)

        device = self.model.device
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        generated_sequences = input_ids.clone()
        active = torch.ones(input_ids.size(0), dtype=torch.bool, device=device)
        past_key_values = None

        for _ in range(max_new_tokens):
            # Prepare model inputs using cached KV values
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask
            )

            # Forward pass with cached computation
            outputs = self.model(**model_inputs, return_dict=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Apply sampling parameters
            next_token = self._sample_next_token(
                logits=next_token_logits, temperature=temperature, top_p=top_p, top_k=top_k, do_sample=do_sample
            )

            # Update sequence status
            active, generated_sequences, input_ids, attention_mask = self._update_generation_state(
                next_token=next_token,
                active=active,
                generated_sequences=generated_sequences,
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

            if not active.any():
                break

        return GenerateDecoderOnlyOutput(sequences=generated_sequences)

    def _sample_next_token(
        self, logits: torch.Tensor, temperature: torch.Tensor, top_p: float, top_k: int, do_sample: bool
    ) -> torch.Tensor:
        """Apply temperature, top-p/k filtering and sampling."""
        if do_sample:
            temperature = temperature.to(logits.device)
            logits = logits / temperature.unsqueeze(-1)

            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                min_values = top_k_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.tensor(-float('inf'), device=logits.device), logits)

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float('inf'))

            probs = torch.nn.functional.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            return torch.argmax(logits, dim=-1)

    def _update_generation_state(
        self,
        next_token: torch.Tensor,
        active: torch.Tensor,
        generated_sequences: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
    ) -> tuple:
        """Update generation state and manage sequence completion."""
        # Check for EOS tokens
        eos_reached = (next_token == eos_token_id) & active
        active = active & ~eos_reached

        # Replace inactive sequences with pad tokens
        next_token = next_token * active.long() + pad_token_id * (~active).long()

        # Update sequences and attention mask
        generated_sequences = torch.cat([generated_sequences, next_token.unsqueeze(-1)], dim=-1)
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, active.unsqueeze(-1).long()], dim=1)

        return active, generated_sequences, input_ids, attention_mask
