"""A simple LLM generation code with custom decoding and KV caching."""

import logging
from typing import Any, Dict, List, Union, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.types import DecodingConfig, TokenUsage, ChatTurn, EnvAction, EnvState

logger = logging.getLogger(__name__)


class LLMGenerator:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_actions_for_rl(self, batch_states: List[List[ChatTurn]], **kwargs) -> List[EnvAction]:
        device = self.model.device
        if self.model.training:
            self.model.eval()

        # Prepare batch messages and tokenize
        batch_messages = []
        for states in batch_states:
            batch_messages.append([{'role': t.role, 'content': t.content} for t in states])

        message_prompt = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(
            message_prompt,
            return_tensors='pt',
            truncation=True,
            padding=True,
            padding_side='left',
            max_length=self.tokenizer.model_max_length,
        ).to(device)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        batch_size, seq_length = input_ids.shape
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Call the core generation method
        temperature = kwargs.get('temperature', 1.0)
        generated_sequences = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get('max_new_tokens', 512),
            temperature=temperature,
            top_p=kwargs.get('top_p', 0.95),
            top_k=kwargs.get('top_k', 0),
            do_sample=kwargs.get('do_sample', True),
        )

        # Extract completions and calculate token usage
        prompt_length = seq_length
        batch_completion_ids = generated_sequences[:, prompt_length:]

        prompt_tokens_count = (inputs['input_ids'] != pad_token_id).sum(dim=1).cpu().tolist()
        completion_tokens_count = (batch_completion_ids != pad_token_id).sum(dim=1).cpu().tolist()

        # Create response objects
        rollouts = []
        for i in range(batch_size):
            completion_ids = batch_completion_ids[i].tolist()
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            rollouts.append(
                EnvAction(
                    text=completion_text,
                    temperature=temperature,
                    usage=TokenUsage(
                        prompt_tokens=prompt_tokens_count[i],
                        completion_tokens=completion_tokens_count[i],
                        total_tokens=prompt_tokens_count[i] + completion_tokens_count[i],
                    ),
                )
            )

        return rollouts

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """Core generation method with manual sampling and KV caching."""
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

        return generated_sequences

    def _sample_next_token(
        self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int, do_sample: bool
    ) -> torch.Tensor:
        """Apply temperature, top-p/k filtering and sampling."""
        if do_sample:
            if temperature != 1.0:
                logits = logits / temperature

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
