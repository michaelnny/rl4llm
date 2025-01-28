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

    @torch.no_grad()
    def generate_actions_for_rl(
        self,
        batch_states: List[List[ChatTurn]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: Optional[int] = 0,
        do_sample: Optional[bool] = True,
        exploring_steps: Optional[int] = 0,  # Kept for compatibility
    ) -> List[EnvAction]:
        device = self.model.device
        if self.model.training:
            self.model.eval()

        # Prepare batch messages and tokenize
        batch_messages = []
        for states in batch_states:
            batch_messages.append([{'role': t.role, 'content': t.content} for t in states])
        
        message_prompt = self.tokenizer.apply_chat_template(
            batch_messages, tokenize=False, add_generation_prompt=True
        )

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

        # Initialize generation variables
        generated_sequences = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        past_key_values = None
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Manual generation loop
        for _ in range(max_new_tokens):
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
            )

            outputs = self.model(**model_inputs, return_dict=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Sampling logic
            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                    min_values = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_values, 
                        torch.tensor(-float('inf'), device=device), 
                        next_token_logits
                    )

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(
                        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('inf'))

                # Sample next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Update active sequences
            eos_reached = (next_tokens == eos_token_id) & active
            active = active & ~eos_reached

            if not active.any():
                break

            # Replace inactive sequences with pad_token_id
            next_tokens = next_tokens * active.long() + pad_token_id * (~active).long()

            # Update sequences and attention mask
            generated_sequences = torch.cat([generated_sequences, next_tokens.unsqueeze(-1)], dim=-1)
            input_ids = next_tokens.unsqueeze(-1)
            current_attention_mask = torch.cat([
                current_attention_mask, 
                active.unsqueeze(-1).long()
            ], dim=1)

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