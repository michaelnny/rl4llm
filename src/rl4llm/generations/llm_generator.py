"""A simple LLM generation code."""

import logging
from typing import Any, Dict, List, Union, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.types import DecodingConfig, TokenUsage, ChatTurn, EnvAction, EnvState

logger = logging.getLogger()


class LLMGenerator:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Ensure pad_token_id is set
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
        exploring_steps: Optional[int] = 0,  # kept for compatibility
    ) -> List[EnvAction]:
        """
        Generate multiple completions for a given input using a specific decoding strategy.

        Args:
            batch_states (List[List[ChatTurn]]): List of dialog messages to generate completions for.

        Returns:
            List of EnvAction dictionaries with the generated answer and other statistics.
        """
        device = self.model.device
        if self.model.training:
            self.model.eval()

        batch_messages = []
        for states in batch_states:
            batch_messages.append([{'role': t.role, 'content': t.content} for t in states])

        # Format chat messages
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

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'eos_token_id': eos_token_id,
            'pad_token_id': pad_token_id,
            'use_cache': True,
            'output_scores': True,
            'output_logits': True,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': do_sample,
        }

        # Generate in a single batched call
        outputs = self.model.generate(**gen_kwargs)
        generated_sequences = outputs.sequences
        prompt_length = input_ids.size(1)
        batch_size = generated_sequences.size(0)

        # Get the completion part (excluding prompt)
        batch_completion_ids = generated_sequences[:, prompt_length:]

        prompt_tokens_count = (input_ids != pad_token_id).sum(dim=1).cpu().tolist()
        completion_tokens_count = (batch_completion_ids != pad_token_id).sum(dim=1).cpu().tolist()

        rollouts = []
        for i in range(batch_size):
            completion_ids = batch_completion_ids[i].tolist()
            # Decode text without special tokens
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
