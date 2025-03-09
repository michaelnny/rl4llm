import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

logger = logging.getLogger(__name__)


class CustomLLMGenerator:
    """
    A custom class for text generation using a language model (LLM).
    Supports batch-specific temperatures, simplified exploration, and special token replacement.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        source_tokens: Optional[List[int]] = None,
        target_token: Optional[int] = None,
        special_patterns: Optional[List[List[int]]] = None,
    ):
        """
        Initialize the CustomLLMGenerator with a pretrained language model.

        Args:
            model (PreTrainedModel): A pre-trained transformer-based model for text generation.
            source_tokens (List[int]): List of token IDs to replace, e.g., "EOS" or "</think>".
            target_token (int): Token ID to replace source tokens with, e.g., "Wait".
            special_patterns (List[List[int]]): List of token sequences that, if present, prevent replacement,
                                               e.g., [[27, 9217, 29]].
        """
        self.model = model
        self.source_tokens = source_tokens or []
        self.target_token = target_token
        self.special_patterns = special_patterns or []

    def _has_pattern(self, input_ids: torch.Tensor, pattern: List[int]) -> torch.Tensor:
        """
        Check if the given pattern exists in the input sequences for each batch.

        Args:
            input_ids (torch.Tensor): Current token IDs of shape [batch_size, seq_len].
            pattern (List[int]): The pattern to check for, e.g., [27, 9217, 29].

        Returns:
            torch.Tensor: Boolean tensor of shape [batch_size] indicating if the pattern is present.
        """
        batch_size, seq_len = input_ids.shape
        pattern_tensor = torch.tensor(pattern, device=input_ids.device)
        pattern_len = pattern_tensor.size(0)

        if seq_len < pattern_len:
            return torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Create sliding windows to check for the pattern
        windows = torch.as_strided(
            input_ids,
            size=(batch_size, seq_len - pattern_len + 1, pattern_len),
            stride=(input_ids.stride(0), input_ids.stride(1), input_ids.stride(1)),
        )
        matches = (windows == pattern_tensor).all(dim=2)
        return matches.any(dim=1)

    def _replace_special_tokens(
        self, next_tokens: torch.Tensor, can_replace: torch.Tensor, replace_prob: float = 0.2
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
        if not self.source_tokens or not self.target_token or replace_prob <= 0.0:
            return next_tokens

        batch_size = next_tokens.size(0)
        device = next_tokens.device
        modified_tokens = next_tokens.clone()

        # Check if next_tokens are in source_tokens
        source_tokens_tensor = torch.tensor(self.source_tokens, device=device)
        mask = (next_tokens.unsqueeze(-1) == source_tokens_tensor).any(dim=-1)

        # Combine conditions: token is a source token, replacement is allowed, and probability check
        replace_mask = mask & can_replace & (torch.rand(batch_size, device=device) < replace_prob)
        modified_tokens = torch.where(replace_mask, torch.full_like(next_tokens, self.target_token), modified_tokens)

        return modified_tokens, replace_mask

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
        assert next_tokens.dim() == unfinished_sequences.dim() == 1
        assert input_ids.size(0) == next_tokens.size(0) == unfinished_sequences.size(0)

        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id or 0) * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id))

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)

        return input_ids, attention_mask, unfinished_sequences

    def _uniform_top_k_sampling(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Uniformly sample from the top-k tokens.

        Args:
            logits (torch.Tensor): Logits of shape [batch_size, vocab_size].
            top_k (int): Number of top tokens to sample from.

        Returns:
            torch.Tensor: Sampled token indices of shape [batch_size].
        """
        assert top_k > 0
        batch_size, vocab_size = logits.shape
        k = min(top_k, vocab_size)

        top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)
        uniform_probs = torch.ones_like(top_k_values) / k
        sampled_indices = torch.multinomial(uniform_probs, num_samples=1)
        next_tokens = torch.gather(top_k_indices, dim=1, index=sampled_indices).squeeze(-1)

        return next_tokens

    def _sample_next_batch_tokens(
        self,
        token_logits: torch.Tensor,
        temperature: torch.Tensor,
        top_p: float,
        top_k: int,
        do_exploration: bool = False,
        explore_top_k: int = 0,
    ) -> torch.Tensor:
        """
        Sample the next token from logits using temperature, top-k, and top-p filtering.

        Args:
            token_logits (torch.Tensor): Logits for next token, shape [batch_size, vocab_size].
            temperature (torch.Tensor): Temperature for scaling, shape [batch_size].
            top_p (float): Nucleus sampling threshold.
            top_k (int): Top-k sampling parameter.
            do_exploration (bool): Whether to perform exploration.
            explore_top_k (int): Top-k for exploration sampling.

        Returns:
            torch.Tensor: Sampled token IDs, shape [batch_size].
        """
        assert token_logits.dim() == 2
        assert token_logits.size(0) == temperature.size(0)

        batch_size, vocab_size = token_logits.shape

        if do_exploration:
            return self._uniform_top_k_sampling(token_logits, top_k=min(16, explore_top_k))

        zero_temp_mask = temperature == 0
        if zero_temp_mask.all():
            return token_logits.argmax(dim=-1)

        scaled_logits = torch.where(
            temperature.unsqueeze(1) > 0, token_logits / temperature.unsqueeze(1).clamp(min=1e-8), token_logits
        )

        if top_k > 0:
            k = min(top_k, vocab_size)
            top_k_values, top_k_indices = torch.topk(scaled_logits, k=k, dim=-1)
            scaled_logits = torch.full_like(scaled_logits, float('-inf'))
            scaled_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            mask = cumulative_probs <= top_p
            mask = torch.cat([torch.ones_like(mask[:, :1], dtype=torch.bool), mask[:, 1:]], dim=1)
            sorted_logits = torch.where(mask, sorted_logits, torch.full_like(sorted_logits, float('-inf')))
            scaled_logits = torch.full_like(scaled_logits, float('-inf'))
            scaled_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = F.softmax(scaled_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        if zero_temp_mask.any():
            greedy_tokens = token_logits.argmax(dim=-1)
            next_tokens = torch.where(zero_temp_mask, greedy_tokens, next_tokens)

        assert next_tokens.dim() == 1 and next_tokens.size(0) == token_logits.size(0)
        return next_tokens

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: Union[torch.Tensor, float],
        pad_token_id: int,
        eos_token_id: int,
        top_p: float = 1.0,
        top_k: int = 0,
        max_new_tokens: int = 50,
        explore_start_steps: int = 0,
        explore_skip_n: int = 0,
        explore_top_k: int = 100,
        explore_replace_prob: float = 0.0,
        **kwargs,
    ) -> GenerateDecoderOnlyOutput:
        """
        Generate text with customizable sampling, exploration, and special token replacement.

        Args:
            input_ids (torch.Tensor): Initial token IDs, shape [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask, shape [batch_size, seq_len].
            temperature (Union[torch.Tensor, float]): Temperature for sampling.
            pad_token_id (int): Token ID for padding.
            eos_token_id (int): Token ID for end of sequence.
            top_p (float): Nucleus sampling threshold (default: 1.0).
            top_k (int): Top-k sampling parameter (default: 0).
            max_new_tokens (int): Max new tokens to generate (default: 50).
            explore_start_steps (int): Steps for exploration (default: 0).
            explore_skip_n (int): Steps to skip exploration (default: 0).
            explore_top_k (int): Top-k for exploration (default: 100).
            explore_replace_prob (float): Probability of token replacement (default: 0.0).
            **kwargs: Additional arguments (unused).

        Returns:
            GenerateDecoderOnlyOutput: Generated sequences.
        """
        batch_size = input_ids.shape[0]
        initial_seq_len = input_ids.size(1)
        generated_tokens = 0
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        has_replaced = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        past_key_values = None
        current_explore_top_k = explore_top_k

        # Normalize temperature
        if isinstance(temperature, (float, int)):
            temperature = torch.full((batch_size,), float(temperature), device=input_ids.device)
        elif isinstance(temperature, list):
            temperature = torch.tensor(temperature, device=input_ids.device)
        else:
            temperature = temperature.to(input_ids.device)

        while generated_tokens < max_new_tokens:
            outputs = self.model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :].float()
            past_key_values = outputs.past_key_values

            # Exploration logic
            do_exploration = explore_start_steps > 0 and (generated_tokens - explore_skip_n) < explore_start_steps
            if explore_skip_n and generated_tokens < explore_skip_n:
                do_exploration = False

            if do_exploration:
                effective_steps = max(0, generated_tokens - explore_skip_n)
                current_explore_top_k = max(10, explore_top_k - effective_steps * 10)

            # Sample next tokens
            next_tokens = self._sample_next_batch_tokens(
                token_logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_exploration=do_exploration,
                explore_top_k=current_explore_top_k,
            )

            # Handle special token replacement, example: '</think>' to "Wait "
            if explore_replace_prob > 0:
                # Check for special patterns and determine if replacement is allowed
                if self.special_patterns:
                    generated_ids = input_ids[:, initial_seq_len:]
                    pattern_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
                    for pattern in self.special_patterns:
                        pattern_found |= self._has_pattern(generated_ids, pattern)
                    can_replace = ~pattern_found
                else:
                    can_replace = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
                # Only allow replacement if no replacement has been made yet
                can_replace = can_replace & ~has_replaced

                next_tokens, replace_mask = self._replace_special_tokens(next_tokens, can_replace, explore_replace_prob)
                # Update the replacement status for sequences where replacement occurred
                has_replaced |= replace_mask

            # Update sequences
            input_ids, attention_mask, unfinished_sequences = self._update_sequences(
                input_ids,
                attention_mask,
                next_tokens,
                unfinished_sequences,
                eos_token_id,
                pad_token_id,
            )

            if unfinished_sequences.max() == 0:
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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = CustomLLMGenerator(model)

    # Example batch input
    message = [
        {
            'role': 'user',
            'content': 'Calen originally had 5 more pencils than does Caleb, and Caleb has 3 less than twice as many pencils as does Candy.  If Calen lost 10 pencils, which left him with 10 pencils, then how many pencils does Candy have?',
        },
    ]

    group_size = 16
    batch_messages = [message] * group_size

    message_prompt = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(message_prompt, return_tensors='pt', padding=True).to(device)

    # Example batch-specific temperatures
    temperatures = torch.linspace(0.0, 0.9, steps=group_size, dtype=torch_dtype).to(device)

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
        explore_start_steps=50,
        explore_top_k=100,
        explore_replace_prob=0.4,
    )

    # Decode the output tokens back to text
    input_len = inputs.input_ids.shape[1]
    generated_texts = tokenizer.batch_decode(output.sequences[:, input_len:], skip_special_tokens=True)

    # print("Generated Texts:")
    for i, text in enumerate(generated_texts):
        print(f"Generated [{i}]: '{text}'\n")
        print('\n\n--\n\n')
