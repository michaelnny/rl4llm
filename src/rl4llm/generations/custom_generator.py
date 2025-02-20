import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

logger = logging.getLogger(__name__)


class CustomLLMGenerator:
    """
    A custom class for LLM text generation with batch-specific temperatures exploring start.
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model

    def _update_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        next_tokens: torch.Tensor,
        unfinished_sequences: torch.Tensor,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update sequences with new tokens and handle finished sequences."""
        assert input_ids.dim() == 2
        assert next_tokens.dim() == unfinished_sequences.dim() == 1
        assert input_ids.size(0) == next_tokens.size(0) == unfinished_sequences.size(0)

        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id or 0) * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id))

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)

        return input_ids, attention_mask, unfinished_sequences

    def _sample_next_tokens(
        self,
        token_logits: torch.Tensor,
        temperature: torch.Tensor,
        top_p: float,
        top_k: int,
        do_exploration: bool = False,
        explore_top_k: int = 0,
        explore_top_k_beta: float = 0.5,
    ) -> torch.Tensor:
        """Optimized batch sampling of next tokens."""
        assert token_logits.size(0) == temperature.size(0)

        # Handle zero temperature case first to match old behavior exactly
        zero_temp_mask = temperature == 0
        if zero_temp_mask.all():
            return token_logits.argmax(dim=-1, keepdim=True)

        if do_exploration and explore_top_k > 1:
            top_k_values, top_k_indices = torch.topk(token_logits, k=explore_top_k, dim=-1)
            probs = F.softmax(top_k_values, dim=-1)
            probs = probs.pow(explore_top_k_beta)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            sampled_indices = torch.multinomial(probs, num_samples=1)
            tokens = torch.gather(top_k_indices, -1, sampled_indices)
        else:
            # Regular sampling path
            scaled_logits = token_logits.clone()  # Clone to avoid modifying input
            scaled_logits[~zero_temp_mask] = scaled_logits[~zero_temp_mask] / temperature[~zero_temp_mask].unsqueeze(-1)

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)), dim=-1)
                indices_to_remove = torch.ones_like(scaled_logits, dtype=torch.bool)
                indices_to_remove.scatter_(-1, top_k_indices, False)
                scaled_logits.masked_fill_(indices_to_remove, float('-inf'))

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                scaled_logits.masked_fill_(indices_to_remove, float('-inf'))

            probs = F.softmax(scaled_logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1)

        # Handle zero temperature cases
        if zero_temp_mask.any():
            argmax_tokens = token_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.where(zero_temp_mask.unsqueeze(-1), argmax_tokens, tokens)

        return tokens.squeeze(-1)  # [batch_size]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: Union[torch.Tensor, float],
        pad_token_id: int,
        eos_token_id: int,
        top_p: float = 1.0,
        top_k: int = 50,
        max_new_tokens: int = 50,
        enable_exploration: bool = False,
        explore_start_steps: int = 0,
        explore_top_k: int = 5,
        explore_top_k_beta: float = 0.5,
        **kwargs,
    ) -> GenerateDecoderOnlyOutput:
        """Generate text with batch-specific temperatures."""
        batch_size = input_ids.shape[0]
        generated_tokens = 0  # Track only the new tokens generated
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        past_key_values = None

        while generated_tokens < max_new_tokens:
            # Get next token logits
            outputs = self.model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits = outputs.logits[:, -1, :].float()
            past_key_values = outputs.past_key_values

            do_exploration = False
            if enable_exploration:
                # Random Start Exploration
                if explore_start_steps is not None and explore_start_steps > 0 and generated_tokens < explore_start_steps:
                    do_exploration = True

            # Sample next tokens
            next_tokens = self._sample_next_tokens(
                token_logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_exploration=do_exploration,
                explore_top_k=explore_top_k,
                explore_top_k_beta=explore_top_k_beta,
            )
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

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = CustomLLMGenerator(model, device)

    # Example batch input
    message = [
        [
            {
                'role': 'user',
                'content': 'Data: Monthly Sales (Jan: $20k, Feb: $25k, Mar: $30k). Suggest a concise and impactful title for a bar chart representing this sales data.',
            },
        ],
        [
            {
                'role': 'user',
                'content': "I have a line chart showing website user engagement metrics: 'Bounce Rate' decreased from 60% to 45% over the last quarter, 'Average Session Duration' increased by 30 seconds, and 'Pages per Visit' remained stable.  What's the main positive conclusion from this chart?",
            },
        ],
        [
            {
                'role': 'user',
                'content': "A pie chart breaks down marketing spend: 40% on 'Social Media Ads', 30% on 'Search Engine Marketing', 20% on 'Email Campaigns', and the rest on 'Content Marketing'. Calculate the percentage allocated to 'Content Marketing'.",
            },
        ],
    ]
    message_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(message_prompt, return_tensors='pt', padding=True).to(device)

    # Example batch-specific temperatures
    temperatures = torch.tensor([0.0, 0.3, 0.5], dtype=torch.float16).to(device)

    # Generate text
    output = generator.generate(
        inputs.input_ids, inputs.attention_mask, temperature=temperatures, max_new_tokens=256, top_p=1.0
    )

    # Decode the output tokens back to text
    input_len = inputs.input_ids.shape[1]
    generated_texts = tokenizer.batch_decode(output.sequences[:, input_len:], skip_special_tokens=True)

    # print("Generated Texts:")
    for i, text in enumerate(generated_texts):
        print(f"Input: {message[i][0]['content']}")
        print(f"Generated: '{text}'\n")
        print('\n\n--\n\n')
