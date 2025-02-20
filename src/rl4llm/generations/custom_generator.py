import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

logger = logging.getLogger(__name__)


class CustomLLMGenerator:
    """
    A custom class for text generation using a language model (LLM).
    It supports batch-specific temperatures and exploration during sampling.
    """

    def __init__(self, model: PreTrainedModel):
        """
        Initialize the CustomLLMGenerator with a pretrained language model.

        Args:
            model (PreTrainedModel): A pre-trained transformer-based model for text generation.
        """
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
        """
        Update the input sequences with newly generated tokens and handle finished sequences.

        Args:
            input_ids (torch.Tensor): The current token IDs of the generated sequence.
            attention_mask (torch.Tensor): The attention mask for the input sequence.
            next_tokens (torch.Tensor): The newly generated tokens to append to the sequences.
            unfinished_sequences (torch.Tensor): A tensor indicating whether the sequences are unfinished.
            eos_token_id (Optional[int]): The token ID representing the end of the sequence (if any).
            pad_token_id (Optional[int]): The token ID used for padding (if any).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Updated input_ids (torch.Tensor): The input sequences with the newly appended tokens.
                - Updated attention_mask (torch.Tensor): The updated attention mask.
                - Updated unfinished_sequences (torch.Tensor): The tensor indicating unfinished sequences.
        """
        assert input_ids.dim() == 2
        assert (
            next_tokens.dim() == unfinished_sequences.dim() == 1
        ), f"Invalid shape: {next_tokens.shape}, {unfinished_sequences.shape}"
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
        """
        Sample the next token from the logits using temperature scaling, top-k filtering,
        and nucleus sampling. Supports exploration with specific parameters.

        Args:
            token_logits (torch.Tensor): The logits for the next token to be sampled, shape [batch_size, vocab_size].
            temperature (torch.Tensor): The temperature for scaling the logits, shape [batch_size].
            top_p (float): The cumulative probability threshold for nucleus sampling.
            top_k (int): The number of top-k candidates to consider for sampling.
            do_exploration (bool, optional): Whether to perform exploration (default: False).
            explore_top_k (int, optional): The number of top-k candidates to explore when exploration is enabled.
            explore_top_k_beta (float, optional): A scaling factor for the exploration probabilities.

        Returns:
            torch.Tensor: The sampled token IDs for the next step in the sequence.
        """
        assert token_logits.dim() == 2
        assert token_logits.size(0) == temperature.size(0)

        # Handle zero temperature case
        zero_temp_mask = temperature == 0
        if zero_temp_mask.all():
            return token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)

        # Handle temperature=0 cases
        greedy_tokens = None
        if zero_temp_mask.any():
            greedy_tokens = token_logits.argmax(dim=-1, keepdim=True).squeeze(-1)

        # Apply temperature scaling
        scaled_logits = token_logits / temperature.unsqueeze(1).clamp(min=1e-8)

        if do_exploration and explore_top_k > 1:
            # Get top-k values and indices for exploration
            top_k_values, top_k_indices = torch.topk(scaled_logits, k=explore_top_k, dim=-1)

            # Convert to probabilities and apply beta power
            probs = F.softmax(top_k_values, dim=-1)
            probs = probs.pow(explore_top_k_beta)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample from the modified distribution
            sampled_indices = torch.multinomial(probs, num_samples=1)
            next_tokens = torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)
        else:
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(scaled_logits, min(top_k, scaled_logits.shape[-1]), dim=-1)
                indices_to_remove = torch.ones_like(scaled_logits, dtype=torch.bool)
                indices_to_remove.scatter_(-1, top_k_indices, False)
                scaled_logits.masked_fill_(indices_to_remove, float('-inf'))

            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Create a mask for tokens to remove
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter the mask back to original indices
                indices_to_remove = torch.zeros_like(scaled_logits, dtype=torch.bool)
                for i in range(token_logits.size(0)):
                    indices_to_remove[i].scatter_(0, sorted_indices[i], sorted_indices_to_remove[i])

                scaled_logits.masked_fill_(indices_to_remove, float('-inf'))

            # Convert to probabilities and sample
            probs = F.softmax(scaled_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Replace tokens for temperature=0 cases
        if greedy_tokens is not None:
            assert greedy_tokens.shape == next_tokens.shape == zero_temp_mask.shape
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
        explore_top_k: int = 5,
        explore_top_k_beta: float = 0.5,
        **kwargs,
    ) -> GenerateDecoderOnlyOutput:
        """
        Generate text using a transformer model with customizable sampling techniques.
        Supports batch-specific temperature, exploration, top-k sampling, and nucleus sampling.

        Args:
            input_ids (torch.Tensor): The initial token IDs for the sequence generation.
            attention_mask (torch.Tensor): The attention mask for the input tokens.
            temperature (Union[torch.Tensor, float]): Temperature scaling for logits. Can be scalar or per-batch.
            pad_token_id (int): The token ID used for padding in the generated sequences.
            eos_token_id (int): The token ID representing the end of sequence.
            top_p (float, optional): Probability threshold for nucleus sampling (default: 1.0).
            top_k (int, optional): Number of top-k candidates to sample from (default: 50).
            max_new_tokens (int, optional): The maximum number of new tokens to generate (default: 50).
            explore_start_steps (int, optional): Number of initial steps to perform exploration (default: 0).
            explore_top_k (int, optional): Number of top-k candidates to consider during exploration (default: 5).
            explore_top_k_beta (float, optional): Scaling factor for exploration probabilities (default: 0.5).
            **kwargs: Additional keyword arguments for model inference (unused here).

        Returns:
            GenerateDecoderOnlyOutput: The generated sequences as a `GenerateDecoderOnlyOutput` object, containing the
            generated token IDs.
        """
        batch_size = input_ids.shape[0]
        generated_tokens = 0  # Track only the new tokens generated
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        past_key_values = None

        # Normalize temperature to a tensor of shape (batch_size,)
        if isinstance(temperature, (float, int)):
            temperature = torch.full((batch_size,), float(temperature), device=input_ids.device)
        elif isinstance(temperature, list):
            temperature = torch.tensor(temperature, device=input_ids.device)
        else:
            temperature = temperature.to(input_ids.device)

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

            # Determine if we should do exploration
            do_exploration = explore_start_steps > 0 and generated_tokens < explore_start_steps

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
