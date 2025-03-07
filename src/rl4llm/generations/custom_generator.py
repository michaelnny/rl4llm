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

    def _inverted_top_k_sampling(self, logits: torch.Tensor, top_k: int = 50, explore_beta: float = 0.7) -> torch.Tensor:
        """
        Top-k sampling with inverted probability distribution to promote diversity.

        Args:
            logits: Original logits [batch_size, vocab_size]
            top_k: Number of top tokens to consider
            explore_beta: Controls how much to favor less likely tokens (0.0 to 1.0)
                            0.0 = original distribution, 1.0 = fully inverted distribution

        Returns:
            Sampled token indices
        """
        assert top_k > 1
        assert 0.0 <= explore_beta <= 1.0
        batch_size, vocab_size = logits.shape
        k = min(top_k, vocab_size)

        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, k=k, dim=-1)

        # Convert to probabilities
        top_k_probs = F.softmax(top_k_values, dim=-1)

        # Invert the probabilities
        inverted_probs = 1.0 - top_k_probs
        final_probs = (1.0 - explore_beta) * top_k_probs + explore_beta * inverted_probs

        # Ensure probabilities sum to 1
        final_probs = final_probs / (torch.sum(final_probs, dim=-1, keepdim=True) + 1e-8)

        # Sample from inverted distribution
        sampled_indices = torch.multinomial(final_probs, num_samples=1)

        # Map back to original token indices
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
        explore_beta: float = 0.0,
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
            explore_beta (float, optional): Inverted probability weight (default: 0.0).

        Returns:
            torch.Tensor: The sampled token IDs for the next step in the sequence.
        """
        assert token_logits.dim() == 2
        assert token_logits.size(0) == temperature.size(0)

        batch_size, vocab_size = token_logits.shape

        # Handle temperature = 0 case (greedy decoding)
        zero_temp_mask = temperature == 0
        if zero_temp_mask.all():
            return token_logits.argmax(dim=-1)

        if do_exploration:
            # Use entropy-adaptive sampling with a single parameter
            next_tokens = self._inverted_top_k_sampling(token_logits, top_k=explore_top_k, explore_beta=explore_beta)
        else:
            # Pre-scale logits with temperature (avoid division by zero)
            scaled_logits = torch.where(
                temperature.unsqueeze(1) > 0, token_logits / temperature.unsqueeze(1).clamp(min=1e-8), token_logits
            )

            # Standard sampling with top-k and top-p filtering
            if top_k is not None and top_k > 0:
                k = min(top_k, vocab_size)
                top_k_values, top_k_indices = torch.topk(scaled_logits, k=k, dim=-1)
                scaled_logits = torch.full_like(scaled_logits, float('-inf'))
                scaled_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

            if top_p is not None and (top_p > 0 and top_p < 1.0):
                # Vectorized nucleus sampling
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                # Mask tokens exceeding top_p
                mask = cumulative_probs <= top_p
                mask = mask | (cumulative_probs == probs)  # Keep at least one token
                sorted_logits = sorted_logits.masked_fill(~mask, float('-inf'))

                # Reconstruct filtered logits in original order
                scaled_logits = torch.full_like(scaled_logits, float('-inf'))
                scaled_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

            # Sample from the filtered distribution
            probs = F.softmax(scaled_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Replace tokens for temperature=0 cases
        if zero_temp_mask.any():
            greedy_tokens = token_logits.argmax(dim=-1)
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
        explore_top_k: int = 100,
        explore_beta: float = 0.5,
        explore_skip_n: int = 3,
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
            explore_top_k (int, optional): Number of top-k candidates to consider during exploration (default: 50).
            explore_beta (float, optional): Inverted probability weight (default: 0.5).
            explore_skip_n (int, optional): Skip explore on the first N token, for example in R1 style the first might be `<think>` token.
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

        assert temperature.size(0) == input_ids.size(0)

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
            if explore_skip_n and generated_tokens < explore_skip_n:  # skip explore on the <think> token
                do_exploration = False

            # Sample next tokens
            next_tokens = self._sample_next_batch_tokens(
                token_logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_exploration=do_exploration,
                explore_top_k=explore_top_k,
                explore_beta=explore_beta,
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
        explore_beta=0.4,
    )

    # Decode the output tokens back to text
    input_len = inputs.input_ids.shape[1]
    generated_texts = tokenizer.batch_decode(output.sequences[:, input_len:], skip_special_tokens=True)

    # print("Generated Texts:")
    for i, text in enumerate(generated_texts):
        print(f"Generated [{i}]: '{text}'\n")
        print('\n\n--\n\n')
