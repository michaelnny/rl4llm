import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput

logger = logging.getLogger(__name__)


class CustomLLMGenerator:
    """
    A custom class for LLM text generation with batch-specific temperatures and KV caching.
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
        if eos_token_id is not None:
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id or 0) * (1 - unfinished_sequences)
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id))

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)

        return input_ids, attention_mask, unfinished_sequences

    def _calculate_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates entropy for each item in the batch of logits."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # Calculate entropy for each batch item
        return entropy

    def _sample_next_token(
        self,
        logits: torch.Tensor,  # Single item logits
        temperature: float,
        top_p: float,
        do_exploration: bool = False,
        exploration_top_k: int = 0,
        exploration_beta: float = 0.5,
    ) -> torch.Tensor:
        """Sample next token for a single item in the batch."""
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        if do_exploration and exploration_top_k > 1:
            top_k_values, top_k_indices = torch.topk(logits, k=exploration_top_k, dim=-1)

            # Convert to probabilities
            probs = F.softmax(top_k_values, dim=-1)

            # Simple but effective: raise probabilities to a power < 1
            # This flattens the distribution, giving lower-probability tokens more chance
            probs = probs.pow(exploration_beta)
            probs = probs / probs.sum()

            # # Uniform sampling within top-k (original behavior if temp is 1.0)
            # probs = torch.ones_like(top_k_values) / exploration_top_k

            sampled_indices = torch.multinomial(probs, num_samples=1)
            return torch.gather(top_k_indices, -1, sampled_indices)
        else:
            logits = logits / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

                logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

    def _sample_next_tokens(
        self,
        token_logits: torch.Tensor,
        temperature: torch.Tensor,
        top_p: float,
        do_exploration: bool = False,
        exploration_top_k: int = 0,
        exploration_beta: float = 1.0,
    ) -> torch.Tensor:
        """Sample next tokens for the entire batch."""
        next_tokens = []
        for logits, temp in zip(token_logits, temperature):
            next_token = self._sample_next_token(
                logits, temp.item(), top_p, do_exploration, exploration_top_k, exploration_beta
            )
            next_tokens.append(next_token)
        return torch.cat(next_tokens, dim=0)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: Union[torch.Tensor, float],
        pad_token_id: int,
        eos_token_id: int,
        top_p: float = 1.0,
        max_new_tokens: int = 50,
        enable_exploration: bool = False,
        random_start_steps: int = 0,
        uncertainty_threshold: float = 0.5,
        exploration_top_k: int = 5,
        exploration_beta: float = 0.5,
        **kwargs,
    ) -> GenerateDecoderOnlyOutput:
        """Generate text with batch-specific temperatures."""
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        cur_len = input_ids.shape[1]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        past_key_values = None

        seq_entropies = []

        while cur_len < max_new_tokens:
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
                # 1. Initial Random Start Exploration
                if random_start_steps is not None and random_start_steps > 0 and (cur_len - prompt_len) < random_start_steps:
                    do_exploration = True

                # 2. Uncertainty-Based Exploration
                elif uncertainty_threshold is not None and exploration_top_k > 0:
                    entropy_values = self._calculate_entropy(next_token_logits)
                    avg_entropy = torch.mean(entropy_values).item()
                    seq_entropies.append(avg_entropy)
                    if avg_entropy < uncertainty_threshold:
                        do_exploration = True

            # Sample next tokens
            next_tokens = self._sample_next_tokens(
                next_token_logits,
                temperature,
                top_p,
                do_exploration,
                exploration_top_k,
                exploration_beta,
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

            cur_len = input_ids.shape[1]

        # for debugging
        if seq_entropies:
            entropies = torch.tensor(seq_entropies)
            logger.debug(f"Mean entropy: {entropies.mean().item()}")
            logger.debug(f"Median entropy: {entropies.median().item()}")
            logger.debug(f"P90 entropy: {torch.quantile(entropies, 0.90).item()}")
            logger.debug(f"P95 entropy: {torch.quantile(entropies, 0.95).item()}")
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
