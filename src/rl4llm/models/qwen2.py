"""Example of a custom QWen2.5 model with an additional value head for RL PPO"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class ExtendedModelOutput(CausalLMOutputWithPast):
    """Extended model output with additional state values"""

    values: Optional[torch.Tensor] = None


class ValueHead(nn.Module):
    """Simplified value head with residual connection and dropout"""

    def __init__(self, hidden_dim: int, scaling_factor: float = 0.75, dropout_prob: float = 0.2):
        super().__init__()

        assert scaling_factor >= 0.5 and scaling_factor <= 2.0
        # Calculate scaled_dim and ensure it's a multiple of 256
        base_dim = int(hidden_dim * scaling_factor)
        scaled_dim = 256 * ((base_dim + 255) // 256)  # Round up to nearest multiple of 256

        self.w1 = nn.Linear(hidden_dim, scaled_dim, bias=False)
        self.w2 = nn.Linear(scaled_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, scaled_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.w1(x)) * self.w3(x)  # SwiGLU
        x = self.w2(x)
        x = self.dropout(x + residual)
        return self.out(x).squeeze(-1)


class CustomQwen2Model(Qwen2ForCausalLM):
    """Custom decoder only transformer QWen2.5 model with additional value head for RL"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.value_head = ValueHead(config.hidden_size, dropout_prob=0.0)

    def forward(self, input_ids=None, attention_mask=None, return_values=False, **kwargs) -> ExtendedModelOutput:
        # Call the original model's forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Compute state values
        values = None
        if return_values:
            values = self.value_head(outputs.hidden_states[-1]).squeeze(-1)

        # Return ExtendedModelOutput with the computed state values
        return ExtendedModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            values=values,
        )

    # def generate(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     generation_config: Optional[GenerationConfig] = None,
    #     logits_processor: Optional[Any] = None,
    #     stopping_criteria: Optional[Any] = None,
    #     prefix_allowed_tokens_fn: Optional[Any] = None,
    #     synced_gpus: Optional[bool] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     """
    #     Enhanced generate method supporting KV cache and sampling, maintaining compatibility
    #     with the standard generate interface.
    #     """
    #     # Prepare generation config
    #     generation_config = generation_config if generation_config is not None else self.generation_config
    #     generation_config = copy.deepcopy(generation_config)
    #     model_kwargs = generation_config.update(**kwargs)
        
    #     # Set default values for generation
    #     if generation_config.max_length is None:
    #         generation_config.max_length = self.config.max_position_embeddings
            
    #     if generation_config.pad_token_id is None:
    #         generation_config.pad_token_id = self.config.pad_token_id
            
    #     if generation_config.eos_token_id is None:
    #         generation_config.eos_token_id = self.config.eos_token_id

    #     # Prepare model inputs
    #     input_ids_len = input_ids.shape[-1]
        
    #     # Initialize generation variables
    #     unfinished_sequences = torch.ones(
    #         input_ids.shape[0], dtype=torch.long, device=input_ids.device
    #     )
        
    #     # Use model's prepare_inputs_for_generation if available
    #     model_kwargs["use_cache"] = True
    #     model_kwargs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        
    #     # Main generation loop
    #     while True:
    #         outputs = self.forward(**model_kwargs)
    #         next_token_logits = outputs.logits[:, -1, :]
            
    #         # Apply temperature if specified
    #         if generation_config.temperature != 1.0:
    #             next_token_logits = next_token_logits / generation_config.temperature
            
    #         # Apply top-p sampling if specified
    #         if generation_config.top_p < 1.0:
    #             sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
    #             cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
    #             sorted_indices_to_remove = cumulative_probs > generation_config.top_p
    #             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #             sorted_indices_to_remove[..., 0] = 0
                
    #             indices_to_remove = sorted_indices_to_remove.scatter(
    #                 1, sorted_indices, sorted_indices_to_remove
    #             )
    #             next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
    #         # Sample next token
    #         probs = torch.softmax(next_token_logits, dim=-1)
    #         next_tokens = torch.multinomial(probs, num_samples=1)
            
    #         # Update sequences
    #         input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
    #         # Update model kwargs for next step
    #         model_kwargs = self.prepare_inputs_for_generation(
    #             input_ids,
    #             past_key_values=outputs.past_key_values,
    #             **model_kwargs
    #         )
            
    #         # Update unfinished sequences
    #         unfinished_sequences = unfinished_sequences.mul(
    #             (next_tokens != generation_config.eos_token_id).long()
    #         )
            
    #         # Stop if max length reached or all sequences finished
    #         if (
    #             unfinished_sequences.max() == 0 or 
    #             input_ids.shape[-1] - input_ids_len >= generation_config.max_new_tokens
    #         ):
    #             break
                
    #     # Prepare output in the standard format
    #     return GenerateOutput(
    #         sequences=input_ids,
    #         scores=None,  # Can be added if needed
    #         attentions=None,  # Can be added if needed
    #         hidden_states=None,  # Can be added if needed
    #     )