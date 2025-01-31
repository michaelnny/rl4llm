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
    """Custom decoder only transformer QWen2.5 model with additional value head for RL PPO"""

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
