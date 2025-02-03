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
from transformers.generation import GenerationConfig
from transformers.generation.utils import GenerateOutput


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
        # residual = x
        x = F.silu(self.w1(x)) * self.w3(x)  # SwiGLU
        x = self.w2(x)
        # x = self.dropout(x + residual)
        x = self.dropout(x)
        return self.out(x).squeeze(-1)


class AttentionValueHead(nn.Module):
    """Value head with a self-attention layer"""

    def __init__(self, hidden_dim: int, num_attention_heads: int = 4, scaling_factor: float = 1.0, dropout_prob: float = 0.1):
        super().__init__()

        assert scaling_factor >= 0.5 and scaling_factor <= 2.0
        scaled_dim = int(256 * (((hidden_dim * scaling_factor) + 255) // 256)) if scaling_factor != 1.0 else hidden_dim

        self.hidden_dim = hidden_dim

        # Self-attention layer with minimal dropout
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout_prob,  # Keep only this dropout in attention mechanism
            batch_first=True,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, scaled_dim), nn.GELU(), nn.Linear(scaled_dim, hidden_dim))

        # Output projection
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        # FFN block with residual
        x = x + self.ffn(x)

        return self.out(x).squeeze(-1)


class CustomQwen2Model(Qwen2ForCausalLM):
    """Custom decoder only transformer QWen2.5 model with additional value head for RL"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.value_head = AttentionValueHead(config.hidden_size, dropout_prob=0.1)

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
