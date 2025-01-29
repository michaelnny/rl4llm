"""Example of a custom QWen2.5 model with an additional value head for RL PPO"""

import torch
import torch.nn as nn


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
    """A simple value head with two linear layers"""

    def __init__(self, hidden_dim: int, num_units: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, num_units, bias=False)
        self.activation1 = nn.Tanh()
        self.linear2 = nn.Linear(num_units, num_units, bias=False)
        self.activation2 = nn.Tanh()
        self.linear3 = nn.Linear(num_units, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute value for the given state"""
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        out = self.linear3(x)
        return out.squeeze(-1)  # [batch_size, seq_len]


class CustomQwen2Model(Qwen2ForCausalLM):
    """Custom decoder only transformer QWen2.5 model with additional value head for RL PPO"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.value_output = ValueHead(config.hidden_size, int(config.hidden_size // 2))

    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> ExtendedModelOutput:
        # Call the original model's forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Compute state values
        values = self.value_output(outputs.hidden_states[-1].detach())

        # Return ExtendedModelOutput with the computed state values
        return ExtendedModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            values=values,
        )
