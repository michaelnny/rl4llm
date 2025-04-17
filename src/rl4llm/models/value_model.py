import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class ValueOutput(ModelOutput):
    """
    Output class for AutoModelWithValueHead, containing predicted values.
    """

    values: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class AutoModelWithValueHead(PreTrainedModel):
    """
    A PreTrainedModel wrapper that adds a value head to a base transformer model.

    This model is useful for tasks like reinforcement learning (e.g., PPO) where
    a value prediction is needed alongside the base model's outputs.

    When loading from a checkpoint containing only the base model weights
    (e.g., 'gpt2'), use `AutoModelWithValueHead.from_pretrained('gpt2', ignore_mismatched_sizes=True)`.
    The value head will be initialized using the logic in `_init_weights`.

    When loading from a checkpoint saved by `model.save_pretrained('./my_checkpoint')`
    (which contains both base and value head weights), use
    `AutoModelWithValueHead.from_pretrained('./my_checkpoint')`. The saved value head
    weights will be loaded correctly.
    """

    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    config_class = AutoConfig  # Use AutoConfig to be flexible

    def __init__(self, config: PretrainedConfig):
        """Initializes the model with a base model and a value head."""
        # Ensure config is PretrainedConfig (though AutoConfig usually inherits)
        if not isinstance(config, PretrainedConfig):
            # If loaded via AutoConfig, it might be a dict-like object initially
            # Let's try to load it properly if it looks like one
            try:
                config = AutoConfig.for_model(**config.to_dict())
            except Exception as e:
                raise TypeError(
                    f"config must be a PretrainedConfig or convertible, got {type(config)}. Error: {e}"
                )
        # This calls self.post_init() -> self.init_weights() -> self.apply(self._init_weights)
        super().__init__(config)

        # Load the base model structure using the config
        # Weights will be loaded later by from_pretrained if applicable
        self.model = AutoModel.from_config(config)

        # --- Handle potential LM head in base model ---
        # Check if the config suggests an LM head exists and might be tied
        has_lm_head_attr = hasattr(self.model, 'lm_head')
        output_embeddings = getattr(
            self.model, 'get_output_embeddings', lambda: None
        )()
        is_tied = (
            getattr(self.config, 'tie_word_embeddings', False)
            and output_embeddings is not None
        )

        if (
            has_lm_head_attr
            and getattr(self.model, 'lm_head', None) is not None
        ):
            if not is_tied:
                logger.info(
                    'Deleting non-tied lm_head found on base model after AutoModel.from_config.'
                )
                # Use delattr for safer deletion
                try:
                    delattr(self.model, 'lm_head')
                except AttributeError:
                    logger.info(  # Should not happen if has_lm_head_attr is true, but good practice
                        'Could not delete lm_head, attribute might not exist directly.'
                    )
            else:
                logger.info(
                    'Keeping tied lm_head (will not be used by value head, but needed for base model consistency).'
                )
        # --- End LM head handling ---

        # Define the value head
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Note: Initialization of value_head happens in _init_weights
        # self.post_init() is called by super().__init__(), no need to call it again

    # Override _init_weights for custom initialization of the value head
    def _init_weights(self, module):
        """Initialize the weights."""
        # Let the base class handle standard initializations first (optional, depends on desired behavior)
        # super()._init_weights(module) # Usually not needed unless modifying base behavior

        # Custom initialization for the value head
        if isinstance(module, nn.Linear) and module is self.value_head:
            std_dev = 0.02
            # Use config attributes safely
            num_hidden_layers = getattr(self.config, 'num_hidden_layers', None)
            if num_hidden_layers:
                # Check if num_hidden_layers is a valid number > 0
                if (
                    isinstance(num_hidden_layers, (int, float))
                    and num_hidden_layers > 0
                ):
                    std_dev /= math.sqrt(2.0 * num_hidden_layers)
                else:
                    logger.warning(
                        f"num_hidden_layers found in config but is not a positive number: {num_hidden_layers}. Using default std_dev."
                    )

            logger.info(
                f'Initializing value head weights with std_dev: {std_dev}'
            )
            nn.init.normal_(module.weight, mean=0.0, std=std_dev)
            if module.bias is not None:
                logger.info('Initializing value head bias to zeros.')
                nn.init.zeros_(module.bias)
        # You might need to initialize other custom layers here if you add more
        # elif isinstance(module, (nn.LayerNorm, nn.Embedding)): # Example if needed
        #     module.weight.data.fill_(1.0) # Example

    def get_input_embeddings(self):
        """Returns the input embeddings layer from the base model."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Sets the input embeddings layer for the base model."""
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> ValueOutput:
        """
        Performs a forward pass through the base model and the value head.
        """
        # Ensure internal calls use return_dict=True for consistent output access
        kwargs['return_dict'] = True
        # Preserve user's request for hidden_states/attentions if passed
        output_hidden_states = kwargs.get('output_hidden_states', False)
        output_attentions = kwargs.get('output_attentions', False)
        kwargs['output_hidden_states'] = output_hidden_states
        kwargs['output_attentions'] = output_attentions

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Get the last hidden state
        # BaseModelOutputWithPooling often has 'pooler_output', but for RL value heads,
        # using the last hidden state of the sequence is more common.
        # Check if last_hidden_state exists, otherwise log an error or adapt.
        if not hasattr(outputs, 'last_hidden_state'):
            raise AttributeError(
                "The base model output does not contain 'last_hidden_state'. "
                'Check the base model type and its forward pass implementation.'
            )
        last_hidden_state = outputs.last_hidden_state

        # Pass the last hidden state through the value head
        # Typically, for value prediction in RL, you might want the value for the *last* token
        # or an average, depending on your setup. Here, we calculate it for all tokens.
        # Squeeze the last dimension (size 1)
        values = self.value_head(last_hidden_state).squeeze(-1)

        return ValueOutput(
            values=values,
            hidden_states=(
                outputs.hidden_states if output_hidden_states else None
            ),
            attentions=outputs.attentions if output_attentions else None,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model"""
        if not self.supports_gradient_checkpointing:
            logger.warning(
                f"{self. MRO} does not support gradient checkpointing."
            )
            return
        # Pass gradient_checkpointing_kwargs if provided, otherwise default behavior
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {
                'use_reentrant': True
            }  # Default often needed

        # Check if the base model actually has the method
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            logger.info('Gradient checkpointing enabled on the base model.')
        else:
            logger.warning(
                "Base model does not have 'gradient_checkpointing_enable' method."
            )
