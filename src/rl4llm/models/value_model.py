"""Custom model wrapper around pretrained model with a value head."""

import math
import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from rl4llm.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


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
    """

    def __init__(self, config):
        """Initializes the model with a base model and a value head."""
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        if hasattr(self.model, "lm_head"):
            del self.model.lm_head
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        self._init_value_head_weights()
        self.post_init()

    def _init_value_head_weights(self):
        """Initializes the value head weights with scaled normal distribution."""
        std_dev = 0.02
        if hasattr(self.config, "num_hidden_layers"):
            std_dev /= math.sqrt(2.0 * self.config.num_hidden_layers)
        logger.info("Initialize value head weights...")
        nn.init.normal_(self.value_head.weight, mean=0.0, std=std_dev)
        if self.value_head.bias is not None:
            nn.init.zeros_(self.value_head.bias)

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

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            **kwargs: Additional arguments passed to the base model.

        Returns:
            ValueOutput: An object containing the predicted values and optionally
                         hidden states and attentions from the base model.
        """
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = kwargs.get("output_hidden_states", False)
        kwargs["output_attentions"] = kwargs.get("output_attentions", False)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        last_hidden_state = outputs.last_hidden_state
        values = self.value_head(last_hidden_state).squeeze(-1)
        return ValueOutput(
            values=values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """
        Loads a pretrained model instance.

        Args:
            pretrained_model_name_or_path (str): Identifier for the pretrained model
                                                 (Hub name or local path).
            *model_args: Positional arguments passed to the underlying `from_pretrained`.
            **kwargs: Keyword arguments passed to the underlying `from_pretrained`.

        Returns:
            AutoModelWithValueHead: The loaded model instance.
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if os.path.isdir(pretrained_model_name_or_path):
            try:
                logger.info(
                    f"Attempting to load model from local path: {pretrained_model_name_or_path}"
                )
                model = super().from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **kwargs,
                )
                logger.info("Successfully loaded model from local path.")
                return model
            except Exception as e:
                logger.warning(
                    f"Could not load model from local path {pretrained_model_name_or_path}: {e}. "
                    f"Falling back to loading base model and initializing value head."
                )
        logger.info(f"Loading base model weights from {pretrained_model_name_or_path}")
        base_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model = cls(config)
        model.model = base_model
        model._init_value_head_weights()
        logger.info(
            f"Initialized value head for model loaded from {pretrained_model_name_or_path}"
        )
        return model
