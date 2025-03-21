import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


class BinaryClassificationHead(nn.Module):
    """Classification head for adapting a pretrained model to binary classification."""

    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2)  # Binary classification

    def forward(self, hidden_state):
        # Use the last token's hidden state for classification
        # hidden_states shape: [batch_size,  hidden_dim]
        last_hidden = self.dropout(hidden_state)
        logits = self.classifier(last_hidden).float()
        return logits


class ClassifierModel(nn.Module):
    """Wrapper around a pretrained model with a classification head."""

    def __init__(self, pretrained_model: PreTrainedModel, dropout_prob=0.1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = self.pretrained_model.config
        self.hidden_size = self.pretrained_model.config.hidden_size
        self.classification_head = BinaryClassificationHead(self.hidden_size, dropout_prob)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get the output from the pretrained model
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)

        # For causal LM models, access hidden states from the tuple
        if hasattr(outputs, 'hidden_states'):
            # If hidden_states is an attribute
            hidden_states = outputs.hidden_states[-1]
        elif isinstance(outputs.hidden_states, tuple):
            # If hidden_states is a tuple
            hidden_states = outputs.hidden_states[-1]

        # If an attention mask is provided, use it to select the last non-padding token for each sample.
        if attention_mask is not None:
            # Compute lengths: number of non-padding tokens per sample.
            lengths = attention_mask.sum(dim=1)
            # Adjust lengths to get zero-indexed positions.
            last_token_indices = lengths - 1
            batch_size = hidden_states.size(0)
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            # Gather the last non-padding token's hidden state for each sample.
            last_hidden = hidden_states[batch_indices, last_token_indices, :]
        else:
            # Fallback: use the last token hidden state.
            last_hidden = hidden_states[:, -1, :]

        # Pass through classification head
        logits = self.classification_head(last_hidden)

        return logits
