import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from rl4llm.models.value_model import AutoModelWithValueHead, ValueOutput


# Mock classes
class MockConfig(PretrainedConfig):
    """Mock configuration class inheriting from PretrainedConfig."""

    model_type = 'mock_model'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs.get('hidden_size', 16)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 2)
        self.vocab_size = kwargs.get('vocab_size', 100)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 32)


class MockModelOutput:
    """Mock output class for the base model."""

    def __init__(self, last_hidden_state, hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions


class MockBaseModel(PreTrainedModel):
    """Mock base model mimicking a transformer model for testing."""

    config_class = MockConfig

    def __init__(self, config: MockConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dummy_layer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError(
                'input_ids cannot be None for MockBaseModel forward'
            )
        emb = self.embeddings(input_ids)
        last_hidden_state = self.dummy_layer(emb)
        hidden_states = (
            (emb, last_hidden_state) * (self.config.num_hidden_layers // 2 + 1)
            if output_hidden_states
            else None
        )
        attentions = None
        return (
            MockModelOutput(last_hidden_state, hidden_states, attentions)
            if return_dict
            else (last_hidden_state,)
        )

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        print(f"Mock saving model to {save_directory}")

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ):
        config = kwargs.pop('config', None) or cls.config_class()
        model = cls(config)
        model_path = os.path.join(
            pretrained_model_name_or_path, 'pytorch_model.bin'
        )
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        return model

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value


# Fixtures
@pytest.fixture(scope='module')
def mock_config_instance():
    """Provides a reusable mock configuration instance."""
    return MockConfig()


@pytest.fixture
def dummy_input(mock_config_instance):
    """Provides dummy input data for testing."""
    seq_len, batch_size = 5, 2
    input_ids = torch.randint(
        0,
        mock_config_instance.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


@pytest.fixture
def value_model_instance(mock_config_instance):
    """Provides an instance of AutoModelWithValueHead with mocked base model."""
    mock_base = MockBaseModel(mock_config_instance)
    with patch(
        'transformers.AutoModel.from_config', return_value=mock_base
    ) as mock_from_config:
        model = AutoModelWithValueHead(mock_config_instance)
    mock_from_config.assert_called_once_with(mock_config_instance)
    assert hasattr(model, '_modules')
    return model


# Tests
def test_initialization(value_model_instance, mock_config_instance):
    """Tests if the model initializes correctly with mocked base model and value head."""
    assert isinstance(value_model_instance, PreTrainedModel)
    assert isinstance(value_model_instance.model, MockBaseModel)
    assert isinstance(value_model_instance.value_head, nn.Linear)
    assert (
        value_model_instance.value_head.in_features
        == mock_config_instance.hidden_size
    )
    assert value_model_instance.value_head.out_features == 1
    assert value_model_instance.value_head.weight.requires_grad


@pytest.mark.parametrize('output_hidden_states', [False, True])
def test_forward_pass(value_model_instance, dummy_input, output_hidden_states):
    """Tests the forward pass with and without hidden states."""
    value_model_instance.eval()
    with torch.no_grad():
        outputs = value_model_instance(
            **dummy_input, output_hidden_states=output_hidden_states
        )
    assert isinstance(outputs, ValueOutput)
    assert outputs.values.shape == dummy_input['input_ids'].shape
    assert outputs.values.dtype == torch.float32
    assert (outputs.hidden_states is not None) == output_hidden_states
    assert outputs.attentions is None
    if output_hidden_states:
        assert len(outputs.hidden_states) >= 2
