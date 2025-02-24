from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from rl4llm.generations import CustomLLMGenerator


@pytest.fixture
def mock_model():
    model = Mock(spec=PreTrainedModel)
    return model


@pytest.fixture
def generator(mock_model):
    return CustomLLMGenerator(mock_model)


def test_update_sequences(generator: CustomLLMGenerator):
    # Test normal sequence update
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])
    next_tokens = torch.tensor([5, 6])
    unfinished_sequences = torch.tensor([1, 1])
    eos_token_id = 0
    pad_token_id = 1

    new_input_ids, new_attention_mask, new_unfinished = generator._update_sequences(
        input_ids, attention_mask, next_tokens, unfinished_sequences, eos_token_id, pad_token_id
    )

    assert new_input_ids.shape == (2, 3)
    assert new_attention_mask.shape == (2, 3)
    assert torch.equal(new_input_ids[:, -1], next_tokens)
    assert torch.all(new_attention_mask[:, -1] == 1)


def test_zero_temperature_sampling(generator: CustomLLMGenerator):
    # Test sampling with temperature = 0 (deterministic)
    logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    temperature = torch.tensor([0.0, 0.0])

    next_tokens = generator._sample_next_tokens(logits, temperature, top_p=1.0, top_k=0)

    expected = torch.tensor([2, 0])  # argmax positions
    assert torch.equal(next_tokens, expected)


def test_exploration_sampling(generator: CustomLLMGenerator):
    # Test exploration sampling
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0, 0.0]])
    temperature = torch.tensor([1.0, 1.0])

    # Set random seed for reproducibility
    torch.manual_seed(42)

    next_tokens = generator._sample_next_tokens(
        logits, temperature, top_p=1.0, top_k=0, do_exploration=True, explore_top_k=3, explore_noise=0.5
    )

    # Assert shape is correct
    assert next_tokens.shape == (2,)
    # Assert values are within top-k range
    assert torch.all(next_tokens < 3)  # Should only sample from top 3 tokens


def test_mixed_temperature_sampling(generator: CustomLLMGenerator):
    # Test sampling with mixed temperatures
    logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    temperature = torch.tensor([0.0, 1.0])

    next_tokens = generator._sample_next_tokens(logits, temperature, top_p=1.0, top_k=0)

    # First sequence should be deterministic (temp=0)
    assert next_tokens[0] == 2  # argmax position
    # Second sequence should be stochastic (temp=1)
    assert next_tokens[1] >= 0 and next_tokens[1] < 3


def test_generate_with_exploration(generator: CustomLLMGenerator):
    # Test the full generation process with exploration
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])
    temperature = torch.tensor([0.8, 1.0])

    # Create a proper mock output that will be returned each time
    class MockOutput:
        def __init__(self):
            self.logits = torch.ones(2, 1, 10)  # batch_size=2, sequence_length=1, vocab_size=10
            self.past_key_values = None

    mock_output = MockOutput()
    # Make sure the model returns the same mock output each time
    generator.model.return_value = mock_output

    output = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=2,
        enable_exploration=True,
        explore_start_steps=1,
    )

    expected_shape = (input_ids.shape[0], input_ids.shape[1] + 2)  # (2, 4) = original shape + 2 new tokens
    assert output.sequences.shape == expected_shape, f"Expected shape {expected_shape}, got {output.sequences.shape}"
    assert len(generator.model.mock_calls) == 2, 'Model should be called exactly twice for 2 new tokens'


def test_early_stopping(generator: CustomLLMGenerator):
    # Test generation stops when EOS token is generated
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.tensor([[1, 1]])
    temperature = torch.tensor([1.0])

    # Mock model to return logits that will generate EOS token
    mock_output = Mock()
    mock_output.logits = torch.zeros(1, 1, 10)
    mock_output.logits[0, 0, 1] = 100  # Make EOS token very likely
    mock_output.past_key_values = None
    generator.model.return_value = mock_output

    output = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=10,  # Set high, should stop early
    )

    # Should have stopped after generating EOS
    assert output.sequences.shape[1] < 12  # Original + max_new_tokens


def test_batch_specific_temperatures(generator: CustomLLMGenerator):
    # Test that different temperatures in batch work correctly
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])
    temperature = torch.tensor([0.0, 1.0])  # One deterministic, one stochastic

    # Create logits with clear max values
    mock_output = Mock()
    mock_output.logits = torch.zeros(2, 1, 10)
    mock_output.logits[0, 0, 5] = 100  # Clear maximum for first sequence
    mock_output.logits[1, 0] = torch.ones(10)  # Uniform for second sequence
    mock_output.past_key_values = None
    generator.model.return_value = mock_output

    output = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=1,
    )

    # First sequence (temp=0) should have generated token 5
    assert output.sequences[0, -1] == 5
    # Second sequence (temp=1) should have generated a valid token
    assert 0 <= output.sequences[1, -1] < 10
