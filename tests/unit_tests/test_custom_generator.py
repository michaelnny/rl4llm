from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateDecoderOnlyOutput

from rl4llm.generations import CustomLLMGenerator


@pytest.fixture
def mock_model():
    model = Mock(spec=PreTrainedModel)
    return model


@pytest.fixture
def generator(mock_model):
    return CustomLLMGenerator(mock_model)


# --- Enhanced Existing Tests ---
def test_update_sequences_with_padding(generator: CustomLLMGenerator):
    # Test sequence update with some sequences finished
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])
    next_tokens = torch.tensor([0, 5])  # First sequence hits EOS
    unfinished_sequences = torch.tensor([1, 1])
    eos_token_id = 0
    pad_token_id = 1

    new_input_ids, new_attention_mask, new_unfinished = generator._update_sequences(
        input_ids, attention_mask, next_tokens, unfinished_sequences, eos_token_id, pad_token_id
    )

    assert torch.equal(new_input_ids, torch.tensor([[1, 2, 0], [3, 4, 5]]))
    assert torch.equal(new_attention_mask, torch.tensor([[1, 1, 1], [1, 1, 1]]))
    assert torch.equal(new_unfinished, torch.tensor([0, 1]))  # First sequence finished


def test_entropy_adaptive_top_k_sampling_diversity(generator: CustomLLMGenerator):
    # Test that entropy-adaptive sampling increases diversity
    logits = torch.tensor([[10.0, 9.0, 8.0, 0.0, 0.0]])  # Strong preference for token 0
    torch.manual_seed(42)

    # Low entropy ratio should concentrate on top tokens
    tokens_low = generator._entropy_adaptive_top_k_sampling(logits, top_k=3, min_entropy_ratio=0.1)
    assert tokens_low.item() in [0, 1, 2]  # Should be top-k

    # High entropy ratio should encourage diversity
    samples = [generator._entropy_adaptive_top_k_sampling(logits, top_k=3, min_entropy_ratio=0.9).item() for _ in range(10)]
    unique_tokens = len(set(samples))
    assert unique_tokens > 1, 'High entropy ratio should yield diverse tokens'


def test_exploration_sampling_edge_cases(generator: CustomLLMGenerator):
    # Test exploration with extreme parameters
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    temperature = torch.tensor([1.0])

    # explore_top_k larger than vocab size
    torch.manual_seed(42)
    tokens = generator._sample_next_batch_tokens(
        logits, temperature, top_p=1.0, top_k=0, do_exploration=True, explore_top_k=10, explore_beta=0.5
    )
    assert tokens.shape == (1,) and 0 <= tokens.item() < 3

    # Zero entropy ratio (should still sample valid tokens)
    tokens = generator._sample_next_batch_tokens(
        logits, temperature, top_p=1.0, top_k=0, do_exploration=True, explore_top_k=3, explore_beta=0.0
    )
    assert tokens.shape == (1,) and 0 <= tokens.item() < 3


def test_mixed_temperature_sampling_with_top_p(generator: CustomLLMGenerator):
    # Test mixed temperatures with nucleus sampling
    logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    temperature = torch.tensor([0.0, 1.0])

    torch.manual_seed(42)
    next_tokens = generator._sample_next_batch_tokens(logits, temperature, top_p=0.7, top_k=0)

    assert next_tokens[0] == 2  # Temp=0, deterministic
    assert next_tokens[1] in [0, 1, 2]  # Temp=1, top-p filtered sampling


# --- New Tests ---
def test_top_k_filtering(generator: CustomLLMGenerator):
    # Test top-k sampling restricts token choices
    logits = torch.tensor([[1.0, 2.0, 5.0, 4.0, 0.0]])
    temperature = torch.tensor([1.0])

    torch.manual_seed(42)
    samples = [generator._sample_next_batch_tokens(logits, temperature, top_p=1.0, top_k=3).item() for _ in range(10)]
    assert all(t in [2, 3, 1] for t in samples)  # Top-3 tokens by logit value


def test_generate_exploration_skip(generator: CustomLLMGenerator):
    # Test exploration skipping for first N tokens
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.tensor([[1, 1]])
    temperature = 1.0
    mock_output = Mock()
    mock_output.logits = torch.ones(1, 1, 5)  # Uniform logits
    mock_output.past_key_values = None
    generator.model.return_value = mock_output

    torch.manual_seed(42)
    output = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=3,
        explore_start_steps=2,
        explore_top_k=3,
        explore_beta=0.5,
        explore_skip_n=1,
    )

    assert output.sequences.shape == (1, 5)  # 2 initial + 3 new
    # First new token (after skip) should use standard sampling, next two should use exploration


def test_generate_with_high_temperature(generator: CustomLLMGenerator):
    # Test generation with very high temperature for diversity
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.tensor([[1, 1]])
    temperature = 10.0
    mock_output = Mock()
    mock_output.logits = torch.tensor([[[1.0, 2.0, 3.0]]])  # Controlled logits
    mock_output.past_key_values = None
    generator.model.return_value = mock_output

    torch.manual_seed(42)
    output = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=3,
    )

    new_tokens = output.sequences[0, 2:].tolist()
    assert len(set(new_tokens)) > 1, 'High temperature should yield diverse tokens'


def test_batch_consistency_with_exploration(generator: CustomLLMGenerator):
    # Test batch consistency with exploration and varying temperatures
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])
    temperature = torch.tensor([0.0, 1.0])  # One greedy, one exploratory

    mock_output = Mock()
    mock_output.logits = torch.tensor([[[5.0, 1.0, 0.0]], [[1.0, 1.0, 1.0]]])  # First greedy, second uniform
    mock_output.past_key_values = None
    generator.model.return_value = mock_output

    torch.manual_seed(42)
    output = generator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=temperature,
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=2,
        explore_start_steps=1,
        explore_top_k=2,
        explore_beta=0.5,
    )

    assert isinstance(output, GenerateDecoderOnlyOutput)
    assert output.sequences[0, -1] == 0  # Greedy picks max logit (0)
    assert output.sequences[1, -1] in [0, 1, 2]  # Exploration samples from uniform
