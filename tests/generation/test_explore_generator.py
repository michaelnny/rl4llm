from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from rl4llm.generation.explore_generator import (
        ExploreLLMGenerator,
        GenerateDecoderOnlyOutput,
    )
except ImportError:
    pytest.skip(
        'Skipping tests because stochastic_llm_generator module not found.',
        allow_module_level=True,
    )


@dataclass
class MockModelOutput:
    logits: torch.Tensor
    past_key_values: tuple = None


@pytest.fixture
def mock_model():
    """Provide a mock model with predefined forward output."""
    model = Mock(spec=PreTrainedModel)
    model.forward.return_value = Mock(
        logits=torch.randn(2, 1, 100),
        past_key_values=tuple(
            tuple(torch.randn(2, 4, 1, 32) for _ in range(2)) for _ in range(4)
        ),
    )
    return model


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer with batch decoding capability."""
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.batch_decode.return_value = ['Test text 1', 'Test text 2']
    tokenizer.eos_token_id = 50
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def device():
    """Provide the available device for tensor operations."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def generator(mock_model, mock_tokenizer, device):
    """Provide an ExploreLLMGenerator instance with mocked dependencies."""
    llm_generator = ExploreLLMGenerator(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=device,
        source_tokens=[50, 51],
        target_tokens=[10, 11],
        prevent_patterns=[[27, 29, 31]],
    )
    mock_model = MagicMock()
    llm_generator.model = mock_model
    vocab_logits = torch.zeros(2, 1, 100, device=device)
    mock_model.side_effect = [
        MockModelOutput(logits=vocab_logits, past_key_values=('mock_past',))
    ] * 3
    return llm_generator


@pytest.fixture
def sample_input_tensors(device):
    """Provide sample input tensors for testing."""
    return {
        'input_ids': torch.randint(0, 100, (2, 5), device=device),
        'attention_mask': torch.ones(2, 5, device=device, dtype=torch.long),
        'logits': torch.randn(2, 100, device=device),
        'temperature': torch.tensor([0.7, 0.0], device=device),
    }


def mock_correctness_callback(text):
    """Mock callback marking text with 'error' as incorrect."""
    return 0.0 if 'error' in text else 1.0


# Initialization Tests
@pytest.mark.parametrize(
    'source_tokens, target_tokens, prevent_patterns',
    [([50, 51], [10, 11], [[27, 29, 31]]), ([], [], [])],
)
def test_initialization(
    mock_model,
    mock_tokenizer,
    device,
    source_tokens,
    target_tokens,
    prevent_patterns,
):
    """Test that the generator initializes with correct attributes."""
    generator = ExploreLLMGenerator(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=device,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        prevent_patterns=prevent_patterns,
    )
    assert generator.model is mock_model
    assert generator.tokenizer is mock_tokenizer
    assert generator.device is device
    assert generator.source_tokens == source_tokens
    assert generator.target_tokens == target_tokens
    assert generator.prevent_patterns == prevent_patterns
    assert (
        (generator.source_tokens_tensor is None)
        if not source_tokens
        else torch.equal(
            generator.source_tokens_tensor,
            torch.tensor(source_tokens, device=device),
        )
    )


# Pattern Detection Tests
@pytest.mark.parametrize(
    'input_ids, pattern, expected',
    [
        (
            torch.tensor([[1, 2, 3, 5, 7, 9, 10], [5, 7, 9, 11, 12, 13, 14]]),
            [5, 7, 9],
            torch.tensor([True, True]),
        ),
        (
            torch.tensor([[1, 2, 3, 5, 8, 9, 10], [5, 7, 10, 11, 12, 13, 14]]),
            [5, 7, 9],
            torch.tensor([False, False]),
        ),
        (
            torch.tensor(
                [
                    [1, 2, 3, 5, 7, 9, 10],
                    [5, 8, 9, 11, 12, 13, 14],
                    [1, 2, 3, 4, 5, 7, 9],
                ]
            ),
            [5, 7, 9],
            torch.tensor([True, False, True]),
        ),
        (
            torch.tensor([[1, 2], [5, 7]]),
            [5, 7, 9],
            torch.tensor([False, False]),
        ),
    ],
)
def test_has_pattern(generator, device, input_ids, pattern, expected):
    """Test pattern detection in sequences."""
    input_ids = input_ids.to(device)
    expected = expected.to(device)
    result = generator._has_pattern(input_ids, pattern)
    assert result.shape == (input_ids.shape[0],)
    assert torch.equal(result, expected)


# Replacement Pattern Tests
@pytest.mark.parametrize(
    'prevent_patterns, generated_ids, expected',
    [
        ([], torch.randint(0, 100, (2, 10)), torch.ones(2, dtype=torch.bool)),
        (
            [[5, 7, 9]],
            torch.tensor(
                [
                    [1, 2, 3, 5, 7, 9, 10],
                    [5, 8, 9, 11, 12, 13, 14],
                    [1, 2, 3, 4, 5, 7, 9],
                ]
            ),
            torch.tensor([False, True, False]),
        ),
        (
            [[1, 2, 3], [5, 6, 7]],
            torch.tensor(
                [
                    [1, 2, 3, 4, 5],
                    [4, 5, 6, 7, 8],
                    [1, 3, 5, 7, 9],
                    [5, 6, 7, 1, 2],
                ]
            ),
            torch.tensor([False, False, True, False]),
        ),
    ],
)
def test_check_replacement_patterns(
    mock_model,
    mock_tokenizer,
    device,
    prevent_patterns,
    generated_ids,
    expected,
):
    """Test replacement pattern checking."""
    generator = ExploreLLMGenerator(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=device,
        prevent_patterns=prevent_patterns,
    )
    generated_ids = generated_ids.to(device)
    expected = expected.to(device)
    result = generator._check_replacement_patterns(generated_ids)
    assert result.shape == (generated_ids.shape[0],)
    assert torch.equal(result, expected)


# Correctness Check Tests
def test_check_correctness_no_callback(generator, device):
    """Test correctness checking with no callback provided."""
    generated_ids = torch.randint(0, 100, (2, 10), device=device)
    can_replace = torch.ones(2, dtype=torch.bool, device=device)
    result = generator._check_correctness(generated_ids, can_replace, None)
    assert result.shape == (2,)
    assert result.all()


def test_check_correctness_with_callback(generator, device, monkeypatch):
    """Test correctness checking with callback marking some as incorrect."""
    generated_ids = torch.randint(0, 100, (3, 10), device=device)
    can_replace = torch.tensor([True, True, False], device=device)
    generator.tokenizer.batch_decode.return_value = [
        'This has an error',
        'This is fine',
    ]
    result = generator._check_correctness(
        generated_ids, can_replace, mock_correctness_callback
    )
    assert result.shape == (3,)
    assert result[0].item() is True
    assert result[1].item() is False
    assert result[2].item() is False


def test_check_correctness_callback_exception(generator, device):
    """Test handling of exceptions in correctness callback."""
    generated_ids = torch.randint(0, 100, (2, 10), device=device)
    can_replace = torch.ones(2, dtype=torch.bool, device=device)

    def failing_callback(text):
        if text == 'Test text 1':
            raise ValueError('Test exception')
        return 0.5

    with patch('builtins.print') as mock_print:
        result = generator._check_correctness(
            generated_ids, can_replace, failing_callback
        )
        assert mock_print.call_count == 1
        assert 'Warning' in mock_print.call_args[0][0]
    assert result[0].item() is True
    assert result[1].item() is True


# Token Replacement Tests
@pytest.mark.parametrize(
    'source_tokens, next_tokens, can_replace, replace_prob, expected_tokens, expected_mask',
    [
        (
            [],
            torch.tensor([5, 6]),
            torch.ones(2, dtype=torch.bool),
            1.0,
            torch.tensor([5, 6]),
            torch.tensor([False, False]),
        ),
        (
            [50, 51],
            torch.tensor([5, 6]),
            torch.zeros(2, dtype=torch.bool),
            1.0,
            torch.tensor([5, 6]),
            torch.tensor([False, False]),
        ),
    ],
)
def test_replace_special_tokens(
    mock_model,
    mock_tokenizer,
    device,
    source_tokens,
    next_tokens,
    can_replace,
    replace_prob,
    expected_tokens,
    expected_mask,
):
    """Test token replacement under various conditions."""
    generator = ExploreLLMGenerator(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=device,
        source_tokens=source_tokens,
        target_tokens=[10, 11],
    )
    next_tokens = next_tokens.to(device)
    can_replace = can_replace.to(device)
    expected_tokens = expected_tokens.to(device)
    expected_mask = expected_mask.to(device)
    modified_tokens, replace_mask = generator._replace_special_tokens(
        next_tokens, can_replace, replace_prob=replace_prob
    )
    assert torch.equal(modified_tokens, expected_tokens)
    assert torch.equal(replace_mask, expected_mask)


def test_replace_special_tokens_with_eligible_sequences(generator, device):
    """Test token replacement for eligible sequences with fixed random seed."""
    torch.manual_seed(42)
    next_tokens = torch.tensor([50, 51, 51, 6], device=device)
    can_replace = torch.tensor(
        [True, False, True, False], dtype=torch.bool, device=device
    )
    modified_tokens, replace_mask = generator._replace_special_tokens(
        next_tokens, can_replace, replace_prob=1.0
    )
    assert modified_tokens[0] in generator.target_tokens
    assert modified_tokens[1] == 51
    assert modified_tokens[2] in generator.target_tokens
    assert modified_tokens[3] == 6
    assert torch.equal(
        replace_mask, torch.tensor([True, False, True, False], device=device)
    )


def test_replace_special_tokens_with_probability(generator, device):
    """Test token replacement with probability less than 1.0."""
    torch.manual_seed(42)
    next_tokens = torch.full((100,), 50, device=device)
    can_replace = torch.ones(100, dtype=torch.bool, device=device)
    modified_tokens, replace_mask = generator._replace_special_tokens(
        next_tokens, can_replace, replace_prob=0.5
    )
    replacements = replace_mask.sum().item()
    assert 0 < replacements < 100
    for i in range(100):
        if replace_mask[i]:
            assert modified_tokens[i] in generator.target_tokens
        else:
            assert modified_tokens[i] == 50


# Replacement Eligibility Tests
def test_determine_replacement_eligibility_no_source_tokens(
    mock_model, mock_tokenizer, device
):
    """Test eligibility when no source tokens are defined."""
    generator = ExploreLLMGenerator(
        model=mock_model, tokenizer=mock_tokenizer, device=device
    )
    generated_ids = torch.randint(0, 100, (2, 10), device=device)
    next_tokens = torch.tensor([5, 6], device=device)
    replacement_counts = torch.zeros(2, device=device)
    result = generator._determine_replacement_eligibility(
        generated_ids, next_tokens, replacement_counts, replace_max_per_seq=3
    )
    assert result.shape == (2,)
    assert not result.any()


def test_determine_replacement_eligibility_tokens_not_in_source(
    generator, device
):
    """Test eligibility when tokens are not in source_tokens."""
    generated_ids = torch.randint(0, 100, (2, 10), device=device)
    next_tokens = torch.tensor([5, 6], device=device)
    replacement_counts = torch.zeros(2, device=device)
    result = generator._determine_replacement_eligibility(
        generated_ids, next_tokens, replacement_counts, replace_max_per_seq=3
    )
    assert not result.any()


def test_determine_replacement_eligibility_full_conditions(generator, device):
    """Test full eligibility conditions including pattern check and max replacements."""
    generated_ids = torch.tensor(
        [[1, 2, 3, 4, 5], [27, 29, 31, 5, 6], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        device=device,
    )
    next_tokens = torch.tensor([50, 50, 50, 50], device=device)
    replacement_counts = torch.tensor([0, 0, 3, 2], device=device)
    result = generator._determine_replacement_eligibility(
        generated_ids, next_tokens, replacement_counts, replace_max_per_seq=3
    )
    assert torch.equal(
        result, torch.tensor([True, False, False, True], device=device)
    )


def test_determine_replacement_eligibility_with_correctness(generator, device):
    """Test eligibility with correctness callback."""
    generated_ids = torch.randint(0, 100, (3, 10), device=device)
    next_tokens = torch.tensor([50, 50, 50], device=device)
    replacement_counts = torch.zeros(3, device=device)
    generator.tokenizer.batch_decode.return_value = [
        'This has an error',
        'This is fine',
        'Another error text',
    ]
    result = generator._determine_replacement_eligibility(
        generated_ids,
        next_tokens,
        replacement_counts,
        replace_max_per_seq=3,
        correctness_callback=mock_correctness_callback,
    )
    assert torch.equal(result, torch.tensor([True, False, True], device=device))


# Sequence Update Tests
def test_update_sequences_no_eos(generator, device):
    """Test sequence updating without EOS tokens."""
    input_ids = torch.randint(0, 100, (2, 5), device=device)
    attention_mask = torch.ones(2, 5, device=device)
    next_tokens = torch.tensor([7, 8], device=device)
    unfinished_sequences = torch.ones(2, device=device)
    new_input_ids, new_attention_mask, new_unfinished = (
        generator._update_sequences(
            input_ids,
            attention_mask,
            next_tokens,
            unfinished_sequences,
            eos_token_id=None,
            pad_token_id=0,
        )
    )
    assert new_input_ids.shape == (2, 6)
    assert torch.equal(new_input_ids[:, -1], next_tokens)
    assert torch.equal(new_unfinished, unfinished_sequences)


def test_update_sequences_with_eos(generator, device):
    """Test sequence updating with EOS tokens."""
    input_ids = torch.randint(0, 100, (3, 5), device=device)
    attention_mask = torch.ones(3, 5, device=device)
    next_tokens = torch.tensor([7, 50, 8], device=device)
    unfinished_sequences = torch.ones(3, device=device)
    new_input_ids, new_attention_mask, new_unfinished = (
        generator._update_sequences(
            input_ids,
            attention_mask,
            next_tokens,
            unfinished_sequences,
            eos_token_id=50,
            pad_token_id=0,
        )
    )
    assert torch.equal(new_unfinished, torch.tensor([1, 0, 1], device=device))


def test_update_sequences_already_finished(generator, device):
    """Test sequence updating with already finished sequences."""
    input_ids = torch.randint(0, 100, (3, 5), device=device)
    attention_mask = torch.ones(3, 5, device=device)
    next_tokens = torch.tensor([7, 8, 9], device=device)
    unfinished_sequences = torch.tensor([0, 1, 1], device=device)
    new_input_ids, new_attention_mask, new_unfinished = (
        generator._update_sequences(
            input_ids,
            attention_mask,
            next_tokens,
            unfinished_sequences,
            eos_token_id=50,
            pad_token_id=0,
        )
    )
    assert new_input_ids[0, -1].item() == 0
    assert new_input_ids[1, -1].item() == 8


# Sampling Tests
def test_uniform_sampling(generator, device):
    """Test uniform sampling from top-k tokens."""
    logits = torch.randn(100, 1000, device=device)
    torch.manual_seed(42)
    sampled_tokens = generator._uniform_sampling(logits, top_k=10)
    top_k_indices = torch.topk(logits, k=10, dim=-1).indices
    assert sampled_tokens.shape == (100,)
    for i in range(100):
        assert sampled_tokens[i] in top_k_indices[i]


def test_temperature_sampling(generator, device):
    """Test sampling with temperature, top-k, and top-p."""
    logits = torch.randn(2, 100, device=device)
    temperatures = torch.tensor([0.7, 0.0], device=device)
    torch.manual_seed(42)
    sampled_tokens = generator._sampling(
        logits, temperatures, top_p=0.9, top_k=50
    )
    assert sampled_tokens.shape == (2,)
    assert sampled_tokens[1].item() == logits[1].argmax().item()


# Repetition Penalty Tests
@pytest.mark.parametrize(
    'penalty, expected_effect',
    [(1.0, lambda x: x), (1.5, lambda x: x / 1.5 if x > 0 else x * 1.5)],
)
def test_repetition_penalty(generator, device, penalty, expected_effect):
    """Test repetition penalty application."""
    input_ids = torch.tensor(
        [[10, 20, 30, 10, 20], [5, 15, 25, 35, 45]], device=device
    )
    logits = torch.zeros(2, 100, device=device)
    logits[0, 10], logits[0, 20], logits[0, 40] = 5.0, -2.0, 3.0
    logits[1, 5] = 2.0
    modified_logits = generator._apply_repetition_penalty(
        logits, input_ids, penalty=penalty
    )
    assert modified_logits[0, 10].item() == pytest.approx(expected_effect(5.0))
    assert modified_logits[0, 20].item() == pytest.approx(expected_effect(-2.0))
    assert modified_logits[0, 40].item() == 3.0


# Generate Method Tests
def test_generate_basic_functionality(generator, device):
    """Test the basic functionality of the generate method."""
    input_ids = torch.randint(0, 100, (2, 5), device=device)
    attention_mask = torch.ones(2, 5, device=device)
    temperature = torch.tensor([0.7, 0.0], device=device)
    with patch.object(generator, '_sampling') as mock_sampling:
        mock_sampling.side_effect = [
            torch.tensor([10, 20], device=device),
            torch.tensor([30, 50], device=device),
            torch.tensor([40, 0], device=device),
        ]
        output = generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
            pad_token_id=0,
            eos_token_id=50,
            max_new_tokens=3,
            top_p=0.9,
            top_k=50,
        )
        assert isinstance(output, GenerateDecoderOnlyOutput)
        assert output.sequences.shape == (2, 8)
        assert torch.equal(output.sequences[:, :5], input_ids)


# Edge Cases
def test_top_k_zero_handling(generator, device):
    """Test handling of top_k=0."""
    logits = torch.randn(2, 100, device=device)
    sampled_tokens = generator._sampling(
        logits, torch.tensor([0.7, 0.7], device=device), top_p=1.0, top_k=0
    )
    assert sampled_tokens.shape == (2,)


def test_uniform_sampling_k_exceeds_vocab(generator, device):
    """Test uniform sampling when k exceeds vocabulary size."""
    logits = torch.randn(2, 50, device=device)
    sampled_tokens = generator._uniform_sampling(logits, top_k=100)
    assert sampled_tokens.shape == (2,)
    assert (sampled_tokens < 50).all()


def test_zero_batch_handling(generator, device):
    """Test handling of input with zero batch size."""
    input_ids = torch.randint(0, 100, (0, 5), device=device)
    attention_mask = torch.ones(0, 5, device=device)
    with pytest.raises(Exception):
        generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=0.7,
            pad_token_id=0,
            eos_token_id=50,
            max_new_tokens=3,
        )
