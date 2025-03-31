from unittest.mock import (  # Use MagicMock for flexibility
    MagicMock,
    Mock,
    call,
    patch,
)

import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    from rl4llm.generations.stochastic_llm_generator import (
        GenerateDecoderOnlyOutput,
        StochasticLLMGenerator,
    )
except ImportError:
    pytest.skip(
        'Skipping tests because stochastic_llm_generator module not found.',
        allow_module_level=True,
    )


# Fixtures for common test setup
@pytest.fixture
def mock_model():
    model = Mock(spec=PreTrainedModel)
    # Configure the model's forward method to return meaningful outputs
    model.forward.return_value = Mock(
        logits=torch.randn(2, 1, 100),  # [batch_size, seq_len, vocab_size]
        past_key_values=tuple(
            tuple(torch.randn(2, 4, 1, 32) for _ in range(2)) for _ in range(4)
        ),
    )
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.batch_decode.return_value = ['Test text 1', 'Test text 2']
    tokenizer.eos_token_id = 50
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def generator(mock_model, mock_tokenizer, device):
    source_tokens = [50, 51]  # Example tokens to replace
    target_tokens = [10, 11]  # Example replacement tokens
    prevent_patterns = [[27, 29, 31]]  # Example pattern to prevent replacement

    llm_generator = StochasticLLMGenerator(
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=device,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        prevent_patterns=prevent_patterns,
    )

    batch_size = 2
    seq_len = 5
    vocab_size = 100

    # Create a mock for the model's forward pass
    mock_model = MagicMock()
    llm_generator.model = mock_model

    # Define what the model should return for each forward pass
    vocab_logits = torch.zeros(batch_size, 1, vocab_size, device=device)
    mock_outputs = [
        MockModelOutput(logits=vocab_logits, past_key_values=('mock_past',)),
        MockModelOutput(logits=vocab_logits, past_key_values=('mock_past',)),
        MockModelOutput(logits=vocab_logits, past_key_values=('mock_past',)),
    ]
    mock_model.side_effect = mock_outputs

    return llm_generator


@pytest.fixture
def sample_input_tensors(device):
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    return {
        'input_ids': torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        ),
        'attention_mask': torch.ones(
            batch_size, seq_len, device=device, dtype=torch.long
        ),
        'logits': torch.randn(batch_size, vocab_size, device=device),
        'temperature': torch.tensor(
            [0.7, 0.0], device=device
        ),  # Mix of sampling and greedy
    }


# Helper function for correctness callback tests
def mock_correctness_callback(text):
    # Example callback that marks text containing "error" as incorrect
    return 0.0 if 'error' in text else 1.0


# Core functionality tests
class TestExploreLLMGeneratorInitialization:
    def test_initialization(self, mock_model, mock_tokenizer, device):
        """Test that the generator initializes with correct attributes."""
        source_tokens = [50, 51]
        target_tokens = [10, 11]
        prevent_patterns = [[27, 29, 31]]

        generator = StochasticLLMGenerator(
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
        assert torch.equal(
            generator.source_tokens_tensor,
            torch.tensor(source_tokens, device=device),
        )

    def test_initialization_with_defaults(
        self, mock_model, mock_tokenizer, device
    ):
        """Test initialization with default values."""
        generator = StochasticLLMGenerator(
            model=mock_model, tokenizer=mock_tokenizer, device=device
        )

        assert generator.source_tokens == []
        assert generator.target_tokens == []
        assert generator.prevent_patterns == []
        assert generator.source_tokens_tensor is None


class TestPatternDetection:
    def test_has_pattern_with_match(self, generator, device):
        """Test pattern detection when a match exists."""
        batch_size = 2
        pattern = [5, 7, 9]
        # Create input with the pattern at different positions
        input_ids = torch.tensor(
            [
                [1, 2, 3, 5, 7, 9, 10],  # Contains pattern
                [5, 7, 9, 11, 12, 13, 14],  # Contains pattern at the start
            ],
            device=device,
        )

        result = generator._has_pattern(input_ids, pattern)

        assert result.shape == (batch_size,)
        assert result.all()  # Both sequences should match

    def test_has_pattern_without_match(self, generator, device):
        """Test pattern detection when no match exists."""
        batch_size = 2
        pattern = [5, 7, 9]
        # Create input without the pattern
        input_ids = torch.tensor(
            [
                [1, 2, 3, 5, 8, 9, 10],  # No match (middle value differs)
                [5, 7, 10, 11, 12, 13, 14],  # No match (last value differs)
            ],
            device=device,
        )

        result = generator._has_pattern(input_ids, pattern)

        assert result.shape == (batch_size,)
        assert not result.any()  # No sequence should match

    def test_has_pattern_with_partial_match(self, generator, device):
        """Test pattern detection with some sequences matching."""
        batch_size = 3
        pattern = [5, 7, 9]
        # Create input with mixed matching
        input_ids = torch.tensor(
            [
                [1, 2, 3, 5, 7, 9, 10],  # Contains pattern
                [5, 8, 9, 11, 12, 13, 14],  # No match
                [1, 2, 3, 4, 5, 7, 9],  # Contains pattern at the end
            ],
            device=device,
        )

        result = generator._has_pattern(input_ids, pattern)

        assert result.shape == (batch_size,)
        assert result[0].item() is True
        assert result[1].item() is False
        assert result[2].item() is True

    def test_has_pattern_with_sequence_shorter_than_pattern(
        self, generator, device
    ):
        """Test pattern detection when sequence is shorter than pattern."""
        pattern = [5, 7, 9]
        # Create input shorter than pattern
        input_ids = torch.tensor(
            [
                [1, 2],  # Shorter than pattern
                [5, 7],  # Shorter than pattern
            ],
            device=device,
        )

        result = generator._has_pattern(input_ids, pattern)

        assert not result.any()  # None should match


class TestReplacementPatterns:
    def test_check_replacement_patterns_no_patterns(self, device):
        """Test replacement pattern checking when no patterns are defined."""
        mock_model = Mock(spec=PreTrainedModel)
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)

        # Create generator with no prevention patterns
        generator = StochasticLLMGenerator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=device,
            prevent_patterns=[],
        )

        batch_size = 2
        generated_ids = torch.randint(0, 100, (batch_size, 10), device=device)

        result = generator._check_replacement_patterns(generated_ids)

        assert result.shape == (batch_size,)
        assert result.all()  # All should be allowed (no patterns to prevent)

    def test_check_replacement_patterns_with_matches(self, generator, device):
        """Test replacement pattern checking with matching patterns."""
        batch_size = 3
        # Set a simple pattern for testing
        generator.prevent_patterns = [[5, 7, 9]]

        # Create input with the pattern in some sequences
        generated_ids = torch.tensor(
            [
                [
                    1,
                    2,
                    3,
                    5,
                    7,
                    9,
                    10,
                ],  # Contains pattern (should NOT be allowed)
                [5, 8, 9, 11, 12, 13, 14],  # No match (should be allowed)
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    7,
                    9,
                ],  # Contains pattern (should NOT be allowed)
            ],
            device=device,
        )

        result = generator._check_replacement_patterns(generated_ids)

        assert result.shape == (batch_size,)
        assert not result[0].item()  # Not allowed due to pattern match
        assert result[1].item()  # Allowed (no pattern match)
        assert not result[2].item()  # Not allowed due to pattern match

    def test_check_replacement_patterns_multiple_patterns(self, device):
        """Test with multiple prevention patterns."""
        mock_model = Mock(spec=PreTrainedModel)
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)

        # Create generator with multiple prevention patterns
        generator = StochasticLLMGenerator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=device,
            prevent_patterns=[[1, 2, 3], [5, 6, 7]],
        )

        batch_size = 4
        # Create input with different patterns
        generated_ids = torch.tensor(
            [
                # Contains first pattern (should NOT be allowed)
                [1, 2, 3, 4, 5],
                # Contains second pattern (should NOT be allowed)
                [4, 5, 6, 7, 8],
                # No match (should be allowed)
                [1, 3, 5, 7, 9],
                # Contains both patterns (should NOT be allowed)
                [5, 6, 7, 1, 2],
            ],
            device=device,
        )

        result = generator._check_replacement_patterns(generated_ids)

        assert result.shape == (batch_size,)
        assert not result[0].item()  # Not allowed (matches first pattern)
        assert not result[1].item()  # Not allowed (matches second pattern)
        assert result[2].item()  # Allowed (no pattern match)
        assert not result[3].item()  # Not allowed (matches both patterns)


class TestCorrectnessCheck:
    def test_check_correctness_no_callback(self, generator, device):
        """Test correctness checking with no callback provided."""
        batch_size = 2
        generated_ids = torch.randint(0, 100, (batch_size, 10), device=device)
        can_replace = torch.ones(batch_size, dtype=torch.bool, device=device)

        result = generator._check_correctness(generated_ids, can_replace, None)

        assert result.shape == (batch_size,)
        assert (
            not result.any()
        )  # All should be correct (no callback to say otherwise)

    def test_check_correctness_with_callback(
        self, generator, device, monkeypatch
    ):
        """Test correctness checking with callback that marks some as incorrect."""
        batch_size = 3
        generated_ids = torch.randint(0, 100, (batch_size, 10), device=device)
        can_replace = torch.tensor(
            [True, True, False], device=device
        )  # One can't be replaced

        # Mock the tokenizer.batch_decode to return controlled test strings
        generator.tokenizer.batch_decode.return_value = [
            'This has an error',  # Should be marked incorrect
            'This is fine',  # Should be marked correct
        ]

        result = generator._check_correctness(
            generated_ids, can_replace, mock_correctness_callback
        )

        assert result.shape == (batch_size,)
        assert result[0].item() is True  # Incorrect (contains "error")
        assert result[1].item() is False  # Correct (doesn't contain "error")
        assert result[2].item() is False  # Not checked due to can_replace=False

    def test_check_correctness_callback_exception(self, generator, device):
        """Test handling of exceptions in the correctness callback."""
        batch_size = 2
        generated_ids = torch.randint(0, 100, (batch_size, 10), device=device)
        can_replace = torch.ones(batch_size, dtype=torch.bool, device=device)

        def failing_callback(text):
            if text == 'Test text 1':
                raise ValueError('Test exception')
            return 0.5  # Half correct

        # Should handle the exception gracefully and continue
        with patch('builtins.print') as mock_print:
            result = generator._check_correctness(
                generated_ids, can_replace, failing_callback
            )

            # Verify the warning was printed
            assert mock_print.call_count == 1
            assert 'Warning' in mock_print.call_args[0][0]

        # The sequence that raised an exception should be marked as correct (false=correct)
        assert result[0].item() is False
        # The other sequence should be marked based on the callback result (0.5 < 1.0, so incorrect)
        assert result[1].item() is True


class TestTokenReplacement:
    def test_replace_special_tokens_no_source_tokens(self, device):
        """Test token replacement when no source tokens are defined."""
        mock_model = Mock(spec=PreTrainedModel)
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)

        # Create generator with no source tokens
        generator = StochasticLLMGenerator(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device=device,
            source_tokens=[],
            target_tokens=[10, 11],
        )

        batch_size = 2
        next_tokens = torch.tensor([5, 6], device=device)
        can_replace = torch.ones(batch_size, dtype=torch.bool, device=device)

        modified_tokens, replace_mask = generator._replace_special_tokens(
            next_tokens, can_replace, replace_prob=1.0
        )

        # No replacements should happen without source tokens
        assert torch.equal(modified_tokens, next_tokens)
        assert not replace_mask.any()

    def test_replace_special_tokens_no_eligible_sequences(
        self, generator, device
    ):
        """Test token replacement when no sequences are eligible."""
        batch_size = 2
        next_tokens = torch.tensor(
            [5, 6], device=device
        )  # Not in source_tokens
        can_replace = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )  # None eligible

        modified_tokens, replace_mask = generator._replace_special_tokens(
            next_tokens, can_replace, replace_prob=1.0
        )

        # No replacements should happen
        assert torch.equal(modified_tokens, next_tokens)
        assert not replace_mask.any()

    def test_replace_special_tokens_with_eligible_sequences(
        self, generator, device
    ):
        """Test token replacement for eligible sequences with fixed random seed."""
        # Fix the random seed for reproducibility
        torch.manual_seed(42)

        # Some tokens are in source_tokens (50, 51)
        next_tokens = torch.tensor([50, 51, 51, 6], device=device)
        can_replace = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device=device
        )

        # With 100% replacement probability
        modified_tokens, replace_mask = generator._replace_special_tokens(
            next_tokens, can_replace, replace_prob=1.0
        )

        # Only tokens in source_tokens should be replaced
        assert (
            modified_tokens[0] in generator.target_tokens
        )  # 50 should be replaced
        assert modified_tokens[1] == 51  # 51 should stay the same
        assert (
            modified_tokens[2] in generator.target_tokens
        )  # 51 should be replaced
        assert modified_tokens[3] == 6  # 6 should stay the same

        # Check the replacement mask
        assert replace_mask[0].item() is True  # Replacement happened
        assert replace_mask[1].item() is False  # No replacement
        assert replace_mask[2].item() is True  # Replacement happened
        assert replace_mask[3].item() is False  # No replacement

    def test_replace_special_tokens_with_probability(self, generator, device):
        """Test token replacement with probability less than 1.0."""
        # Run the test multiple times with different seeds to ensure
        # we get both replacement and non-replacement cases
        batch_size = 100  # Large batch to ensure we observe both outcomes
        next_tokens = torch.full(
            (batch_size,), 50, device=device
        )  # All are source tokens
        can_replace = torch.ones(batch_size, dtype=torch.bool, device=device)

        # With 50% replacement probability
        torch.manual_seed(42)  # Fixed seed for reproducibility
        modified_tokens, replace_mask = generator._replace_special_tokens(
            next_tokens, can_replace, replace_prob=0.5
        )

        # Some should be replaced and some not
        replacements = replace_mask.sum().item()
        assert 0 < replacements < batch_size

        # Tokens where replace_mask is True should be from target_tokens
        for i in range(batch_size):
            if replace_mask[i]:
                assert modified_tokens[i] in generator.target_tokens
            else:
                assert modified_tokens[i] == 50


class TestReplacementEligibility:
    def test_determine_replacement_eligibility_no_source_tokens(self, device):
        """Test eligibility when no source tokens are defined."""
        mock_model = Mock(spec=PreTrainedModel)
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)

        # Create generator with no source tokens
        generator = StochasticLLMGenerator(
            model=mock_model, tokenizer=mock_tokenizer, device=device
        )

        batch_size = 2
        generated_ids = torch.randint(0, 100, (batch_size, 10), device=device)
        next_tokens = torch.tensor([5, 6], device=device)
        replacement_counts = torch.zeros(batch_size, device=device)

        result = generator._determine_replacement_eligibility(
            generated_ids,
            next_tokens,
            replacement_counts,
            explore_max_replacements=3,
        )

        # No replacements should be eligible without source tokens
        assert result.shape == (batch_size,)
        assert not result.any()

    def test_determine_replacement_eligibility_tokens_not_in_source(
        self, generator, device
    ):
        """Test eligibility when tokens are not in source_tokens."""
        batch_size = 2
        generated_ids = torch.randint(0, 100, (batch_size, 10), device=device)
        next_tokens = torch.tensor(
            [5, 6], device=device
        )  # Not in source_tokens
        replacement_counts = torch.zeros(batch_size, device=device)

        result = generator._determine_replacement_eligibility(
            generated_ids,
            next_tokens,
            replacement_counts,
            explore_max_replacements=3,
        )

        # No eligibility when tokens aren't in source_tokens
        assert not result.any()

    def test_determine_replacement_eligibility_full_conditions(
        self, generator, device
    ):
        """Test full eligibility conditions including pattern check and max replacements."""
        batch_size = 4
        # Create some sequences with and without the prevent pattern
        generated_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # No prevent pattern
                [27, 29, 31, 5, 6],  # Has prevent pattern
                [1, 2, 3, 4, 5],  # No prevent pattern
                [1, 2, 3, 4, 5],  # No prevent pattern
            ],
            device=device,
        )
        # All tokens are in source_tokens
        next_tokens = torch.tensor([50, 50, 50, 50], device=device)
        # Some are at max replacements
        replacement_counts = torch.tensor([0, 0, 3, 2], device=device)

        result = generator._determine_replacement_eligibility(
            generated_ids,
            next_tokens,
            replacement_counts,
            explore_max_replacements=3,
        )

        assert result[0].item() is True  # Eligible: no pattern, below max
        assert result[1].item() is False  # Not eligible: has prevent pattern
        assert result[2].item() is False  # Not eligible: at max replacements
        assert result[3].item() is True  # Eligible: no pattern, below max

    def test_determine_replacement_eligibility_with_correctness(
        self, generator, device
    ):
        """Test eligibility with correctness callback."""
        batch_size = 3
        generated_ids = torch.randint(0, 100, (batch_size, 10), device=device)
        next_tokens = torch.tensor(
            [50, 50, 50], device=device
        )  # All in source_tokens
        replacement_counts = torch.zeros(batch_size, device=device)

        # Mock the tokenizer.batch_decode to return controlled test strings
        generator.tokenizer.batch_decode.return_value = [
            'This has an error',  # Should be marked incorrect (eligible)
            'This is fine',  # Should be marked correct (not eligible)
            'Another error text',  # Should be marked incorrect (eligible)
        ]

        result = generator._determine_replacement_eligibility(
            generated_ids,
            next_tokens,
            replacement_counts,
            explore_max_replacements=3,
            correctness_callback=mock_correctness_callback,
        )

        assert result[0].item() is True  # Eligible: incorrect
        assert result[1].item() is False  # Not eligible: correct
        assert result[2].item() is True  # Eligible: incorrect


class TestSequenceUpdate:
    def test_update_sequences_no_eos(self, generator, device):
        """Test sequence updating without EOS tokens."""
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        next_tokens = torch.tensor([7, 8], device=device)
        unfinished_sequences = torch.ones(batch_size, device=device)

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

        # Check dimensions
        assert new_input_ids.shape == (batch_size, seq_len + 1)
        assert new_attention_mask.shape == (batch_size, seq_len + 1)
        assert new_unfinished.shape == (batch_size,)

        # Check content
        assert torch.equal(new_input_ids[:, :-1], input_ids)
        assert torch.equal(new_input_ids[:, -1], next_tokens)
        assert torch.equal(new_attention_mask[:, :-1], attention_mask)
        assert torch.equal(
            new_attention_mask[:, -1], torch.ones(batch_size, device=device)
        )
        assert torch.equal(new_unfinished, unfinished_sequences)

    def test_update_sequences_with_eos(self, generator, device):
        """Test sequence updating with EOS tokens."""
        batch_size = 3
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        next_tokens = torch.tensor(
            [7, 50, 8], device=device
        )  # Second token is EOS
        unfinished_sequences = torch.ones(batch_size, device=device)
        eos_token_id = 50

        new_input_ids, new_attention_mask, new_unfinished = (
            generator._update_sequences(
                input_ids,
                attention_mask,
                next_tokens,
                unfinished_sequences,
                eos_token_id=eos_token_id,
                pad_token_id=0,
            )
        )

        # Check updated unfinished sequences
        assert new_unfinished[0].item() == 1  # Still unfinished
        assert new_unfinished[1].item() == 0  # Finished (generated EOS)
        assert new_unfinished[2].item() == 1  # Still unfinished

    def test_update_sequences_already_finished(self, generator, device):
        """Test sequence updating with already finished sequences."""
        batch_size = 3
        seq_len = 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        next_tokens = torch.tensor([7, 8, 9], device=device)
        # First sequence already finished
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

        # Check that padding was applied to finished sequences
        assert new_input_ids[0, -1].item() == 0  # Padding for finished sequence
        assert new_input_ids[1, -1].item() == 8  # Normal token for unfinished
        assert new_input_ids[2, -1].item() == 9  # Normal token for unfinished

        # Check attention mask - should be 0 for padded position
        assert new_attention_mask[0, -1].item() == 0  # No attention for padding
        assert new_attention_mask[1, -1].item() == 1  # Normal attention
        assert new_attention_mask[2, -1].item() == 1  # Normal attention


class TestSampling:
    def test_uniform_sampling(self, generator, device):
        """Test uniform sampling from top-k tokens."""
        batch_size = 100  # Large batch to check distribution
        vocab_size = 1000
        top_k = 10
        logits = torch.randn(batch_size, vocab_size, device=device)

        # Get expected top-k indices for each batch
        top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1)

        # Sample with fixed seed
        torch.manual_seed(42)
        sampled_tokens = generator._uniform_sampling(logits, top_k)

        # Check shape
        assert sampled_tokens.shape == (batch_size,)

        # Verify each sampled token is in the top-k set for its batch
        for i in range(batch_size):
            assert sampled_tokens[i] in top_k_indices[i]

        # Check with k=1 (should return top token)
        top_1_sampled = generator._uniform_sampling(logits, 1)
        top_1_expected = logits.argmax(dim=-1)
        assert torch.equal(top_1_sampled, top_1_expected)

    def test_temperature_sampling(self, generator, device):
        """Test sampling with temperature, top-k and top-p."""
        batch_size = 2
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size, device=device)

        # Test with mixed temperatures: one for sampling, one for greedy
        temperatures = torch.tensor([0.7, 0.0], device=device)

        # Sample with fixed seed
        torch.manual_seed(42)
        sampled_tokens = generator._sampling(
            logits, temperatures, top_p=0.9, top_k=50
        )

        # Check shape
        assert sampled_tokens.shape == (batch_size,)

        # For zero temperature, should be greedy (same as argmax)
        assert sampled_tokens[1].item() == logits[1].argmax().item()

        # For non-zero temperature, should be within top-k
        top_k_values, top_k_indices = torch.topk(logits[0], k=50, dim=-1)
        assert sampled_tokens[0] in top_k_indices

        # Test with all zero temperatures
        all_zero_temp = torch.zeros(batch_size, device=device)
        greedy_tokens = generator._sampling(
            logits, all_zero_temp, top_p=0.9, top_k=50
        )
        assert torch.equal(greedy_tokens, logits.argmax(dim=-1))


class TestRepetitionPenalty:
    def test_repetition_penalty_no_penalty(self, generator, device):
        """Test with penalty=1.0 (no effect)."""
        batch_size = 2
        vocab_size = 100
        seq_len = 5

        logits = torch.randn(batch_size, vocab_size, device=device)
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )

        # With penalty=1.0, should return unchanged logits
        modified_logits = generator._apply_repetition_penalty(
            logits, input_ids, penalty=1.0
        )

        assert torch.equal(modified_logits, logits)

    def test_repetition_penalty_with_penalty(self, generator, device):
        """Test with penalty > 1.0."""
        batch_size = 2
        vocab_size = 100

        # Create repeating tokens in input_ids
        input_ids = torch.tensor(
            [
                [10, 20, 30, 10, 20],  # Repeats tokens 10, 20
                [5, 15, 25, 35, 45],  # No repeats
            ],
            device=device,
        )

        # Create logits with both positive and negative values
        logits = torch.zeros(batch_size, vocab_size, device=device)
        # Set some positive and negative values for repeated tokens
        logits[0, 10] = 5.0  # Positive logit for repeated token 10
        logits[0, 20] = -2.0  # Negative logit for repeated token 20
        logits[0, 40] = 3.0  # Positive logit for non-repeated token
        logits[1, 5] = 2.0  # Positive logit for repeated token in seq 2

        penalty = 1.5
        modified_logits = generator._apply_repetition_penalty(
            logits, input_ids, penalty=penalty
        )

        # Check penalties were applied correctly
        # Positive logits for repeated tokens should be divided by penalty
        assert modified_logits[0, 10].item() == pytest.approx(5.0 / penalty)

        # Negative logits for repeated tokens should be multiplied by penalty
        assert modified_logits[0, 20].item() == pytest.approx(-2.0 * penalty)

        # Non-repeated tokens should be unchanged
        assert modified_logits[0, 40].item() == 3.0

        # Tokens in second sequence should be affected based on that sequence
        assert modified_logits[1, 5].item() == pytest.approx(2.0 / penalty)

    def test_repetition_penalty_empty_sequence(self, generator, device):
        """Test with empty input sequence."""
        batch_size = 2
        vocab_size = 100

        logits = torch.randn(batch_size, vocab_size, device=device)
        # Empty sequence for first batch, normal for second
        input_ids = torch.tensor(
            [
                [],  # Empty
                [5, 15, 25],  # Normal
            ],
            device=device,
        )

        # Should handle empty sequence gracefully
        modified_logits = generator._apply_repetition_penalty(
            logits, input_ids, penalty=1.5
        )

        # First batch should be unchanged (no tokens to penalize)
        assert torch.equal(modified_logits[0], logits[0])
        # Second batch should have penalties applied


class TestTokenReplacementIntegration:
    def test_apply_token_replacement_no_conditions_met(self, generator, device):
        """Test when no conditions for replacement are met."""
        batch_size = 2
        seq_len = 5
        next_tokens = torch.tensor(
            [5, 6], device=device
        )  # Not in source_tokens
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        initial_seq_len = 2
        replacement_counts = torch.zeros(batch_size, device=device)

        modified_tokens, updated_counts = generator._apply_token_replacement(
            next_tokens,
            input_ids,
            initial_seq_len,
            replacement_counts,
            explore_max_replacements=3,
            explore_replace_prob=0.5,
            correctness_callback=None,
        )

        # No replacements should happen
        assert torch.equal(modified_tokens, next_tokens)
        assert torch.equal(updated_counts, replacement_counts)

    def test_apply_token_replacement_conditions_met(self, generator, device):
        """Test when conditions for replacement are met."""
        batch_size = 3
        seq_len = 10
        initial_seq_len = 5
        next_tokens = torch.tensor(
            [50, 5, 51], device=device
        )  # First and third in source_tokens
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        replacement_counts = torch.zeros(batch_size, device=device)

        # Mock the determine_replacement_eligibility to control output
        with patch.object(
            generator, '_determine_replacement_eligibility'
        ) as mock_determine:
            # First and third sequences are eligible
            mock_determine.return_value = torch.tensor(
                [True, False, True], device=device
            )

            # Mock replace_special_tokens to control output
            with patch.object(
                generator, '_replace_special_tokens'
            ) as mock_replace:
                # First token replaced, third not (random outcome)
                replaced_tokens = torch.tensor([10, 5, 51], device=device)
                replace_mask = torch.tensor([True, False, False], device=device)
                mock_replace.return_value = (replaced_tokens, replace_mask)

                modified_tokens, updated_counts = (
                    generator._apply_token_replacement(
                        next_tokens,
                        input_ids,
                        initial_seq_len,
                        replacement_counts,
                        explore_max_replacements=3,
                        explore_replace_prob=0.5,
                        correctness_callback=None,
                    )
                )

                # Verify the results
                assert torch.equal(modified_tokens, replaced_tokens)
                assert torch.equal(updated_counts, replace_mask.long())

                # Verify mocked functions were called correctly
                mock_determine.assert_called_once()
                mock_replace.assert_called_once()
                # Check the args for replace_special_tokens call
                _, args, _ = mock_replace.mock_calls[0]
                assert torch.equal(args[0], next_tokens)
                assert torch.equal(args[1], mock_determine.return_value)
                assert args[2] == 0.5  # explore_replace_prob


# Create a mock output class to mimic the model's output structure
from dataclasses import dataclass


@dataclass
class MockModelOutput:
    logits: torch.Tensor
    past_key_values: tuple = None


class TestGenerateMethod:
    batch_size = 2
    seq_len = 5
    vocab_size = 100

    def test_generate_basic_functionality(self, generator, device):
        """Test the basic functionality of the generate method."""

        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=device
        )
        temperature = torch.tensor([0.7, 0.0], device=device)

        # Mock sampling to return controlled tokens
        with patch.object(generator, '_sampling') as mock_sampling:
            # First return normal tokens, then EOS for second sequence
            mock_sampling.side_effect = [
                torch.tensor([10, 20], device=device),
                torch.tensor([30, 50], device=device),  # 50 is EOS
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

            # Check that the output has the correct type
            assert isinstance(output, GenerateDecoderOnlyOutput)
            # Check the shape considering 3 tokens were generated
            assert output.sequences.shape == (self.batch_size, self.seq_len + 3)
            # Check that initial input_ids are preserved
            assert torch.equal(output.sequences[:, : self.seq_len], input_ids)
            # Check generated tokens based on our mocked _sampling
            assert output.sequences[0, self.seq_len].item() == 10
            assert output.sequences[0, self.seq_len + 1].item() == 30
            assert output.sequences[0, self.seq_len + 2].item() == 40
            assert output.sequences[1, self.seq_len].item() == 20
            assert output.sequences[1, self.seq_len + 1].item() == 50  # EOS
            assert output.sequences[1, self.seq_len + 2].item() == 0  # Padding

            # Verify _sampling was called with expected arguments
            assert mock_sampling.call_count == 3
            # Check args for first call
            _, args, kwargs = mock_sampling.mock_calls[0]
            assert torch.is_tensor(kwargs['token_logits'])  # logits
            assert torch.equal(kwargs['temperature'], temperature)
            assert kwargs['top_p'] == 0.9
            assert kwargs['top_k'] == 50

    def test_generate_with_exploration(self, generator, device):
        """Test generation with exploration parameters."""

        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=device
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=device
        )
        temperature = 0.7

        # Mock _uniform_sampling and _sampling to control exploration behavior
        with patch.object(generator, '_uniform_sampling') as mock_uniform:
            with patch.object(generator, '_sampling') as mock_sampling:
                # Return different tokens for exploration vs. normal sampling
                mock_uniform.return_value = torch.tensor(
                    [15, 25], device=device
                )
                mock_sampling.return_value = torch.tensor(
                    [10, 20], device=device
                )

                output = generator.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=temperature,
                    pad_token_id=0,
                    eos_token_id=50,
                    max_new_tokens=3,
                    explore_start_steps=2,  # Explore for first 2 steps
                    explore_skip_n=0,
                    explore_top_k=20,
                )

                # Check that exploration was used for first two tokens
                assert mock_uniform.call_count == 2
                # And normal sampling for the third
                assert mock_sampling.call_count == 1

                # Verify exploration parameters were passed correctly
                _, args, kwargs = mock_uniform.mock_calls[0]
                assert kwargs['top_k'] == 20  # explore_top_k for first step

                # Second step should use a reduced top_k (0.8 * 20 = 16)
                _, args, kwargs = mock_uniform.mock_calls[1]
                assert kwargs['top_k'] == 16  # reduced explore_top_k

    def test_generate_with_token_replacement(self, generator, device):
        """Test generation with token replacement."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        temperature = 0.7

        # Mock _sampling and _apply_token_replacement
        with patch.object(generator, '_sampling') as mock_sampling:
            with patch.object(
                generator, '_apply_token_replacement'
            ) as mock_replace:
                # Return regular tokens first, then modified tokens
                mock_sampling.return_value = torch.tensor(
                    [10, 20], device=device
                )
                # First call: just return same tokens (no replacement)
                # Second call: return modified tokens
                # Third call: return same tokens again (no replacement)
                mock_replace.side_effect = [
                    (
                        torch.tensor([10, 20], device=device),
                        torch.zeros(batch_size, device=device),
                    ),
                    (
                        torch.tensor([15, 25], device=device),
                        torch.ones(batch_size, device=device),
                    ),
                ]

                output = generator.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=temperature,
                    pad_token_id=0,
                    eos_token_id=50,
                    max_new_tokens=3,
                    explore_replace_prob=0.5,
                    explore_max_replacements=2,
                )

                # Verify replacement was called for each step
                assert mock_replace.call_count >= 2

                # Check generated tokens reflect the replacement in second step
                assert output.sequences[0, seq_len + 1].item() == 10  # Replaced
                assert output.sequences[1, seq_len + 1].item() == 20  # Replaced

                assert (
                    output.sequences[0, seq_len + 2].item() == 15
                )  # Not replaced
                assert (
                    output.sequences[1, seq_len + 2].item() == 25
                )  # Not replaced

    def test_generate_early_stopping(self, generator, device):
        """Test generation with early stopping when all sequences are finished."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        temperature = 0.7

        # Mock _sampling to return EOS tokens in the second step
        with patch.object(generator, '_sampling') as mock_sampling:
            mock_sampling.side_effect = [
                torch.tensor([10, 20], device=device),
                torch.tensor([50, 50], device=device),  # Both are EOS
                # This should never be called due to early stopping
                torch.tensor([30, 40], device=device),
            ]

            output = generator.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                pad_token_id=0,
                eos_token_id=50,
                max_new_tokens=5,  # Allow more than we'll generate
            )

            # Should only call _sampling twice due to early stopping
            assert mock_sampling.call_count == 2

            # Check shape - should only have added 2 tokens
            assert output.sequences.shape == (batch_size, seq_len + 2)

            # Check generated tokens - second token should be EOS for both
            assert output.sequences[0, seq_len + 1].item() == 50  # EOS
            assert output.sequences[1, seq_len + 1].item() == 50  # EOS

    def test_generate_with_repetition_penalty(self, generator, device):
        """Test generation with repetition penalty."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        temperature = 0.7

        # Mock _apply_repetition_penalty and _sampling
        with patch.object(
            generator, '_apply_repetition_penalty'
        ) as mock_penalty:
            with patch.object(generator, '_sampling') as mock_sampling:
                # Set up controlled returns
                mock_penalty.side_effect = (
                    lambda logits, input_ids, penalty: logits * 2
                )  # Just double logits for testing
                mock_sampling.return_value = torch.tensor(
                    [10, 20], device=device
                )

                generator.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    temperature=temperature,
                    pad_token_id=0,
                    eos_token_id=50,
                    max_new_tokens=3,
                    repetition_penalty=1.5,
                )

                # Verify penalty was applied for each generation step
                assert mock_penalty.call_count == 3

                # Check repetition_penalty was passed correctly
                _, _, kwargs = mock_penalty.mock_calls[0]
                assert kwargs['penalty'] == 1.5

    def test_generate_with_temperature_variations(self, generator, device):
        """Test generation with various temperature formats."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Mock _sampling to isolate temperature handling
        with patch.object(generator, '_sampling') as mock_sampling:
            mock_sampling.return_value = torch.tensor([10, 20], device=device)

            # Test with float temperature
            generator.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=0.8,
                pad_token_id=0,
                eos_token_id=50,
                max_new_tokens=1,
            )

            # Check temperature was converted to tensor
            _, args, kwargs = mock_sampling.mock_calls[0]
            assert torch.is_tensor(kwargs['temperature'])
            assert kwargs['temperature'].shape == (batch_size,)
            assert torch.allclose(
                kwargs['temperature'][0],
                torch.tensor(0.8),
                rtol=1e-5,
                atol=1e-8,
            )

            mock_sampling.reset_mock()

            # Test with list temperature
            generator.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=[0.7, 0.9],
                pad_token_id=0,
                eos_token_id=50,
                max_new_tokens=1,
            )

            # Check list was converted to tensor
            _, args, kwargs = mock_sampling.mock_calls[0]
            assert torch.is_tensor(kwargs['temperature'])
            assert kwargs['temperature'].shape == (batch_size,)
            assert torch.allclose(
                kwargs['temperature'][0],
                torch.tensor(0.7),
                rtol=1e-5,
                atol=1e-8,
            )
            assert torch.allclose(
                kwargs['temperature'][1],
                torch.tensor(0.9),
                rtol=1e-5,
                atol=1e-8,
            )


class TestEdgeCases:
    def test_top_k_zero_handling(self, generator, device):
        """Test handling of top_k=0."""
        batch_size = 2
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size, device=device)

        # With top_k=0, should use all logits (not filter)
        sampled_tokens = generator._sampling(
            logits, torch.tensor([0.7, 0.7], device=device), top_p=1.0, top_k=0
        )

        assert sampled_tokens.shape == (batch_size,)

    def test_uniform_sampling_k_exceeds_vocab(self, generator, device):
        """Test uniform sampling when k exceeds vocabulary size."""
        batch_size = 2
        vocab_size = 50
        logits = torch.randn(batch_size, vocab_size, device=device)

        # Request more tokens than available
        sampled_tokens = generator._uniform_sampling(logits, top_k=100)

        # Should handle gracefully and sample from all tokens
        assert sampled_tokens.shape == (batch_size,)
        # Each token should be valid (within vocab range)
        assert (sampled_tokens >= 0).all()
        assert (sampled_tokens < vocab_size).all()

    def test_zero_batch_handling(self, generator, device):
        """Test handling of input with zero batch size (should raise error)."""
        vocab_size = 100
        # Create tensors with 0 batch size
        input_ids = torch.randint(0, vocab_size, (0, 5), device=device)
        attention_mask = torch.ones(0, 5, device=device)

        # Should raise an assertion error or similar
        with pytest.raises(Exception):
            generator.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=0.7,
                pad_token_id=0,
                eos_token_id=50,
                max_new_tokens=3,
            )


# Integration tests - combining multiple aspects
class TestIntegration:
    def test_exploration_and_replacement(self, generator, device):
        """Test interaction between exploration and token replacement."""
        batch_size = 2
        seq_len = 5
        vocab_size = 100

        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Setup controlled mocks
        with patch.object(generator, '_uniform_sampling') as mock_uniform:
            with patch.object(generator, '_sampling') as mock_sampling:
                with patch.object(
                    generator, '_apply_token_replacement'
                ) as mock_replace:
                    # Return source tokens during exploration
                    mock_uniform.return_value = torch.tensor(
                        [50, 51], device=device
                    )  # Source tokens
                    mock_sampling.return_value = torch.tensor(
                        [10, 20], device=device
                    )
                    # Replace only during exploration
                    mock_replace.side_effect = lambda *args, **kwargs: (
                        (
                            torch.tensor([15, 16], device=device),
                            torch.ones(batch_size, device=device),
                        )
                        if args[0][0] in generator.source_tokens
                        else (args[0], torch.zeros(batch_size, device=device))
                    )

                    output = generator.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        temperature=0.7,
                        pad_token_id=0,
                        eos_token_id=100,  # Set high to prevent early stopping
                        max_new_tokens=3,
                        explore_start_steps=2,
                        explore_replace_prob=1.0,
                        explore_max_replacements=2,
                    )

                    # Exploration should be used for first 2 steps
                    assert mock_uniform.call_count == 2
                    # Sampling for the last step
                    assert mock_sampling.call_count == 1
                    # Replacement should be called for each step
                    assert mock_replace.call_count == 2

                    # Generated tokens should reflect replacements during exploration
                    assert (
                        output.sequences[0, seq_len].item() == 50
                    )  # Original prompt token
                    assert (
                        output.sequences[0, seq_len + 1].item() == 15
                    )  # Replaced from exploration
                    # Last token should be from normal sampling without replacement
                    assert output.sequences[0, seq_len + 2].item() == 10


# Run tests for specific functionality in isolation
class TestIsolatedFunctionality:
    def test_top_p_filtering(self, generator, device):
        """Test top-p (nucleus) sampling in isolation."""
        batch_size = 2
        vocab_size = 10  # Small vocab for easier verification

        # Create controlled logits with known probabilities
        logits = torch.zeros(batch_size, vocab_size, device=device)
        # First batch: tokens have equal probability
        logits[0] = torch.tensor(
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device
        )
        # Second batch: first token has very high probability
        logits[1] = torch.tensor(
            [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device
        )

        # Apply sampling with top_p = 0.5
        torch.manual_seed(42)  # For reproducibility
        sampled_tokens = generator._sampling(
            logits, torch.tensor([0.7, 0.7], device=device), top_p=0.5, top_k=0
        )

        # For first batch, only top 4 tokens (with prob=0.5) should be considered
        assert sampled_tokens[0] < 4

        # For second batch, only the top token should be considered (prob > 0.5)
        assert sampled_tokens[1].item() == 0

    def test_correctness_callback_integration(self, generator, device):
        """Test correctness callback integration with token replacement."""
        batch_size = 3
        seq_len = 10

        # Create input where some are eligible for replacement
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        next_tokens = torch.tensor(
            [50, 50, 50], device=device
        )  # All in source_tokens
        replacement_counts = torch.zeros(batch_size, device=device)

        # Mock tokenizer to return controlled texts
        generator.tokenizer.batch_decode.return_value = [
            'This has an error',  # incorrect
            'This is fine',  # correct
            'Another error text',  # incorrect
        ]

        # Mock _replace_special_tokens to verify input
        with patch.object(generator, '_replace_special_tokens') as mock_replace:
            mock_replace.return_value = (
                next_tokens,
                torch.zeros(batch_size, device=device),
            )

            generator._apply_token_replacement(
                next_tokens,
                input_ids,
                0,
                replacement_counts,
                explore_max_replacements=3,
                explore_replace_prob=1.0,
                correctness_callback=mock_correctness_callback,
            )

            # Verify only incorrect sequences were considered for replacement
            _, args, _ = mock_replace.mock_calls[0]
            expected_mask = torch.tensor([True, False, True], device=device)
            assert torch.equal(args[1], expected_mask)


# Parameterized tests for exploring combinations
@pytest.mark.parametrize(
    'top_k,top_p,temp',
    [
        (0, 1.0, 1.0),  # No top-k, no top-p, standard temp
        (50, 0.9, 0.7),  # Standard values
        (0, 0.5, 0.0),  # Top-p only, greedy
        (10, 1.0, 0.0),  # Top-k only, greedy
    ],
)
def test_sampling_combinations(generator, device, top_k, top_p, temp):
    """Test various combinations of sampling parameters."""
    batch_size = 2
    vocab_size = 100
    logits = torch.randn(batch_size, vocab_size, device=device)

    # Set temperature as tensor
    temperature = torch.full((batch_size,), temp, device=device)

    sampled_tokens = generator._sampling(
        logits, temperature, top_p=top_p, top_k=top_k
    )

    # Basic shape check
    assert sampled_tokens.shape == (batch_size,)

    # For greedy decoding, should return argmax
    if temp == 0.0:
        assert torch.equal(sampled_tokens, logits.argmax(dim=-1))
