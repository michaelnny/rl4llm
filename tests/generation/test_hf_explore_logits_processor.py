import warnings
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizer

from rl4llm.generation.hf_explore_processor import HfExploreLogitsProcessor

# --- Fixtures ---


@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer with a defined vocab size."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 100
    tokenizer.batch_decode = MagicMock(
        side_effect=lambda x, **kwargs: [f"decoded_{i}" for i in range(len(x))]
    )
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    return tokenizer


@pytest.fixture
def device():
    """Provides the torch device (CPU for testing)."""
    return torch.device('cpu')


@pytest.fixture
def default_config(mock_tokenizer):
    """Provides a default configuration dictionary for the processor."""
    return {
        'initial_seq_len': 5,
        'tokenizer': mock_tokenizer,
        'temperature': [1.0, 1.0],
        'group_size': 2,
        'explore_steps': 0,
        'explore_skip_n': 0,
        'explore_top_k': 20,
        'explore_decay': 0.9,
        'replace_source_tokens': None,
        'replace_target_tokens': None,
        'replace_prevent_patterns': None,
        'replace_prob': 0.0,
        'replace_max_per_seq': 0,
        'replace_boost_value': 100.0,
        'replace_check_top_n': 3,
        'correctness_callback': None,
    }


@pytest.fixture
def input_ids(default_config, device):
    """Provides sample input_ids tensor."""
    batch_size = default_config['group_size']
    seq_len = (
        default_config['initial_seq_len'] + 3
    )  # Simulate a few generation steps
    return torch.randint(
        2, 50, (batch_size, seq_len), device=device
    )  # Avoid pad/eos


@pytest.fixture
def scores(default_config, mock_tokenizer, device):
    """Provides sample scores tensor."""
    batch_size = default_config['group_size']
    vocab_size = mock_tokenizer.vocab_size
    return torch.randn((batch_size, vocab_size), device=device)


@pytest.fixture
def sequence_indices(default_config, device):
    """Provides sample sequence indices tensor matching group_size."""
    return torch.arange(default_config['group_size'], device=device)


# --- Initialization Tests ---


def test_initialization_valid(default_config):
    """Tests successful initialization with default valid parameters."""
    processor = HfExploreLogitsProcessor(**default_config)
    assert processor.initial_seq_len == default_config['initial_seq_len']
    assert processor.group_size == default_config['group_size']
    assert processor.temperature.shape == (default_config['group_size'],)
    assert torch.all(processor.temperature == 1.0)
    assert processor.replacement_counts.shape == (default_config['group_size'],)
    assert torch.all(processor.replacement_counts == 0)
    assert processor.current_step == 0
    assert processor.replace_max_per_seq == 0  # Disabled by default


@pytest.mark.parametrize(
    'param, value, error_type',
    [
        ('initial_seq_len', -1, ValueError),
        (
            'initial_seq_len',
            1.5,
            ValueError,
        ),  # Type enforced by annotation but check runtime
        ('group_size', 0, ValueError),
        ('group_size', -1, ValueError),
        ('temperature', -0.5, ValueError),
        ('temperature', [1.0, -0.1], ValueError),
        ('temperature', torch.tensor([1.0, -0.1]), ValueError),
        ('temperature', [1.0], ValueError),
        ('explore_steps', -1, ValueError),
        ('explore_skip_n', -1, ValueError),
        ('explore_top_k', 0, ValueError),
        ('explore_decay', 0.0, ValueError),
        ('explore_decay', 1.1, ValueError),
        ('replace_source_tokens', 'not_a_list', TypeError),
        ('replace_target_tokens', 'not_a_list', TypeError),
        ('replace_prevent_patterns', 'not_a_list', TypeError),
        ('replace_prevent_patterns', [[1], 'not_a_list'], TypeError),
        ('replace_prob', -0.1, ValueError),
        ('replace_prob', 1.1, ValueError),
        ('replace_max_per_seq', -1, ValueError),
        ('replace_boost_value', 'not_a_float', TypeError),
        ('replace_check_top_n', 0, ValueError),
    ],
)
def test_initialization_invalid_params(
    default_config, param, value, error_type
):
    """Tests initialization failure with various invalid parameters."""
    config = default_config.copy()
    config[param] = value
    with pytest.raises(error_type):
        HfExploreLogitsProcessor(**config)


@pytest.mark.parametrize(
    'temp_input, group_size, expected_tensor',
    [
        ([1.0, 0.5], 2, torch.tensor([1.0, 0.5])),
        (torch.tensor([1.0, 0.5]), 2, torch.tensor([1.0, 0.5])),
    ],
)
def test_initialize_temperature(
    default_config, temp_input, group_size, expected_tensor, device
):
    """Tests the _initialize_temperature helper method."""
    config = default_config.copy()
    config['temperature'] = temp_input
    config['group_size'] = group_size
    processor = HfExploreLogitsProcessor(**config)
    torch.testing.assert_close(
        processor.temperature,
        expected_tensor.to(device=device, dtype=torch.float32),
    )
    assert processor.temperature.device == device


def test_initialization_replacement_setup(default_config):
    """Tests that replacement tensors are created correctly when enabled."""
    config = default_config.copy()
    config['replace_source_tokens'] = [10, 11]
    config['replace_target_tokens'] = [20, 21]
    config['replace_max_per_seq'] = 5
    processor = HfExploreLogitsProcessor(**config)
    assert processor.replace_max_per_seq == 5
    assert processor.source_tokens_tensor is not None
    assert processor.target_tokens_tensor is not None


def test_initialization_replacement_disabled(default_config):
    """Tests that replacement is disabled if max_per_seq is 0 or tokens missing."""
    config_no_max = default_config.copy()
    config_no_max['replace_source_tokens'] = [10]
    config_no_max['replace_target_tokens'] = [20]
    config_no_max['replace_max_per_seq'] = 0
    processor_no_max = HfExploreLogitsProcessor(**config_no_max)
    assert processor_no_max.replace_max_per_seq == 0

    config_no_source = default_config.copy()
    config_no_source['replace_target_tokens'] = [20]
    config_no_source['replace_max_per_seq'] = 5
    processor_no_source = HfExploreLogitsProcessor(**config_no_source)
    assert processor_no_source.replace_max_per_seq == 0

    config_no_target = default_config.copy()
    config_no_target['replace_source_tokens'] = [10]
    config_no_target['replace_max_per_seq'] = 5
    processor_no_target = HfExploreLogitsProcessor(**config_no_target)
    assert processor_no_target.replace_max_per_seq == 0


# --- Temperature Scaling Tests ---


def test_temperature_scaling_default(
    default_config, input_ids, scores, sequence_indices
):
    """Tests temperature scaling with default temperature (1.0)."""
    processor = HfExploreLogitsProcessor(**default_config)
    original_scores = scores.clone()
    processed_scores = processor(
        input_ids, scores, sequence_indices=sequence_indices
    )
    torch.testing.assert_close(
        processed_scores, original_scores
    )  # Temp 1.0 should not change scores


def test_temperature_scaling_non_default(
    default_config, input_ids, scores, sequence_indices
):
    """Tests temperature scaling with a temperature != 1.0."""
    temp = 0.5
    config = default_config.copy()
    # Ensure temperature matches batch size if providing a list/tensor
    config['temperature'] = [temp] * config['group_size']
    processor = HfExploreLogitsProcessor(**config)
    original_scores = scores.clone()
    processed_scores = processor(
        input_ids, scores, sequence_indices=sequence_indices
    )
    # Need to handle potential clamping in the processor's safe_temp
    safe_temp = torch.clamp(torch.tensor(temp), min=1e-8)
    torch.testing.assert_close(processed_scores, original_scores / safe_temp)


def test_temperature_scaling_zero(
    default_config, input_ids, scores, sequence_indices
):
    """Tests temperature scaling with zero temperature (greedy decoding)."""
    config = default_config.copy()
    config['temperature'] = [0.0] * config['group_size']  # Match batch size
    processor = HfExploreLogitsProcessor(**config)
    # Pass a clone as the processor modifies scores
    processed_scores = processor(
        input_ids, scores.clone(), sequence_indices=sequence_indices
    )

    expected_argmax = scores.argmax(dim=-1)
    for i in range(scores.shape[0]):
        assert torch.isinf(processed_scores[i]).sum() == scores.shape[1] - 1
        assert not torch.isinf(processed_scores[i, expected_argmax[i]])
        assert (
            processed_scores[i, expected_argmax[i]] == 100.0
        )  # Check exact boost value


def test_temperature_scaling_batch(
    default_config, input_ids, scores, sequence_indices, device
):
    """Tests temperature scaling with different temperatures per batch item."""
    temps = [0.5, 1.5]
    config = default_config.copy()
    config['temperature'] = temps
    processor = HfExploreLogitsProcessor(**config)
    original_scores = scores.clone()
    processed_scores = processor(
        input_ids, scores, sequence_indices=sequence_indices
    )

    expected_scores = original_scores.clone()
    expected_scores[0] /= temps[0]
    expected_scores[1] /= temps[1]
    torch.testing.assert_close(processed_scores, expected_scores)


def test_temperature_scaling_batch_with_zero(
    default_config, input_ids, scores, sequence_indices, device
):
    """Tests batch temperature scaling including a zero temperature."""
    temps = [0.5, 0.0]
    config = default_config.copy()
    config['temperature'] = temps
    processor = HfExploreLogitsProcessor(**config)
    original_scores = scores.clone()
    processed_scores = processor(
        input_ids, scores, sequence_indices=sequence_indices
    )

    # Check first sequence (temp=0.5)
    torch.testing.assert_close(
        processed_scores[0], original_scores[0] / temps[0]
    )

    # Check second sequence (temp=0.0)
    expected_argmax_1 = scores[1].argmax()
    assert torch.isinf(processed_scores[1]).sum() == scores.shape[1] - 1
    assert not torch.isinf(processed_scores[1, expected_argmax_1])
    assert processed_scores[1, expected_argmax_1] > 0


# --- Exploration Logic Tests ---


def test_exploration_active(
    default_config, input_ids, scores, sequence_indices
):
    """Tests that exploration logic modifies scores when active."""
    config = default_config.copy()
    config['explore_steps'] = 5
    config['explore_top_k'] = 10
    processor = HfExploreLogitsProcessor(**config)

    # Simulate being in the exploration phase (step 0)
    processor.current_step = 0  # Set manually for testing __call__ effect
    seq_len = default_config['initial_seq_len'] + 1  # First step after prompt
    current_input_ids = input_ids[:, :seq_len]

    processed_scores = processor(
        current_input_ids, scores.clone(), sequence_indices=sequence_indices
    )

    # Check that only top_k scores are non-infinite and equal (0.0)
    for i in range(scores.shape[0]):
        non_inf_mask = ~torch.isinf(processed_scores[i])
        assert non_inf_mask.sum().item() == config['explore_top_k']
        assert torch.all(processed_scores[i][non_inf_mask] == 0.0)


def test_exploration_inactive_before_skip(
    default_config, input_ids, scores, sequence_indices
):
    """Tests that exploration is inactive before explore_skip_n steps."""
    config = default_config.copy()
    config['explore_steps'] = 5
    config['explore_skip_n'] = 2
    config['explore_top_k'] = 10
    processor = HfExploreLogitsProcessor(**config)

    # Simulate being before the exploration phase (step 1)
    processor.current_step = 1  # Set manually
    seq_len = default_config['initial_seq_len'] + 2  # Step 1 after prompt
    current_input_ids = input_ids[:, :seq_len]
    original_scores_scaled = scores.clone() / processor.temperature[
        sequence_indices
    ].unsqueeze(1)

    processed_scores = processor(
        current_input_ids, scores.clone(), sequence_indices=sequence_indices
    )

    # Scores should only be temperature scaled, not explored
    torch.testing.assert_close(processed_scores, original_scores_scaled)


def test_exploration_decay(default_config, input_ids, scores, sequence_indices):
    """Tests the decay of explore_top_k over exploration steps."""
    config = default_config.copy()
    config['explore_steps'] = 3
    config['explore_skip_n'] = 0
    config['explore_top_k'] = 20
    config['explore_decay'] = 0.5
    processor = HfExploreLogitsProcessor(**config)

    # Step 0
    seq_len_0 = default_config['initial_seq_len'] + 1
    ids_0 = input_ids[:, :seq_len_0]
    scores_0 = processor(
        ids_0, scores.clone(), sequence_indices=sequence_indices
    )
    k_0 = int(config['explore_top_k'] * (config['explore_decay'] ** 0))
    assert (~torch.isinf(scores_0[0])).sum().item() == k_0  # 20

    # Step 1
    seq_len_1 = default_config['initial_seq_len'] + 2
    ids_1 = input_ids[:, :seq_len_1]
    scores_1 = processor(
        ids_1, scores.clone(), sequence_indices=sequence_indices
    )
    k_1 = int(config['explore_top_k'] * (config['explore_decay'] ** 1))
    assert (~torch.isinf(scores_1[0])).sum().item() == k_1  # 10

    # Step 2
    seq_len_2 = default_config['initial_seq_len'] + 3
    ids_2 = input_ids[:, :seq_len_2]
    scores_2 = processor(
        ids_2, scores.clone(), sequence_indices=sequence_indices
    )
    k_2 = int(config['explore_top_k'] * (config['explore_decay'] ** 2))
    assert (~torch.isinf(scores_2[0])).sum().item() == k_2  # 5

    # Step 3 (after exploration)
    seq_len_3 = default_config['initial_seq_len'] + 4
    ids_3 = input_ids[:, :seq_len_3]
    scores_3 = processor(
        ids_3, scores.clone(), sequence_indices=sequence_indices
    )
    assert not torch.all(
        scores_3[0] == 0.0
    )  # Should not be explored uniform dist
    assert (
        ~torch.isinf(scores_3[0])
    ).sum().item() >= k_2  # Should be more than last k


# --- Replacement Logic Tests ---


@pytest.fixture
def replacement_config(default_config):
    """Provides a config with replacement enabled."""
    config = default_config.copy()
    config['replace_source_tokens'] = [10, 11]
    config['replace_target_tokens'] = [20, 21]
    config['replace_prevent_patterns'] = [[5, 6], [7]]
    config['replace_prob'] = 1.0  # Ensure replacement happens if eligible
    config['replace_max_per_seq'] = 2
    config['replace_check_top_n'] = 3
    return config


@pytest.fixture
def replacement_processor(replacement_config):
    """Provides a processor instance with replacement enabled."""
    return HfExploreLogitsProcessor(**replacement_config)


# Test _check_replacement_patterns
@pytest.mark.parametrize(
    'generated_ids_list, patterns, expected_allowed_list',
    [
        ([[1, 2, 3]], [[5, 6]], [True]),  # Pattern not present
        ([[1, 5, 6]], [[5, 6]], [False]),  # Pattern present at end
        ([[5, 6, 1]], [[5, 6]], [False]),  # Pattern present at start
        ([[1, 5, 1, 6]], [[5, 6]], [True]),  # Pattern not contiguous
        ([[1, 7, 2]], [[5, 6], [7]], [False]),  # Second pattern present
        ([[1, 2, 3], [4, 5, 6]], [[5, 6]], [True, False]),  # Batch check
        ([[]], [[5, 6]], [True]),  # Empty generated sequence
        ([[1, 2]], [[5, 6, 7]], [True]),  # Sequence shorter than pattern
        ([[1, 2, 3]], [], [True]),  # No patterns defined
        ([[1, 2, 3]], [[]], [True]),  # Empty pattern list
    ],
)
def test_check_replacement_patterns(
    replacement_processor,
    generated_ids_list,
    patterns,
    expected_allowed_list,
    device,
):
    """Tests the _check_replacement_patterns helper method."""
    replacement_processor.replace_prevent_patterns = patterns
    generated_ids = torch.tensor(
        generated_ids_list, device=device, dtype=torch.long
    )
    expected_allowed = torch.tensor(
        expected_allowed_list, device=device, dtype=torch.bool
    )
    allowed = replacement_processor._check_replacement_patterns(generated_ids)
    torch.testing.assert_close(allowed, expected_allowed)


# Test _check_correctness
def test_check_correctness_no_callback(
    replacement_processor, input_ids, device
):
    """Tests _check_correctness when no callback is provided."""
    replacement_processor.correctness_callback = None
    pattern_allows = torch.tensor([True, True], device=device)
    is_incorrect = replacement_processor._check_correctness(
        input_ids, pattern_allows
    )
    # Should assume incorrect (eligible) if no callback
    torch.testing.assert_close(
        is_incorrect, torch.tensor([True, True], device=device)
    )


def test_check_correctness_with_callback(
    replacement_processor, input_ids, device
):
    """Tests _check_correctness using the callback results."""
    mock_callback = MagicMock(return_value=[0.5, 1.0])
    replacement_processor.correctness_callback = mock_callback
    # Ensure processor tensors are on the right device for internal checks
    replacement_processor._ensure_device(device)

    pattern_allows = torch.tensor([True, True], device=device)
    # _check_correctness returns is_incorrect mask (True if score < 1.0)
    is_incorrect = replacement_processor._check_correctness(
        input_ids, pattern_allows
    )

    # Check that the callback was called with the output from the mock tokenizer's side_effect
    # The side_effect produces ['decoded_0', 'decoded_1'] for a batch of 2
    mock_callback.assert_called_once_with(['decoded_0', 'decoded_1'])

    # Expected: seq0 is incorrect (0.5 < 1.0), seq1 is correct (1.0 >= 1.0)
    # This part of the test remains the same, checking the output mask based on callback return values
    torch.testing.assert_close(
        is_incorrect,
        torch.tensor([True, False], device=device, dtype=torch.bool),
    )


def test_check_correctness_callback_error(
    replacement_processor, input_ids, device
):
    """Tests _check_correctness handling callback errors."""
    mock_callback = MagicMock(side_effect=ValueError('Callback failed'))
    replacement_processor.correctness_callback = mock_callback
    replacement_processor.tokenizer.batch_decode.return_value = ['seq0', 'seq1']

    pattern_allows = torch.tensor([True, True], device=device)
    with warnings.catch_warnings(record=True) as w:
        is_incorrect = replacement_processor._check_correctness(
            input_ids, pattern_allows
        )
        assert len(w) == 1
        assert 'Correctness callback failed' in str(w[0].message)

    # Should assume incorrect on error
    torch.testing.assert_close(
        is_incorrect, torch.tensor([True, True], device=device)
    )


def test_check_correctness_callback_wrong_length(
    replacement_processor, input_ids, device
):
    """Tests _check_correctness handling callback returning wrong number of scores."""
    mock_callback = MagicMock(
        return_value=[0.5]
    )  # Only one score for two inputs
    replacement_processor.correctness_callback = mock_callback
    replacement_processor.tokenizer.batch_decode.return_value = ['seq0', 'seq1']

    pattern_allows = torch.tensor([True, True], device=device)
    with warnings.catch_warnings(record=True) as w:
        is_incorrect = replacement_processor._check_correctness(
            input_ids, pattern_allows
        )
        assert len(w) == 1
        assert 'returned 1 scores, expected 2' in str(w[0].message)

    # Should assume incorrect on length mismatch
    torch.testing.assert_close(
        is_incorrect, torch.tensor([True, True], device=device)
    )


def test_check_correctness_respects_pattern_mask(
    replacement_processor, input_ids, device
):
    """Tests that correctness callback is only called for sequences allowed by patterns."""
    mock_callback = MagicMock(
        return_value=[0.5]
    )  # Callback returns score for the one sequence it receives
    replacement_processor.correctness_callback = mock_callback
    # Let the default mock tokenizer fixture's side_effect work:
    # side_effect=lambda x, **kwargs: [f"decoded_{i}" for i in range(len(x))]
    # When called with one sequence, it will return ['decoded_0']

    # Ensure processor tensors are on the right device for internal checks
    replacement_processor._ensure_device(device)

    pattern_allows = torch.tensor([False, True], device=device)
    # _check_correctness returns is_incorrect mask
    is_incorrect = replacement_processor._check_correctness(
        input_ids, pattern_allows
    )

    # Check tokenizer was called once with the correct input tensor slice (index 1)
    replacement_processor.tokenizer.batch_decode.assert_called_once()
    call_args, call_kwargs = (
        replacement_processor.tokenizer.batch_decode.call_args
    )
    indices_passed_to_decode = torch.where(pattern_allows)[
        0
    ]  # Should be tensor([1])
    # Ensure we compare the tensor passed to the mock with the expected slice
    torch.testing.assert_close(
        call_args[0].cpu(), input_ids[indices_passed_to_decode].cpu()
    )
    assert call_kwargs.get('skip_special_tokens') is True  # Check kwargs too

    # Check callback was called with the text returned by the mock tokenizer's side_effect
    # For a single input sequence (input_ids[1]), the side_effect returns ['decoded_0']
    mock_callback.assert_called_once_with(['decoded_0'])

    # Expected: seq0 is correct by default (not checked -> is_incorrect=False),
    # seq1 is incorrect (checked, score 0.5 < 1.0 -> is_incorrect=True)
    torch.testing.assert_close(
        is_incorrect,
        torch.tensor([False, True], device=device, dtype=torch.bool),
    )


# Test _determine_replacement_eligibility
@patch.object(HfExploreLogitsProcessor, '_check_replacement_patterns')
@patch.object(HfExploreLogitsProcessor, '_check_correctness')
def test_determine_eligibility(
    mock_check_correctness,
    mock_check_patterns,
    replacement_processor,
    input_ids,
    device,
):
    """Tests the combination of factors for replacement eligibility."""
    # Setup: Seq 0: count=1, pattern allows, incorrect. Seq 1: count=2 (max), pattern prevents, correct
    replacement_processor.replacement_counts = torch.tensor(
        [1, 2], device=device
    )  # Seq 1 hits max (2)
    mock_check_patterns.return_value = torch.tensor(
        [True, False], device=device
    )  # Seq 0 allows, Seq 1 prevents
    mock_check_correctness.return_value = torch.tensor(
        [True, False], device=device
    )  # Seq 0 incorrect, Seq 1 correct (though won't be checked)

    eligible = replacement_processor._determine_replacement_eligibility(
        input_ids
    )

    # Only Seq 0 should be eligible (count < max AND pattern allows AND incorrect)
    torch.testing.assert_close(
        eligible, torch.tensor([True, False], device=device)
    )
    mock_check_patterns.assert_called_once()
    # Correctness check should only be performed where count < max AND pattern allows
    mock_check_correctness.assert_called_once()
    torch.testing.assert_close(
        mock_check_correctness.call_args[0][1],
        torch.tensor([True, False], device=device),
    )  # needs_correctness_check mask


# Test _apply_replacement_logic
@patch('torch.rand_like')
def test_apply_replacement_logic_success(
    mock_rand, replacement_processor, scores, device
):
    """Tests applying replacement when conditions are met."""
    mock_rand.return_value = torch.tensor(
        [0.4, 0.6], device=device
    )  # Ensure prob check passes for first
    replacement_processor.replace_prob = 0.5

    # Seq 0: Eligible, Top N includes source token 10. Seq 1: Eligible, Top N does not include source tokens.
    eligible_mask = torch.tensor([True, True], device=device)
    # Make top 3 for seq 0 include 10, top 3 for seq 1 not include 10 or 11
    scores[0, 10] = 1000.0
    scores[0, 5] = 900.0
    scores[0, 6] = 800.0
    scores[1, 50] = 1000.0
    scores[1, 51] = 900.0
    scores[1, 52] = 800.0
    _, original_top_n = torch.topk(
        scores, k=replacement_processor.replace_check_top_n, dim=-1
    )

    initial_scores = scores.clone()
    initial_counts = replacement_processor.replacement_counts.clone()

    replacement_processor._apply_replacement_logic(
        scores, eligible_mask, original_top_n
    )

    # Seq 0 should have replacement applied (Eligible & TopNSource & ProbMet)
    assert torch.isinf(scores[0, 10])  # Source token penalized
    assert torch.isinf(scores[0, 11])  # Source token penalized
    # One target token should be boosted
    boosted_target_0 = (
        scores[0, replacement_processor.target_tokens_tensor]
        > initial_scores[0, replacement_processor.target_tokens_tensor]
    ).any()
    assert boosted_target_0
    assert replacement_processor.replacement_counts[0] == initial_counts[0] + 1

    # Seq 1 should NOT have replacement applied (Eligible but TopNSource is False)
    torch.testing.assert_close(scores[1], initial_scores[1])
    assert replacement_processor.replacement_counts[1] == initial_counts[1]


@patch('torch.rand_like')
def test_apply_replacement_logic_prob_fail(
    mock_rand, replacement_processor, scores, device
):
    """Tests that replacement is skipped if probability check fails."""
    mock_rand.return_value = torch.tensor(
        [0.6], device=device
    )  # Ensure prob check fails
    replacement_processor.replace_prob = 0.5

    eligible_mask = torch.tensor(
        [True], device=device
    )  # Single item batch for simplicity
    scores[0, 10] = 1000.0  # Make top N include source token 10
    _, original_top_n = torch.topk(
        scores, k=replacement_processor.replace_check_top_n, dim=-1
    )

    initial_scores = scores.clone()
    initial_counts = replacement_processor.replacement_counts.clone()

    replacement_processor._apply_replacement_logic(
        scores, eligible_mask, original_top_n
    )

    # Scores and counts should be unchanged
    torch.testing.assert_close(scores, initial_scores)
    torch.testing.assert_close(
        replacement_processor.replacement_counts, initial_counts
    )


def test_apply_replacement_logic_ineligible(
    replacement_processor, scores, device
):
    """Tests that replacement is skipped if eligibility mask is false."""
    eligible_mask = torch.tensor([False, False], device=device)
    scores[0, 10] = 1000.0  # Make top N include source token 10
    _, original_top_n = torch.topk(
        scores, k=replacement_processor.replace_check_top_n, dim=-1
    )

    initial_scores = scores.clone()
    initial_counts = replacement_processor.replacement_counts.clone()

    replacement_processor._apply_replacement_logic(
        scores, eligible_mask, original_top_n
    )

    # Scores and counts should be unchanged
    torch.testing.assert_close(scores, initial_scores)
    torch.testing.assert_close(
        replacement_processor.replacement_counts, initial_counts
    )


# --- Integration Tests (__call__) ---


def test_call_updates_step_count(
    default_config, input_ids, scores, sequence_indices
):
    """Tests that the internal step count is updated correctly."""
    processor = HfExploreLogitsProcessor(**default_config)
    assert processor.current_step == 0

    # Call with sequence length > initial_seq_len
    processor(input_ids, scores.clone(), sequence_indices=sequence_indices)
    assert processor.current_step == 1

    # Call again
    processor(input_ids, scores.clone(), sequence_indices=sequence_indices)
    assert processor.current_step == 2

    # Call with sequence length <= initial_seq_len (should not increment)
    prompt_ids = input_ids[:, : default_config['initial_seq_len']]
    processor(prompt_ids, scores.clone(), sequence_indices=sequence_indices)
    assert processor.current_step == 2


def test_call_replacement_skipped_during_exploration(
    replacement_processor, input_ids, scores, sequence_indices
):
    """Tests that replacement is skipped during the exploration phase."""
    replacement_processor.explore_steps = 2
    replacement_processor.explore_skip_n = 0
    replacement_processor.replace_prob = 1.0
    replacement_processor.replace_max_per_seq = 1

    # Simulate being in exploration step 1
    seq_len = replacement_processor.initial_seq_len + 2
    current_input_ids = input_ids[:, :seq_len]
    current_scores = scores.clone()

    with (
        patch.object(
            HfExploreLogitsProcessor, '_determine_replacement_eligibility'
        ) as mock_determine,
        patch.object(
            HfExploreLogitsProcessor, '_apply_replacement_logic'
        ) as mock_apply,
    ):

        processed_scores = replacement_processor(
            current_input_ids, current_scores, sequence_indices=sequence_indices
        )

        # Replacement checks should not be called
        mock_determine.assert_not_called()
        mock_apply.assert_not_called()

        # Scores should reflect exploration, not replacement
        assert torch.all(
            processed_scores[0] <= 0.0
        )  # Exploration sets to 0 or -inf
        assert torch.isinf(processed_scores[0]).sum() > 0
