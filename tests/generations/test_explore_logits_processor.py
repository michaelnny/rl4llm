import random

import numpy as np
import pytest
import torch
from transformers.generation.logits_process import LogitsProcessor

from rl4llm.generations.explore_processor import ExploreLogitsProcessor


# Set a fixed seed for reproducibility
@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)


# Create a basic processor fixture
@pytest.fixture
def basic_processor():
    return ExploreLogitsProcessor(
        temperatures=[1.0],
        explore_steps=2,  # Updated parameter name
        explore_skip=0,  # Updated parameter name
        explore_top_k=5,
        replace_source_tokens=[100, 101],
        replace_target_tokens=[200, 201],
        replace_prevent_patterns=[[300, 301]],
        replace_prob=1.0,  # 100% for deterministic testing
        replace_max_per_seq=2,  # Updated parameter name
        replace_threshold=0.8,  # Updated parameter name
    )


# Create mock data for testing
@pytest.fixture
def mock_data():
    # Create sample input_ids and logits
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # Create logits where token 100 has high probability in both sequences
    logits = (
        torch.ones((2, 1000)) * -100.0
    )  # Very low probability for most tokens
    logits[:, 100] = 10.0  # High probability for token 100
    logits[:, 101] = 5.0  # Medium probability for token 101

    return {'input_ids': input_ids, 'logits': logits}


def test_processor_is_logits_processor(basic_processor):
    """Test that the processor is a subclass of LogitsProcessor."""
    assert isinstance(basic_processor, LogitsProcessor)


def test_exploration_phase(basic_processor, mock_data):
    """Test that exploration phase uses uniform sampling from top-k."""
    input_ids = mock_data['input_ids']
    logits = mock_data['logits']

    # Step 0 - should be in exploration phase
    processed_logits = basic_processor(input_ids, logits)

    # Find which tokens have high probability
    high_prob_tokens = (processed_logits[0] > 0).nonzero(as_tuple=True)[0]

    # Should have exactly top_k tokens with high probability
    assert len(high_prob_tokens) == basic_processor.explore_top_k

    # All selected tokens should have the same probability (uniform)
    high_probs = processed_logits[0, high_prob_tokens]
    assert torch.allclose(
        high_probs, high_probs[0] * torch.ones_like(high_probs)
    )


def test_temperature_scaling(mock_data):
    """Test that temperature scaling works correctly."""
    input_ids = mock_data['input_ids']
    logits = mock_data['logits']

    # Create processor with different temperatures for each sequence
    processor = ExploreLogitsProcessor(
        temperatures=[1.0, 2.0],  # Different temperatures
        explore_steps=0,  # No exploration for this test
        explore_skip=0,
        explore_top_k=5,
        replace_source_tokens=[],
        replace_target_tokens=[],
        replace_prevent_patterns=[],
        replace_prob=0.0,  # No replacement
        replace_max_per_seq=2,
        replace_threshold=0.8,
    )

    # Process logits
    processed_logits = processor(input_ids, logits)

    # Second sequence should have lower values due to higher temperature
    assert torch.all(processed_logits[0, 100] > processed_logits[1, 100])

    # Test that the ratio is approximately the temperature ratio
    ratio = processed_logits[0, 100] / processed_logits[1, 100]
    assert 1.9 < ratio < 2.1  # Around 2.0, allowing for floating-point error


def test_token_replacement(basic_processor, mock_data):
    """Test that token replacement works correctly."""
    input_ids = mock_data['input_ids']
    logits = mock_data['logits']

    # Reset the processor to ensure clean state
    basic_processor.reset()

    # First, let's determine exactly when token replacement should occur based on the code
    # The condition is: self.current_step > (self.explore_steps + self.explore_skip)
    threshold_step = (
        basic_processor.explore_steps + basic_processor.explore_skip
    )

    # Move to exactly the step where token replacement should begin
    for _ in range(threshold_step + 1):
        basic_processor(input_ids, logits.clone())

    # Now token replacement should occur
    processed_logits = basic_processor(input_ids, logits.clone())

    # Print debug info to understand the issue
    print(f"Current step: {basic_processor.current_step}")
    print(f"Threshold for replacement: {threshold_step + 1}")
    print(f"Source token 100 score: {processed_logits[0, 100]}")
    print(f"Source token 101 score: {processed_logits[0, 101]}")
    print(f"Target token 200 score: {processed_logits[0, 200]}")
    print(f"Target token 201 score: {processed_logits[0, 201]}")

    # Source tokens should have lower probability
    assert (
        processed_logits[0, 100] < 0
    ), 'Source token 100 should have reduced probability'
    assert (
        processed_logits[0, 101] < 0
    ), 'Source token 101 should have reduced probability'

    # Target tokens should have higher probability
    assert (
        processed_logits[0, 200] > 0
    ), 'Target token 200 should have increased probability'
    assert (
        processed_logits[0, 201] > 0
    ), 'Target token 201 should have increased probability'

    # Target tokens should have higher probability than source tokens
    assert (
        processed_logits[0, 200] > processed_logits[0, 100]
    ), 'Target token 200 should be higher than source 100'
    assert (
        processed_logits[0, 201] > processed_logits[0, 101]
    ), 'Target token 201 should be higher than source 101'


def test_source_token_threshold(mock_data):
    """Test that the replace_threshold controls when replacements happen."""
    input_ids = mock_data['input_ids']

    # Create a more balanced set of logits to get realistic probabilities
    # We'll use zero as the base value to get more reasonable softmax probabilities
    logits = torch.zeros((2, 1000))

    # First sequence has token 100 with high relative probability
    logits[0, 100] = 5.0  # This should give a high probability after softmax
    # Add some competing tokens so 100 isn't the only non-zero token
    logits[0, 101:110] = 2.0

    # Second sequence has token 100 with lower relative probability
    logits[1, 100] = 1.0  # This should give a lower probability after softmax
    # Add competing tokens with higher values to ensure token 100 has low probability
    logits[1, 101:110] = 3.0

    # Calculate actual probabilities
    probs = torch.softmax(logits, dim=-1)

    # Measure the actual probabilities and set threshold in between
    prob_high = probs[0, 100].item()
    prob_low = probs[1, 100].item()
    threshold = (prob_high + prob_low) / 2

    print(f"High probability: {prob_high}")
    print(f"Low probability: {prob_low}")
    print(f"Using threshold: {threshold}")

    # Create processor with threshold between the two probabilities
    processor = ExploreLogitsProcessor(
        temperatures=[1.0],
        explore_steps=0,
        explore_skip=0,
        explore_top_k=5,
        replace_source_tokens=[100],
        replace_target_tokens=[200],
        replace_prevent_patterns=[],
        replace_prob=1.0,
        replace_max_per_seq=2,
        replace_threshold=threshold,
    )

    # Ensure we're at a step where token replacement should happen
    threshold_step = processor.explore_steps + processor.explore_skip + 1
    for _ in range(threshold_step):
        processor(input_ids, logits.clone())

    # Process logits for token replacement
    processed_logits = processor(input_ids, logits.clone())

    # Print debug information
    print(f"Current step: {processor.current_step}")
    print(f"Token 100 probability seq 0: {probs[0, 100]}")
    print(f"Token 100 probability seq 1: {probs[1, 100]}")
    print(f"Threshold: {processor.replace_threshold}")
    print(f"Processed logit 100 seq 0: {processed_logits[0, 100]}")
    print(f"Processed logit 100 seq 1: {processed_logits[1, 100]}")

    # First sequence should have replacement (probability above threshold)
    assert (
        probs[0, 100] > processor.replace_threshold
    ), 'Probability should be above threshold'
    assert (
        processed_logits[0, 100] < 0
    ), 'Source token 100 should be reduced for seq 0'
    assert (
        processed_logits[0, 200] > 0
    ), 'Target token 200 should be boosted for seq 0'

    # Second sequence should not have replacement (probability below threshold)
    assert (
        probs[1, 100] < processor.replace_threshold
    ), 'Probability should be below threshold'
    # Either token 100 is unchanged or only affected by temperature scaling
    assert (
        abs(
            processed_logits[1, 100]
            - (logits[1, 100] / processor.temperatures[0])
        )
        < 1e-5
    ), "Token 100 shouldn't be modified beyond temperature scaling for seq 1"


def test_prevent_patterns(mock_data):
    """Test that replacement is prevented when specified patterns are present."""
    # Create input_ids with a prevent pattern in second sequence
    input_ids = torch.tensor([[1, 2, 3], [300, 301, 6]])
    logits = mock_data['logits']

    # Create processor with 100% replacement probability
    processor = ExploreLogitsProcessor(
        temperatures=[1.0],
        explore_steps=0,  # No exploration for this test
        explore_skip=0,
        explore_top_k=5,
        replace_source_tokens=[100],
        replace_target_tokens=[200],
        replace_prevent_patterns=[
            [300, 301]
        ],  # This pattern is in the second sequence
        replace_prob=1.0,  # 100% replacement
        replace_max_per_seq=2,
        replace_threshold=0.8,
    )

    # Advance steps to the EXACT point where token replacement kicks in
    # The condition is: current_step > (explore_steps + explore_skip)
    threshold_step = processor.explore_steps + processor.explore_skip
    for _ in range(threshold_step + 1):
        processor(input_ids, logits.clone())

    # Process logits for token replacement
    processed_logits = processor(input_ids, logits.clone())

    # Print debug information
    print(f"Current step: {processor.current_step}")
    print(f"Processed logit 100 seq 0: {processed_logits[0, 100]}")
    print(f"Processed logit 200 seq 0: {processed_logits[0, 200]}")
    print(f"Processed logit 100 seq 1: {processed_logits[1, 100]}")
    print(f"Processed logit 200 seq 1: {processed_logits[1, 200]}")

    # First sequence should have replacement (source token reduced)
    assert (
        processed_logits[0, 100] < 0
    ), 'Source token should be reduced for seq 0'
    assert (
        processed_logits[0, 200] > 0
    ), 'Target token should be boosted for seq 0'

    # Second sequence should not have replacement due to prevent pattern
    assert (
        processed_logits[1, 100] > 0
    ), 'Source token should remain high for seq 1'
    # Two possibilities: either target not boosted, or it was already boosted in previous steps
    if processed_logits[1, 200] > 0:
        # If it's boosted, it should be less than the boost value (100.0)
        assert (
            processed_logits[1, 200] < 90
        ), "Target shouldn't be newly boosted for seq 1"


def test_reset_method(basic_processor, mock_data):
    """Test that the reset method resets internal state."""
    input_ids = mock_data['input_ids']
    logits = mock_data['logits']

    # Advance steps and perform some replacements
    for _ in range(
        basic_processor.explore_steps + basic_processor.explore_skip + 5
    ):
        basic_processor(input_ids, logits)

    # Current step should be advanced and replacements should be tracked
    assert basic_processor.current_step > 0
    assert basic_processor.replacement_counts is not None

    # Reset the processor
    basic_processor.reset()

    # Check that internal state is reset
    assert basic_processor.current_step == 0
    assert basic_processor.replacement_counts is None


def test_update_config_method(basic_processor, mock_data):
    """Test that the update_config method updates parameters correctly."""
    # Original settings
    original_prob = basic_processor.replace_prob
    original_max = basic_processor.replace_max_per_seq

    # Update configuration
    basic_processor.update_config(replace_prob=0.5, replace_max_per_seq=5)

    # Check updated values
    assert basic_processor.replace_prob == 0.5
    assert basic_processor.replace_max_per_seq == 5
    assert basic_processor.replace_prob != original_prob
    assert basic_processor.replace_max_per_seq != original_max


def test_update_source_tokens(basic_processor, mock_data):
    """Test that updating source tokens updates the source tokens set."""
    new_source_tokens = [102, 103]

    # Update source tokens
    basic_processor.update_config(replace_source_tokens=new_source_tokens)

    # Check that both list and set are updated
    assert basic_processor.replace_source_tokens == new_source_tokens
    assert basic_processor.replace_source_tokens_set == set(new_source_tokens)


def test_max_replacements_limit(basic_processor, mock_data):
    """Test that the number of replacements is limited to replace_max_per_seq."""
    input_ids = mock_data['input_ids']
    logits = mock_data['logits']

    # Reset to ensure fresh state
    basic_processor.reset()

    # Skip the exploration phase and move past it
    for _ in range(
        basic_processor.explore_steps + basic_processor.explore_skip + 1
    ):
        basic_processor(input_ids, logits)

    # Get original logits to compare
    original_logits = logits.clone()

    # Process for token replacement for more steps than replace_max_per_seq
    replacements_made = 0
    for i in range(5):
        before_scores = original_logits.clone()
        processed_logits = basic_processor(input_ids, before_scores)

        # Check if token 100 was replaced (score reduced significantly)
        if processed_logits[0, 100] < 0:
            replacements_made += 1

    # Should have made exactly replace_max_per_seq replacements
    assert replacements_made == basic_processor.replace_max_per_seq


def test_empty_source_or_target(mock_data):
    """Test that processor works correctly with empty source or target tokens."""
    input_ids = mock_data['input_ids']
    logits = mock_data['logits']

    # Create processor with empty source tokens
    processor1 = ExploreLogitsProcessor(
        temperatures=[1.0],
        explore_steps=0,
        explore_skip=0,
        explore_top_k=5,
        replace_source_tokens=[],  # Empty source
        replace_target_tokens=[200],
        replace_prevent_patterns=[],
        replace_prob=1.0,
        replace_max_per_seq=2,
        replace_threshold=0.8,
    )

    # Create processor with empty target tokens
    processor2 = ExploreLogitsProcessor(
        temperatures=[1.0],
        explore_steps=0,
        explore_skip=0,
        explore_top_k=5,
        replace_source_tokens=[100],
        replace_target_tokens=[],  # Empty target
        replace_prevent_patterns=[],
        replace_prob=1.0,
        replace_max_per_seq=2,
        replace_threshold=0.8,
    )

    # Advance steps to get past any exploration phase
    for _ in range(5):
        processor1(input_ids, logits)
        processor2(input_ids, logits)

    # Process logits with both processors
    processed1 = processor1(input_ids, logits)
    processed2 = processor2(input_ids, logits)

    # With empty source, no replacement should happen
    assert torch.allclose(
        processed1, logits.clone() / processor1.temperatures[0]
    )

    # With empty target, no replacement should happen
    assert torch.allclose(
        processed2, logits.clone() / processor2.temperatures[0]
    )


def test_temperature_tensor(mock_data):
    """Test that processor works correctly with temperature as a tensor."""
    input_ids = mock_data['input_ids']
    logits = mock_data['logits']

    # Create temperature tensor
    temperatures = torch.tensor([1.0, 2.0])

    # Create processor with tensor temperatures
    processor = ExploreLogitsProcessor(
        temperatures=temperatures,
        explore_steps=0,
        explore_skip=0,
        explore_top_k=5,
        replace_source_tokens=[],
        replace_target_tokens=[],
        replace_prevent_patterns=[],
        replace_prob=0.0,
        replace_max_per_seq=2,
        replace_threshold=0.8,
    )

    # Process logits
    processed_logits = processor(input_ids, logits)

    # Check that temperature scaling is applied correctly
    assert torch.allclose(processed_logits[0], logits[0] / temperatures[0])
    assert torch.allclose(processed_logits[1], logits[1] / temperatures[1])


def test_temperature_broadcast(mock_data):
    """Test that temperature is correctly broadcast to match batch size."""
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3 sequences
    logits = torch.ones((3, 1000))

    # Create processor with single temperature
    processor = ExploreLogitsProcessor(
        temperatures=[1.5],  # Single value
        explore_steps=0,
        explore_skip=0,
        explore_top_k=5,
        replace_source_tokens=[],
        replace_target_tokens=[],
        replace_prevent_patterns=[],
        replace_prob=0.0,
        replace_max_per_seq=2,
        replace_threshold=0.8,
    )

    # Process logits
    processed_logits = processor(input_ids, logits)

    # All sequences should be scaled by the same temperature
    assert torch.allclose(processed_logits[0], logits[0] / 1.5)
    assert torch.allclose(processed_logits[1], logits[1] / 1.5)
    assert torch.allclose(processed_logits[2], logits[2] / 1.5)
