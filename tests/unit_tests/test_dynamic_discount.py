import pytest
import torch

from rl4llm.core.grpo import GRPOTrainer


def test_valid_inputs():
    # Test with episode_length = 5000, max_length = 10000, min_gamma = 0.999, max_gamma = 0.9999
    result = GRPOTrainer.compute_dynamic_discount(5000, 0.999, 0.9999, 10000)
    assert result > 0.999 and result < 0.9999, f"Unexpected result: {result}"

    # Test with episode_length = max_length, should give max_gamma
    result = GRPOTrainer.compute_dynamic_discount(10000, 0.999, 0.9999, 10000)
    assert result == 0.9999, f"Expected max_gamma, but got {result}"

    # Test with episode_length smaller than max_length
    result = GRPOTrainer.compute_dynamic_discount(2000, 0.999, 0.9999, 10000)
    assert result > 0.999 and result < 0.9999, f"Unexpected result: {result}"


def test_edge_cases():
    # Test with episode_length close to min_gamma
    result = GRPOTrainer.compute_dynamic_discount(100, 0.999, 0.9999, 10000)
    assert 0.999 < result and result < 0.9991, f"Expected min_gamma, but got {result}"

    # Test with episode_length very close to max_length
    result = GRPOTrainer.compute_dynamic_discount(9999, 0.999, 0.9999, 10000)
    assert result < 0.9999 and result > 0.999, f"Unexpected result: {result}"


def test_invalid_inputs():
    # Test with episode_length <= 0 (invalid input)
    with pytest.raises(AssertionError, match='Episode length must be greater than 0.'):
        GRPOTrainer.compute_dynamic_discount(0, 0.999, 0.9999, 10000)

    # Test with max_length <= 1000 (invalid input)
    with pytest.raises(AssertionError, match='Max length must be greater than 1000.'):
        GRPOTrainer.compute_dynamic_discount(5000, 0.999, 0.9999, 1000)

    # Test with min_gamma <= 0 (invalid input)
    with pytest.raises(AssertionError, match='Min discount must be in the range'):
        GRPOTrainer.compute_dynamic_discount(5000, 0.0, 0.9999, 10000)

    # Test with max_gamma >= 1 (invalid input)
    with pytest.raises(AssertionError, match='Max discount must be in the range'):
        GRPOTrainer.compute_dynamic_discount(5000, 0.999, 1.0, 10000)

    # Test with min_gamma >= max_gamma (invalid input)
    with pytest.raises(AssertionError, match='Min discount must be less than max discount.'):
        GRPOTrainer.compute_dynamic_discount(5000, 0.9999, 0.999, 10000)


def test_boundary_conditions():
    # Test with min_gamma == max_gamma (should be an invalid input)
    with pytest.raises(AssertionError, match='Min discount must be less than max discount.'):
        GRPOTrainer.compute_dynamic_discount(5000, 0.9999, 0.9999, 10000)

    # Test with episode_length being very small compared to max_length
    result = GRPOTrainer.compute_dynamic_discount(10, 0.999, 0.9999, 10000)
    assert result > 0.999, f"Expected value closer to min_gamma, but got {result}"

    # Test with episode_length equal to max_length, gamma should be exactly max_gamma
    result = GRPOTrainer.compute_dynamic_discount(10000, 0.999, 0.9999, 10000)
    assert result == 0.9999, f"Expected max_gamma, but got {result}"
