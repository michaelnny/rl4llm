import pytest
import torch

from rl4llm.core.grpo import GRPOTrainer


def test_masked_monte_carlo_returns_no_discount():
    """Tests with discount = 1.0, which is what most RL for LLM uses"""

    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool)

    gamma = 1.0

    result = GRPOTrainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    # Check that returns only contain values for assistant turns (mask == 1)
    expected_result = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    assert torch.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_masked_monte_carlo_returns_discount():
    """Test with discount"""
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool)

    gamma = 0.9

    result = GRPOTrainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    # Check that returns only contain values for assistant turns (mask == 1)
    expected_result = torch.tensor([0.0, 0.0, 0.0, 0.81, 0.9, 1.0], dtype=torch.float32)
    assert torch.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"


def test_masked_monte_carlo_returns_no_assistant_turns():
    """Test with no assistant turn"""
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool)  # All user turns (mask = 0)
    gamma = 0.9

    result = GRPOTrainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    # All rewards are for the user, so returns should be zero
    assert torch.equal(result, torch.zeros_like(mask, dtype=torch.float32)), f"Expected zeros, but got {result}"


def test_masked_monte_carlo_returns_all_assistant_turns():
    """Tests with all assistant turns"""
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)  # All assistant turns (mask = 1)
    gamma = 0.9

    result = GRPOTrainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    expected_result = torch.tensor([0.656, 0.729, 0.81, 0.9, 1.0], dtype=torch.float32)
    assert torch.allclose(result, expected_result, atol=1e-2), f"Expected {expected_result}, but got {result}"


def test_masked_monte_carlo_returns_multi_turns():
    """Tests with multi-turn"""
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.bool)
    gamma = 0.9

    result = GRPOTrainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    expected_result = torch.tensor([0.0, 0.0, 1.179, 1.31, 0.0, 0.0, 0.9, 1.0], dtype=torch.float32)
    assert torch.allclose(result, expected_result, atol=1e-2), f"Expected {expected_result}, but got {result}"
