from unittest.mock import MagicMock

import pytest
import torch

from rl4llm.core.grpo import GRPOConfig, GRPOTrainer
from rl4llm.graders import FormatGrader, MathGrader


@pytest.fixture
def base_trainer() -> GRPOTrainer:

    return GRPOTrainer(
        config=GRPOConfig(),
        policy_model=MagicMock(),
        tokenizer=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        train_ds=MagicMock(),
        test_ds=MagicMock(),
        device=torch.device('cpu'),
        torch_dtype=torch.float32,
        artifacts_path='/tmp/unit_test_artifacts',
    )


def test_masked_monte_carlo_returns(base_trainer: GRPOTrainer):
    """Tests compute masked monte carlo returns"""

    # Without discount
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool)

    gamma = 1.0

    result = base_trainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    # Check that returns only contain values for assistant turns (mask == 1)
    expected_result = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    assert torch.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test with discount
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool)

    gamma = 0.9

    result = base_trainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    # Check that returns only contain values for assistant turns (mask == 1)
    expected_result = torch.tensor([0.0, 0.0, 0.0, 0.81, 0.9, 1.0], dtype=torch.float32)
    assert torch.allclose(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test with no assistant turn
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool)  # All user turns (mask = 0)
    gamma = 0.9

    result = base_trainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    # All rewards are for the user, so returns should be zero
    assert torch.equal(result, torch.zeros_like(mask, dtype=torch.float32)), f"Expected zeros, but got {result}"

    # Tests with all assistant turns
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)  # All assistant turns (mask = 1)
    gamma = 0.9

    result = base_trainer.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    expected_result = torch.tensor([0.656, 0.729, 0.81, 0.9, 1.0], dtype=torch.float32)
    assert torch.allclose(result, expected_result, atol=1e-2), f"Expected {expected_result}, but got {result}"


def test_dynamic_discount(base_trainer: GRPOTrainer):
    """Tests compute dynamic discount"""

    patch_config = GRPOConfig(
        min_gamma=0.999,
        max_gamma=0.9999,
        max_completion_length=10000,
    )
    base_trainer.config = patch_config

    result = base_trainer.compute_dynamic_discount(5000)
    assert result > 0.999 and result < 0.9999, f"Unexpected result: {result}"

    # Test with episode_length = max_length, should give max_gamma
    result = base_trainer.compute_dynamic_discount(10000)
    assert result == 0.9999, f"Expected max_gamma, but got {result}"

    # Test with episode_length smaller than max_length
    result = base_trainer.compute_dynamic_discount(2000)
    assert result > 0.999 and result < 0.9999, f"Unexpected result: {result}"

    # Test with episode_length close to min_gamma
    result = base_trainer.compute_dynamic_discount(100)
    assert 0.999 < result and result < 0.9991, f"Expected min_gamma, but got {result}"

    # Test with episode_length very close to max_length
    result = base_trainer.compute_dynamic_discount(9999)
    assert result < 0.9999 and result > 0.999, f"Unexpected result: {result}"
