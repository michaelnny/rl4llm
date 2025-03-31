import gc

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4llm.core.training_mixin import TrainingMixin


def test_clean_up():
    # Test that clean_up() runs without errors.
    try:
        TrainingMixin.clean_up()
    except Exception as e:
        pytest.fail(f"clean_up() raised an exception: {e}")


def test_compute_grad_norm():
    # Create a simple linear model and compute gradients.
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()

    computed_norm = TrainingMixin.compute_grad_norm(model)
    # Manually compute the expected gradient norm.
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            total_norm_sq += (
                torch.linalg.vector_norm(grad, dtype=grad.dtype) ** 2
            )
    expected_norm = total_norm_sq.sqrt()
    assert torch.allclose(computed_norm, expected_norm, atol=1e-6)


def test_masked_sum():
    values = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    # Global sum: 1 + 3 + 5 = 9
    result_global = TrainingMixin.masked_sum(values, mask)
    expected_global = torch.tensor(9.0)
    assert torch.allclose(result_global, expected_global)

    # Sum along dim=1: [1+3, 5] = [4, 5]
    result_dim = TrainingMixin.masked_sum(values, mask, dim=1)
    expected_dim = torch.tensor([4.0, 5.0])
    assert torch.allclose(result_dim, expected_dim)


def test_masked_mean():
    values = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    # Global mean: (1+3+5)/3 = 3.0
    result_global = TrainingMixin.masked_mean(values, mask)
    expected_global = torch.tensor(3.0)
    assert torch.allclose(result_global, expected_global, atol=1e-6)

    # Mean along dim=1: note that the implementation keeps the reduced dimension.
    # For first row: (1+3)/2 = 2, and for second row: (5)/1 = 5.
    # Expected shape is [2, 1] due to keepdim=True.
    result_dim = TrainingMixin.masked_mean(values, mask, dim=1)
    expected_dim = torch.tensor([[2.0], [5.0]])
    assert torch.allclose(result_dim, expected_dim, atol=1e-6)


def test_whiten():
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # With shift_mean True, the output should have zero mean and unit variance along the specified dim.
    whitened = TrainingMixin.whiten(values, shift_mean=True, dim=1)
    mean = whitened.mean(dim=1)
    var = whitened.var(dim=1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

    # With shift_mean False, the mean of the output should match the original mean.
    whitened_ns = TrainingMixin.whiten(values, shift_mean=False, dim=1)
    original_mean = values.mean(dim=1)
    mean_ns = whitened_ns.mean(dim=1)
    assert torch.allclose(mean_ns, original_mean, atol=1e-5)


def test_masked_whiten():
    # Define a small tensor and a mask.
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    # For the first row, valid indices are 0 and 2: [1, 3] with mean=2 and var=1.
    # For the second row, only index 1 is valid: [5] -> mean=5, var≈0.
    whitened = TrainingMixin.masked_whiten(values, mask, shift_mean=True, dim=1)
    epsilon = 1e-8
    expected_row0 = torch.tensor(
        [
            (1 - 2) / torch.sqrt(torch.tensor(1.0 + epsilon)),
            2.0,  # unchanged (mask False)
            (3 - 2) / torch.sqrt(torch.tensor(1.0 + epsilon)),
        ]
    )
    expected_row1 = torch.tensor(
        [
            4.0,  # unchanged
            (5 - 5) / torch.sqrt(torch.tensor(epsilon)),  # becomes 0
            6.0,  # unchanged
        ]
    )
    expected = torch.stack([expected_row0, expected_row1])
    assert torch.allclose(whitened, expected, atol=1e-5)


def test_compute_logprobs_from_logits():
    # Define a logits tensor with shape [batch_size, seq_len, vocab_size]
    logits = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]],
        ]
    )
    actions = torch.tensor([[0, 1, 2], [3, 2, 1]])
    # Compute expected log probabilities using torch.log_softmax and torch.gather.
    expected_logprobs = []
    for i in range(logits.shape[0]):
        sample_logits = logits[i].float()
        log_probs = torch.log_softmax(sample_logits, dim=-1)
        sample_actions = actions[i].unsqueeze(1)
        sample_logprob = torch.gather(
            log_probs, dim=1, index=sample_actions
        ).squeeze(1)
        expected_logprobs.append(sample_logprob)
    expected = torch.stack(expected_logprobs)
    result = TrainingMixin.compute_logprobs_from_logits(logits, actions)
    assert torch.allclose(result, expected, atol=1e-5)

    # Also test with loss_masks applied.
    loss_masks = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    result_masked = TrainingMixin.compute_logprobs_from_logits(
        logits, actions, loss_masks=loss_masks
    )
    expected_masked = expected * loss_masks.float()
    assert torch.allclose(result_masked, expected_masked, atol=1e-5)


def test_compute_entropy_from_logits():
    # Define logits tensor.
    logits = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]],
        ]
    )
    # Compute expected entropy: -sum(p * log(p)) along the vocab dimension.
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    expected_entropy = -(probs * log_probs).sum(dim=-1)
    result = TrainingMixin.compute_entropy_from_logits(logits)
    assert torch.allclose(result, expected_entropy, atol=1e-5)

    # Test with loss_masks.
    loss_masks = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    expected_entropy_masked = expected_entropy * loss_masks.float()
    result_masked = TrainingMixin.compute_entropy_from_logits(
        logits, loss_masks=loss_masks
    )
    assert torch.allclose(result_masked, expected_entropy_masked, atol=1e-5)


def test_compute_masked_monte_carlo_returns():
    # Test 1: All mask True with gamma=1.0.
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, True, True, True])
    gamma = 1.0
    # Calculation (backwards):
    # t=3: returns[3] = 4
    # t=2: returns[2] = 3 + 1*4 = 7
    # t=1: returns[1] = 2 + 1*7 = 9
    # t=0: returns[0] = 1 + 1*9 = 10
    expected = torch.tensor([10.0, 9.0, 7.0, 4.0])
    result = TrainingMixin.compute_masked_monte_carlo_returns(
        rewards, mask, gamma
    )
    assert torch.allclose(result, expected, atol=1e-5)

    # Test 2: All mask True with gamma=0.5.
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    gamma = 0.5
    # t=3: returns[3] = 4
    # t=2: returns[2] = 3 + 0.5*4 = 5
    # t=1: returns[1] = 2 + 0.5*5 = 4.5
    # t=0: returns[0] = 1 + 0.5*4.5 = 3.25
    expected = torch.tensor([3.25, 4.5, 5.0, 4.0])
    result = TrainingMixin.compute_masked_monte_carlo_returns(
        rewards, mask, gamma
    )
    assert torch.allclose(result, expected, atol=1e-5)

    # Test 3: Mixed mask values.
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([False, True, False, True])
    gamma = 1.0
    # Calculation:
    # t=3: returns[3] = 4        (mask True)
    # t=2: returns[2] = 3        (mask False)
    # t=1: returns[1] = 2 + 1*3 = 5   (mask True, using g from t=2)
    # t=0: returns[0] = 1        (mask False)
    # After multiplying by mask: [0, 5, 0, 4]
    expected = torch.tensor([0.0, 5.0, 0.0, 4.0])
    result = TrainingMixin.compute_masked_monte_carlo_returns(
        rewards, mask, gamma
    )
    assert torch.allclose(result, expected, atol=1e-5)

    # Test 4: Invalid input dimensions (non-1D tensors).
    rewards = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[True, False]])
    with pytest.raises(AssertionError):
        TrainingMixin.compute_masked_monte_carlo_returns(rewards, mask, gamma)

    # Test 5: Invalid gamma (0.0 is not allowed).
    rewards = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, True, True])
    with pytest.raises(AssertionError):
        TrainingMixin.compute_masked_monte_carlo_returns(rewards, mask, 0.0)

    # Test 6: Invalid gamma (> 1.0).
    with pytest.raises(AssertionError):
        TrainingMixin.compute_masked_monte_carlo_returns(rewards, mask, 1.1)
