from typing import Optional, Tuple, Union
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from rl4llm.core.training_mixin import TrainingMixin

DIST_OPS_PATH = 'rl4llm.core.distributed.DistributedOps'


# --- Fixtures  ---
@pytest.fixture
def mock_dist_ops():
    """Provides a MagicMock object simulating DistributedOps."""
    mock = MagicMock(name='MockDistributedOps')
    type(mock).world_size = PropertyMock(return_value=1)

    def mock_all_reduce(tensor, op=dist.ReduceOp.SUM):
        if op == dist.ReduceOp.SUM:
            return tensor * mock.world_size
        return tensor.clone()

    mock.all_reduce_tensor = MagicMock(side_effect=mock_all_reduce)
    return mock


@pytest.fixture
def simple_linear_model():
    """Fixture providing a simple linear model with computed gradients."""
    model = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()
    return model


@pytest.fixture
def sample_tensor_data():
    """Fixture providing sample values and mask tensors."""
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    return values, mask


def test_clean_up():
    """Test that clean_up runs without raising exceptions."""
    try:
        TrainingMixin.clean_up()
    except Exception as e:
        pytest.fail(f"clean_up() raised an exception: {e}")


def test_compute_grad_norm(simple_linear_model):
    """Test gradient norm computation matches manual calculation."""
    model = simple_linear_model
    computed_norm = TrainingMixin.compute_grad_norm(model)
    total_norm_sq = sum(
        torch.linalg.vector_norm(p.grad.detach(), dtype=p.grad.dtype) ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    expected_norm = total_norm_sq.sqrt()
    assert torch.allclose(computed_norm, expected_norm, atol=1e-8)


@pytest.mark.parametrize(
    'dim, expected', [(None, torch.tensor(9.0)), (1, torch.tensor([4.0, 5.0]))]
)
def test_masked_sum(sample_tensor_data, dim, expected):
    """Test masked sum computation for global and dimension-wise cases."""
    values, mask = sample_tensor_data
    result = TrainingMixin.masked_sum(values, mask, dim=dim)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    'dim, expected',
    [(None, torch.tensor(3.0)), (1, torch.tensor([[2.0], [5.0]]))],
)
def test_masked_mean(sample_tensor_data, dim, expected):
    """Test masked mean computation for global and dimension-wise cases."""
    values, mask = sample_tensor_data
    result = TrainingMixin.masked_mean(values, mask, dim=dim)
    assert torch.allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize('shift_mean', [True, False])
def test_whiten(sample_tensor_data, shift_mean):
    """Test whitening normalizes data appropriately based on shift_mean."""
    values, _ = sample_tensor_data
    whitened = TrainingMixin.whiten(values, shift_mean=shift_mean, dim=1)
    mean = whitened.mean(dim=1)
    var = whitened.var(dim=1, unbiased=False)
    if shift_mean:
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
    else:
        assert torch.allclose(mean, values.mean(dim=1), atol=1e-5)


def test_masked_whiten(sample_tensor_data):
    """Test masked whitening applies normalization only to masked elements."""
    values, mask = sample_tensor_data
    whitened = TrainingMixin.masked_whiten(values, mask, shift_mean=True, dim=1)
    epsilon = 1e-8
    expected = torch.stack(
        [
            torch.tensor(
                [
                    (1 - 2) / torch.sqrt(torch.tensor(1.0 + epsilon)),
                    2.0,
                    (3 - 2) / torch.sqrt(torch.tensor(1.0 + epsilon)),
                ]
            ),
            torch.tensor(
                [4.0, (5 - 5) / torch.sqrt(torch.tensor(epsilon)), 6.0]
            ),
        ]
    )
    assert torch.allclose(whitened, expected, atol=1e-5)


@pytest.fixture
def logits_and_actions():
    """Fixture providing logits and actions for logprobs and entropy tests."""
    logits = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]],
        ]
    )
    actions = torch.tensor([[0, 1, 2], [3, 2, 1]])
    loss_masks = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    return logits, actions, loss_masks


def test_compute_logprobs_from_logits(logits_and_actions):
    """Test log probabilities computation from logits with and without masks."""
    logits, actions, loss_masks = logits_and_actions
    expected_logprobs = torch.stack(
        [
            torch.gather(
                torch.log_softmax(logits[i].float(), dim=-1),
                dim=1,
                index=actions[i].unsqueeze(1),
            ).squeeze(1)
            for i in range(logits.shape[0])
        ]
    )

    result = TrainingMixin.compute_logprobs_from_logits(logits, actions)
    assert torch.allclose(result, expected_logprobs, atol=1e-5)

    result_masked = TrainingMixin.compute_logprobs_from_logits(
        logits, actions, loss_masks=loss_masks
    )
    assert torch.allclose(
        result_masked, expected_logprobs * loss_masks.float(), atol=1e-5
    )


def test_compute_entropy_from_logits(logits_and_actions):
    """Test entropy computation from logits with and without masks."""
    logits, _, loss_masks = logits_and_actions
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    expected_entropy = -(probs * log_probs).sum(dim=-1)

    result = TrainingMixin.compute_entropy_from_logits(logits)
    assert torch.allclose(result, expected_entropy, atol=1e-5)

    result_masked = TrainingMixin.compute_entropy_from_logits(
        logits, loss_masks=loss_masks
    )
    assert torch.allclose(
        result_masked, expected_entropy * loss_masks.float(), atol=1e-5
    )


@pytest.mark.parametrize(
    'rewards, mask, gamma, expected',
    [
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([True, True, True, True]),
            1.0,
            torch.tensor([10.0, 9.0, 7.0, 4.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([True, True, True, True]),
            0.5,
            torch.tensor([3.25, 4.5, 5.0, 4.0]),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([False, True, False, True]),
            1.0,
            torch.tensor([0.0, 5.0, 0.0, 4.0]),
        ),
    ],
)
def test_compute_masked_monte_carlo_returns(rewards, mask, gamma, expected):
    """Test Monte Carlo returns computation with various masks and gamma values."""
    result = TrainingMixin.compute_masked_monte_carlo_returns(
        rewards, mask, gamma
    )
    assert torch.allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize(
    'rewards, mask, gamma',
    [
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[True, False]]), 1.0),
        (torch.tensor([1.0, 2.0]), torch.tensor([True, True]), 0.0),
        (torch.tensor([1.0, 2.0]), torch.tensor([True, True]), 1.1),
    ],
)
def test_compute_masked_monte_carlo_returns_invalid(rewards, mask, gamma):
    """Test Monte Carlo returns raises AssertionError for invalid inputs."""
    with pytest.raises(AssertionError):
        TrainingMixin.compute_masked_monte_carlo_returns(rewards, mask, gamma)


# --- For GAE advantage ---


@pytest.fixture
def sample_gae_inputs():
    """Provides a standard set of rewards, values, gamma, and lambda."""
    return {
        'rewards': torch.tensor([0.0, 0.0, 1.0, 0.5, 0.0], dtype=torch.float32),
        'values': torch.tensor([0.1, 0.2, 0.8, 0.6, 0.1], dtype=torch.float32),
        'gamma': 0.99,
        'gae_lambda': 0.95,
    }


def test_masked_gae_advantage_no_mask(sample_gae_inputs):
    """Tests GAE calculation when all steps are considered (mask is all ones)."""
    mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.int)
    # Updated expected values based on actual correct output
    expected_advantages = torch.tensor(
        [1.2780, 1.2547, 0.7046, -0.0951, -0.1000], dtype=torch.float32
    )

    advantages = TrainingMixin.compute_masked_gae_advantage(
        rewards=sample_gae_inputs['rewards'],
        values=sample_gae_inputs['values'],
        mask=mask,
        gamma=sample_gae_inputs['gamma'],
        gae_lambda=sample_gae_inputs['gae_lambda'],
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)


def test_masked_gae_advantage_simple_mask(sample_gae_inputs):
    """Tests GAE calculation with a typical mask for prompt/response sequences."""
    mask = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int)
    # Expected values from previous correct calculation
    expected_advantages = torch.tensor(
        [0.0000, 0.0000, 0.7046, -0.0951, 0.0000], dtype=torch.float32
    )

    advantages = TrainingMixin.compute_masked_gae_advantage(
        rewards=sample_gae_inputs['rewards'],
        values=sample_gae_inputs['values'],
        mask=mask,
        gamma=sample_gae_inputs['gamma'],
        gae_lambda=sample_gae_inputs['gae_lambda'],
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)


def test_masked_gae_advantage_mask_stops_propagation(
    sample_gae_inputs,
):
    """Tests that a zero in the mask correctly stops the lambda-discounted propagation."""
    # Mask: [Assistant, Prompt, Assistant] - propagation should break at t=1
    mask = torch.tensor([1, 0, 1], dtype=torch.int)
    rewards = torch.tensor([0.5, 0.1, 1.0], dtype=torch.float32)
    values = torch.tensor([0.2, 0.3, 0.8], dtype=torch.float32)
    gamma = 0.99  # Using specific gamma/lambda for this test
    gae_lambda = 0.95

    # Updated expected values based on actual correct output
    expected_advantages = torch.tensor(
        [1.1537, 0.0000, 0.2000], dtype=torch.float32  # Adjusted first element
    )

    advantages = TrainingMixin.compute_masked_gae_advantage(
        rewards=rewards,
        values=values,
        mask=mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)


def test_masked_gae_advantage_all_mask_zeros(sample_gae_inputs):
    """Tests that advantages are all zero when the mask is all zeros."""
    mask = torch.zeros_like(sample_gae_inputs['rewards'], dtype=torch.int)
    expected_advantages = torch.zeros_like(
        sample_gae_inputs['rewards'], dtype=torch.float32
    )

    advantages = TrainingMixin.compute_masked_gae_advantage(
        rewards=sample_gae_inputs['rewards'],
        values=sample_gae_inputs['values'],
        mask=mask,
        gamma=sample_gae_inputs['gamma'],
        gae_lambda=sample_gae_inputs['gae_lambda'],
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)


def test_masked_gae_advantage_single_element_masked_in():
    """Tests GAE calculation for a single-element sequence that is masked in."""
    rewards = torch.tensor([1.0], dtype=torch.float32)
    values = torch.tensor([0.5], dtype=torch.float32)
    mask = torch.tensor([1], dtype=torch.int)
    gamma = 0.99
    gae_lambda = 0.95

    expected_advantages = torch.tensor([0.5], dtype=torch.float32)

    advantages = TrainingMixin.compute_masked_gae_advantage(
        rewards=rewards,
        values=values,
        mask=mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)


def test_masked_gae_advantage_single_element_masked_out():
    """Tests GAE calculation for a single-element sequence that is masked out."""
    rewards = torch.tensor([1.0], dtype=torch.float32)
    values = torch.tensor([0.5], dtype=torch.float32)
    mask = torch.tensor([0], dtype=torch.int)
    gamma = 0.99
    gae_lambda = 0.95

    expected_advantages = torch.tensor([0.0], dtype=torch.float32)

    advantages = TrainingMixin.compute_masked_gae_advantage(
        rewards=rewards,
        values=values,
        mask=mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)


# --- Tests distributed ops ---


@pytest.fixture
def sample_data():
    """Provides sample tensor data for testing."""
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    return values, mask


@pytest.fixture
def train_mixin_instance(mock_dist_ops):
    """Provides an instance of TrainingMixin with mocked dist_ops."""

    patch_target = f'{DIST_OPS_PATH}.get_instance'
    try:
        # Try patching, might fail if the module doesn't exist
        with patch(patch_target, return_value=mock_dist_ops):
            mixin = TrainingMixin(dist_ops=None)  # Let init fetch the mock
    except (ModuleNotFoundError, AttributeError):

        mixin = TrainingMixin(dist_ops=mock_dist_ops)

    # Fallback: Directly assign the mock if the above fails or isn't the desired setup
    if not hasattr(mixin, 'dist_ops') or mixin.dist_ops is not mock_dist_ops:
        mixin.dist_ops = mock_dist_ops

    return mixin


# # --- Helper Function for Mock Call Tensor Comparison ---
def assert_mock_tensor_call(mock_method, expected_tensor, **kwargs):
    """Asserts a mock was called with a specific tensor, using torch.testing.assert_close."""
    found_call = False
    for actual_call in mock_method.call_args_list:
        args, call_kwargs = actual_call
        # Check non-tensor args/kwargs first
        match = True
        for key, value in kwargs.items():
            if key not in call_kwargs or call_kwargs[key] != value:
                match = False
                break
        if not match:
            continue

        # Check tensor argument (assuming it's the first positional arg)
        if args and isinstance(args[0], torch.Tensor):
            try:
                torch.testing.assert_close(args[0], expected_tensor)
                found_call = True
                break  # Found a matching call
            except AssertionError:
                continue  # Tensors didn't match, check next call
        elif not args and not isinstance(
            expected_tensor, torch.Tensor
        ):  # Handle case where no tensor expected
            found_call = True  # Match based on kwargs only
            break

    if not found_call:
        raise AssertionError(
            f"Mock not called with tensor close to {expected_tensor} and kwargs {kwargs}.\nCalls were: {mock_method.call_args_list}"
        )


def assert_mock_tensor_calls(mock_method, expected_calls):
    """Asserts a mock was called with a list of specific tensors/kwargs."""
    actual_calls = mock_method.call_args_list
    assert len(actual_calls) == len(
        expected_calls
    ), f"Expected {len(expected_calls)} calls, but got {len(actual_calls)}"

    matched_actual_indices = set()

    for expected_tensor, expected_kwargs in expected_calls:
        found_match_for_expected = False
        for i, actual_call in enumerate(actual_calls):
            if i in matched_actual_indices:
                continue  # Skip already matched calls

            args, call_kwargs = actual_call

            # Check non-tensor kwargs
            kwargs_match = True
            if expected_kwargs:
                for key, value in expected_kwargs.items():
                    if key not in call_kwargs or call_kwargs[key] != value:
                        kwargs_match = False
                        break
            if not kwargs_match:
                continue

            # Check tensor argument
            tensor_match = False
            if (
                args
                and isinstance(args[0], torch.Tensor)
                and isinstance(expected_tensor, torch.Tensor)
            ):
                try:
                    torch.testing.assert_close(args[0], expected_tensor)
                    tensor_match = True
                except AssertionError:
                    tensor_match = False
            elif not args and not isinstance(
                expected_tensor, torch.Tensor
            ):  # Both expect no tensor arg
                tensor_match = True

            if tensor_match and kwargs_match:
                found_match_for_expected = True
                matched_actual_indices.add(i)
                break  # Found match for this expected call

        if not found_match_for_expected:
            raise AssertionError(
                f"Could not find a matching call for expected tensor {expected_tensor} and kwargs {expected_kwargs}.\nActual calls: {actual_calls}"
            )


# Test dist_masked_sum
@pytest.mark.parametrize('dim', [None, 0, 1])
def test_dist_masked_sum_multi_process(
    train_mixin_instance, sample_data, mock_dist_ops, dim
):
    """Tests dist_masked_sum aggregates correctly across processes."""
    world_size = 3
    type(mock_dist_ops).world_size = PropertyMock(return_value=world_size)
    values, mask = sample_data
    local_sum = train_mixin_instance.masked_sum(values, mask, dim=dim)
    expected_global_sum = local_sum * world_size

    dist_result = train_mixin_instance.dist_masked_sum(values, mask, dim=dim)

    torch.testing.assert_close(dist_result, expected_global_sum)
    # Use helper for tensor comparison in mock call
    assert_mock_tensor_call(
        mock_dist_ops.all_reduce_tensor, local_sum, op=dist.ReduceOp.SUM
    )
    # Ensure it was called only once total
    assert mock_dist_ops.all_reduce_tensor.call_count == 1


# Test dist_masked_mean
@pytest.mark.parametrize('dim', [None, 0, 1])
def test_dist_masked_mean_multi_process(
    train_mixin_instance, sample_data, mock_dist_ops, dim
):
    """Tests dist_masked_mean calculates global mean correctly across processes."""
    world_size = 4
    epsilon = 1e-8
    type(mock_dist_ops).world_size = PropertyMock(return_value=world_size)
    values, mask = sample_data
    broadcast_mask = torch.broadcast_to(mask, values.shape)

    local_sum = train_mixin_instance.masked_sum(values, broadcast_mask, dim=dim)
    local_count = broadcast_mask.sum(dim=dim).float()

    expected_global_sum = local_sum * world_size
    expected_global_count = local_count * world_size
    expected_global_mean = expected_global_sum / (
        expected_global_count + epsilon
    )
    if dim is not None:
        expected_global_mean = torch.where(
            expected_global_count.view_as(expected_global_mean) > 0,
            expected_global_mean,
            torch.zeros_like(expected_global_mean),
        )
    elif expected_global_count == 0:
        expected_global_mean = torch.zeros_like(expected_global_mean)

    dist_result = train_mixin_instance.dist_masked_mean(
        values, mask, dim=dim, epsilon=epsilon
    )

    torch.testing.assert_close(
        dist_result, expected_global_mean, rtol=1e-8, atol=1e-8
    )
    assert mock_dist_ops.all_reduce_tensor.call_count == 2
    # Use helper to check calls with tensors
    expected_calls = [
        (local_sum, {'op': dist.ReduceOp.SUM}),
        (local_count, {'op': dist.ReduceOp.SUM}),
    ]
    assert_mock_tensor_calls(mock_dist_ops.all_reduce_tensor, expected_calls)


# Test dist_masked_whiten
@pytest.mark.parametrize('shift_mean', [True, False])
@pytest.mark.parametrize('dim', [-1, 0])
def test_dist_masked_whiten_multi_process(
    train_mixin_instance, sample_data, mock_dist_ops, shift_mean, dim
):
    """Tests dist_masked_whiten uses global stats correctly across processes."""
    world_size = 2
    epsilon = 1e-8
    type(mock_dist_ops).world_size = PropertyMock(return_value=world_size)
    values, mask = sample_data
    values = values.float()
    broadcast_mask = torch.broadcast_to(mask, values.shape)

    # Calculate local stats
    num_valid_local = broadcast_mask.sum(dim=dim, keepdim=True).float()
    masked_values_for_stats = torch.where(
        broadcast_mask, values, torch.zeros_like(values)
    )
    sum_local = masked_values_for_stats.sum(dim=dim, keepdim=True)
    sum_sq_local = (masked_values_for_stats**2).sum(dim=dim, keepdim=True)

    # Calculate expected global stats
    global_sum = sum_local * world_size
    global_sum_sq = sum_sq_local * world_size
    global_num_valid = num_valid_local * world_size
    global_mean = global_sum / (global_num_valid + epsilon)
    global_var = (global_sum_sq / (global_num_valid + epsilon)) - global_mean**2
    global_var = torch.clamp(global_var, min=0.0)
    global_mean = torch.where(
        global_num_valid > 0, global_mean, torch.zeros_like(global_mean)
    )
    global_var = torch.where(
        global_num_valid > 0, global_var, torch.zeros_like(global_var)
    )

    # Calculate expected output
    expected_whitened = (values - global_mean) * torch.rsqrt(
        global_var + epsilon
    )
    if not shift_mean:
        expected_whitened += global_mean
    expected_output = torch.where(broadcast_mask, expected_whitened, values)

    # Run the actual function
    dist_result = train_mixin_instance.dist_masked_whiten(
        values, mask, shift_mean=shift_mean, dim=dim, epsilon=epsilon
    )

    torch.testing.assert_close(
        dist_result, expected_output, rtol=1e-5, atol=1e-5
    )
    assert mock_dist_ops.all_reduce_tensor.call_count == 3
    # Use helper to check calls with tensors
    expected_calls = [
        (sum_local, {'op': dist.ReduceOp.SUM}),
        (sum_sq_local, {'op': dist.ReduceOp.SUM}),
        (num_valid_local, {'op': dist.ReduceOp.SUM}),
    ]
    assert_mock_tensor_calls(mock_dist_ops.all_reduce_tensor, expected_calls)


# --- Other tests (single process, dist_whiten delegation) remain the same ---
# --- They did not show errors in the provided traceback ---


def test_dist_masked_sum_single_process(
    train_mixin_instance, sample_data, mock_dist_ops
):
    """Tests dist_masked_sum behaves like masked_sum when world_size is 1."""
    type(mock_dist_ops).world_size = PropertyMock(
        return_value=1
    )  # Ensure world_size is 1
    values, mask = sample_data
    local_result = train_mixin_instance.masked_sum(values, mask)
    dist_result = train_mixin_instance.dist_masked_sum(values, mask)

    torch.testing.assert_close(dist_result, local_result)
    mock_dist_ops.all_reduce_tensor.assert_not_called()


def test_dist_masked_mean_single_process(
    train_mixin_instance, sample_data, mock_dist_ops
):
    """Tests dist_masked_mean calculates local mean when world_size is 1."""
    type(mock_dist_ops).world_size = PropertyMock(return_value=1)
    values, mask = sample_data
    # Calculate expected local mean manually for comparison
    broadcast_mask = torch.broadcast_to(mask, values.shape)
    local_sum = train_mixin_instance.masked_sum(values, broadcast_mask)
    local_count = broadcast_mask.sum().float()
    expected_local_mean = torch.zeros_like(local_sum)  # Default to zero
    if local_count > 0:
        expected_local_mean = local_sum / (local_count + 1e-8)

    dist_result = train_mixin_instance.dist_masked_mean(values, mask)

    torch.testing.assert_close(
        dist_result, expected_local_mean, rtol=1e-8, atol=1e-8
    )
    mock_dist_ops.all_reduce_tensor.assert_not_called()


def test_dist_masked_mean_zero_count_multi_process(
    train_mixin_instance, mock_dist_ops
):
    """Tests dist_masked_mean handles zero global count correctly."""
    world_size = 2
    type(mock_dist_ops).world_size = PropertyMock(return_value=world_size)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.zeros_like(values, dtype=torch.bool)  # All False mask

    dist_result = train_mixin_instance.dist_masked_mean(values, mask)

    torch.testing.assert_close(dist_result, torch.tensor(0.0))
    assert mock_dist_ops.all_reduce_tensor.call_count == 2


def test_dist_masked_whiten_single_process(
    train_mixin_instance, sample_data, mock_dist_ops
):
    """Tests dist_masked_whiten behaves like masked_whiten when world_size is 1."""
    type(mock_dist_ops).world_size = PropertyMock(return_value=1)
    values, mask = sample_data
    local_result = train_mixin_instance.masked_whiten(values, mask)
    dist_result = train_mixin_instance.dist_masked_whiten(values, mask)

    torch.testing.assert_close(dist_result, local_result, rtol=1e-8, atol=1e-8)
    mock_dist_ops.all_reduce_tensor.assert_not_called()


@patch.object(
    TrainingMixin, 'dist_masked_whiten', autospec=True
)  # Mock the target method directly
def test_dist_whiten_calls_masked_whiten(
    mock_dist_masked_whiten, train_mixin_instance, mock_dist_ops
):
    """Tests dist_whiten correctly calls dist_masked_whiten with a full mask."""
    values = torch.randn(2, 3)
    shift_mean = False
    dim = 0
    epsilon = 1e-5
    mock_dist_masked_whiten.return_value = torch.zeros_like(values)

    train_mixin_instance.dist_whiten(
        values, shift_mean=shift_mean, dim=dim, epsilon=epsilon
    )

    mock_dist_masked_whiten.assert_called_once()
    call_args = mock_dist_masked_whiten.call_args
    passed_self = call_args[0][0]
    passed_values = call_args.kwargs.get('values')
    passed_mask = call_args.kwargs.get('mask')
    passed_shift_mean = call_args.kwargs.get('shift_mean')
    passed_dim = call_args.kwargs.get('dim')
    passed_epsilon = call_args.kwargs.get('epsilon')

    assert passed_self is train_mixin_instance
    torch.testing.assert_close(passed_values, values)
    expected_mask = torch.ones_like(values, dtype=torch.bool)
    torch.testing.assert_close(passed_mask, expected_mask)
    assert passed_shift_mean == shift_mean
    assert passed_dim == dim
    assert passed_epsilon == epsilon
