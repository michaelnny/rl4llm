from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch

from rl4llm.core.deepspeed_mixin import DeepSpeedUtilsMixin

# Attempt to import deepspeed and skip tests if unavailable
try:
    import deepspeed
except ImportError:
    deepspeed = None


# --- Define a dummy PreTrainedModel for type hinting and spec ---
# This is better than mocking it globally for spec usage.
class DummyPreTrainedModel:
    def save_pretrained(self, output_dir: str):
        pass  # Dummy method for spec


# --- Fixtures ---

# Skip all tests in this module if deepspeed is not installed
pytestmark = pytest.mark.skipif(
    deepspeed is None, reason='deepspeed not installed'
)


@pytest.fixture
def mixin_instance() -> DeepSpeedUtilsMixin:
    """Provides an instance of the DeepSpeedUtilsMixin."""
    return DeepSpeedUtilsMixin()


@pytest.fixture
def mock_engine(mocker) -> MagicMock:
    """Provides a mock DeepSpeedEngine object."""
    # Use autospec=True if deepspeed is installed to get better mocking,
    # otherwise fall back to a basic MagicMock.
    engine = mocker.MagicMock(
        spec=deepspeed.DeepSpeedEngine if deepspeed else None
    )

    # Create a standard MagicMock for the module, don't spec it with another mock.
    engine.module = mocker.MagicMock()
    # Ensure the mocked module has the save_pretrained method for tests that need it
    engine.module.save_pretrained = mocker.MagicMock()

    # Provide default implementations for methods called by the mixin
    engine.parameters = mocker.MagicMock(
        return_value=[torch.nn.Parameter(torch.zeros(1))]
    )
    engine.zero_offload_param = mocker.MagicMock(return_value=None)
    engine.zero_offload_optimizer = mocker.MagicMock(return_value=None)
    engine.bfloat16_enabled = mocker.MagicMock(return_value=False)
    engine.fp16_enabled = mocker.MagicMock(return_value=False)
    engine.zero_optimization_stage = mocker.MagicMock(
        return_value=0
    )  # Default to no ZeRO
    return engine


@pytest.fixture
def mock_distributed(mocker):
    """Provides mocks for torch.distributed functions."""
    # Patch torch.distributed using mocker for convenience
    mock_dist = mocker.patch(
        'torch.distributed', create=True
    )  # create=True allows patching even if not imported
    mock_dist.get_rank.return_value = 0  # Default to rank 0
    mock_dist.barrier = mocker.MagicMock()  # Mock the barrier function
    return mock_dist


@pytest.fixture
def mock_gathered_parameters(mocker):
    """Provides a mock for deepspeed.zero.GatheredParameters context manager."""
    # Mock the class itself to control its instantiation and context management
    mock_context = MagicMock()
    # Use mocker.patch.object if deepspeed.zero exists, otherwise mock directly
    if hasattr(deepspeed, 'zero'):
        mock_class = mocker.patch.object(
            deepspeed.zero, 'GatheredParameters', return_value=mock_context
        )
    else:  # Fallback if deepspeed or deepspeed.zero doesn't exist (though skipped)
        mock_class = mocker.patch(
            'deepspeed.zero.GatheredParameters',
            return_value=mock_context,
            create=True,
        )

    # Ensure the context manager protocol is followed
    mock_context.__enter__.return_value = None
    mock_context.__exit__.return_value = None
    return mock_class  # Return the mock class to check calls to it


# --- Tests ---


def test_with_unwrapped_model_non_zero3(
    mixin_instance, mock_engine, mock_gathered_parameters
):
    """Tests unwrapping the model when ZeRO-3 is not enabled."""
    mock_engine.zero_optimization_stage.return_value = 2  # Not ZeRO-3

    with mixin_instance.with_unwrapped_model(mock_engine) as model:
        assert model is mock_engine.module
    mock_gathered_parameters.assert_not_called()  # Ensure GatheredParameters wasn't used


def test_with_unwrapped_model_zero3(
    mixin_instance, mock_engine, mock_gathered_parameters
):
    """Tests unwrapping the model when ZeRO-3 is enabled."""
    mock_engine.zero_optimization_stage.return_value = 3  # ZeRO-3 enabled

    with mixin_instance.with_unwrapped_model(mock_engine) as model:
        assert model is mock_engine.module
    # Check that GatheredParameters was called correctly
    mock_gathered_parameters.assert_called_once_with(mock_engine.parameters())
    # Check that the context manager was entered
    mock_gathered_parameters.return_value.__enter__.assert_called_once()
    mock_gathered_parameters.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize(
    'stage, expected',
    [(3, True), (2, False), (1, False), (0, False)],
    ids=['stage3', 'stage2', 'stage1', 'stage0'],
)
def test_is_zero3_enabled(mixin_instance, mock_engine, stage, expected):
    """Tests the ZeRO-3 check for different optimization stages."""
    mock_engine.zero_optimization_stage.return_value = stage
    assert mixin_instance.is_zero3_enabled(mock_engine) is expected


@pytest.mark.parametrize(
    'stage, expected',
    [(2, True), (3, False), (1, False), (0, False)],
    ids=['stage2', 'stage3', 'stage1', 'stage0'],
)
def test_is_zero2_enabled(mixin_instance, mock_engine, stage, expected):
    """Tests the ZeRO-2 check for different optimization stages."""
    mock_engine.zero_optimization_stage.return_value = stage
    assert mixin_instance.is_zero2_enabled(mock_engine) is expected


@pytest.mark.parametrize(
    'stage, offload_device, expected',
    [
        (3, 'cpu', True),
        (3, 'nvme', True),
        (3, 'gpu', False),  # Invalid device for offload check
        (3, None, False),  # Offload config is None
        (2, 'cpu', False),  # Not ZeRO-3
        (0, 'cpu', False),  # Not ZeRO-3
    ],
    ids=['z3_cpu', 'z3_nvme', 'z3_gpu', 'z3_none', 'z2_cpu', 'z0_cpu'],
)
def test_is_params_offload_enabled(
    mocker, mixin_instance, mock_engine, stage, offload_device, expected
):
    """Tests the parameter offload check under various conditions."""
    mock_engine.zero_optimization_stage.return_value = stage
    if offload_device is not None:
        # Mock the config object returned by zero_offload_param
        offload_config = MagicMock()
        offload_config.device = offload_device
        mock_engine.zero_offload_param.return_value = offload_config
    else:
        mock_engine.zero_offload_param.return_value = None

    assert mixin_instance.is_params_offload_enabled(mock_engine) is expected


@pytest.mark.parametrize(
    'offload_device, expected',
    [
        ('cpu', True),
        ('nvme', True),
        ('gpu', False),  # Invalid device for offload check
        (None, False),  # Offload config is None
    ],
    ids=['cpu', 'nvme', 'gpu', 'none'],
)
def test_is_optimizer_offload_enabled(
    mocker, mixin_instance, mock_engine, offload_device, expected
):
    """Tests the optimizer offload check for different devices."""
    if offload_device is not None:
        # Mock the config object returned by zero_offload_optimizer
        offload_config = MagicMock()
        offload_config.device = offload_device
        mock_engine.zero_offload_optimizer.return_value = offload_config
    else:
        mock_engine.zero_offload_optimizer.return_value = None

    assert mixin_instance.is_optimizer_offload_enabled(mock_engine) is expected


@pytest.mark.parametrize(
    'stage, opt_offload_device, expected',
    [
        (3, None, True),  # Z3, optimizer not offloaded -> can offload state
        (3, 'cpu', False),  # Z3, optimizer offloaded -> cannot offload state
        (3, 'nvme', False),  # Z3, optimizer offloaded -> cannot offload state
        (2, None, False),  # Z2 -> cannot offload state
        (0, 'cpu', False),  # Z0 -> cannot offload state
    ],
    ids=[
        'z3_no_opt_offload',
        'z3_cpu_opt_offload',
        'z3_nvme_opt_offload',
        'z2_no_opt_offload',
        'z0_cpu_opt_offload',
    ],
)
def test_can_offload_state(
    mocker, mixin_instance, mock_engine, stage, opt_offload_device, expected
):
    """Tests if the engine state can be offloaded based on ZeRO stage and optimizer offload."""
    mock_engine.zero_optimization_stage.return_value = stage
    if opt_offload_device is not None:
        offload_config = MagicMock()
        offload_config.device = opt_offload_device
        mock_engine.zero_offload_optimizer.return_value = offload_config
    else:
        mock_engine.zero_offload_optimizer.return_value = None

    assert mixin_instance.can_offload_state(mock_engine) is expected


@pytest.mark.parametrize(
    'bf16, fp16, expected_dtype',
    [
        (True, True, torch.bfloat16),  # bf16 takes precedence
        (True, False, torch.bfloat16),
        (False, True, torch.float16),
        (False, False, torch.float32),
    ],
    ids=['bf16_fp16', 'bf16_only', 'fp16_only', 'fp32_only'],
)
def test_get_torch_dtype(
    mixin_instance, mock_engine, bf16, fp16, expected_dtype
):
    """Tests determining the torch dtype from engine config."""
    mock_engine.bfloat16_enabled.return_value = bf16
    mock_engine.fp16_enabled.return_value = fp16
    assert mixin_instance.get_torch_dtype(mock_engine) == expected_dtype


@pytest.mark.parametrize(
    'stage, rank, should_save',
    [
        (3, 0, True),  # Z3, rank 0 saves
        (3, 1, False),  # Z3, other ranks don't save directly
        (2, 0, True),  # Z2, rank 0 saves
        (2, 1, False),  # Z2, other ranks don't save
        (0, 0, True),  # Z0, rank 0 saves
        (0, 1, False),  # Z0, other ranks don't save
    ],
    ids=[
        'z3_rank0',
        'z3_rank1',
        'z2_rank0',
        'z2_rank1',
        'z0_rank0',
        'z0_rank1',
    ],
)
def test_save_weights_hf_pretrained(
    mixin_instance,
    mock_engine,
    mock_distributed,
    mock_gathered_parameters,
    stage,
    rank,
    should_save,
    mocker,  # Add mocker here
):
    """Tests saving weights based on ZeRO stage and distributed rank."""
    output_dir = '/fake/path'
    mock_engine.zero_optimization_stage.return_value = stage
    mock_distributed.get_rank.return_value = rank

    # Ensure the module mock has the save_pretrained method before calling the function
    # This was moved to the fixture, but double-checking or setting it here is also fine.
    # if not hasattr(mock_engine.module, 'save_pretrained'):
    #      mock_engine.module.save_pretrained = mocker.MagicMock()

    mixin_instance.save_weights_hf_pretrained(mock_engine, output_dir)

    # Check if GatheredParameters context was used only for ZeRO-3
    if stage == 3:
        mock_gathered_parameters.assert_called_once_with(
            mock_engine.parameters(), modifier_rank=0
        )
        mock_gathered_parameters.return_value.__enter__.assert_called_once()
        mock_gathered_parameters.return_value.__exit__.assert_called_once()
    else:
        mock_gathered_parameters.assert_not_called()

    # Check if save_pretrained was called correctly
    if should_save:
        mock_engine.module.save_pretrained.assert_called_once_with(output_dir)
    else:
        mock_engine.module.save_pretrained.assert_not_called()

    # Barrier should always be called
    mock_distributed.barrier.assert_called_once()
