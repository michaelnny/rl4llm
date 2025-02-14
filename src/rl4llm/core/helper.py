from typing import Optional, Tuple, Union

import torch


def whiten(values: torch.FloatTensor, shift_mean: bool = True, dim: int = -1) -> torch.Tensor:
    # Compute the mean and variance along the specified dimension
    mean = values.mean(dim=dim, keepdim=True)
    var = values.var(dim=dim, unbiased=False, keepdim=True)

    # Perform whitening (normalize)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)

    # If shift_mean is False, add back the mean
    if not shift_mean:
        whitened += mean
    return whitened


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True, dim: int = -1) -> torch.Tensor:
    """Whiten values with masked values.

    Args:
        values: Input tensor of shape [batch_size, sequence_length]
        mask: Boolean mask of same shape as values
        shift_mean: Whether to shift the mean to zero
        dim: Dimension along which to perform whitening (default: -1 for sequence dimension)
    """
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    # Create a copy of values to avoid modifying the input
    output = values.clone()

    # Apply mask and reshape to 2D
    valid_values = values[mask]

    # Whiten the valid values
    valid_values = whiten(valid_values, shift_mean, dim)

    # Put the whitened values back
    output[mask] = valid_values

    return output
