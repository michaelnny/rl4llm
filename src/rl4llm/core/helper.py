from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    if dim is not None:
        return (values * mask).sum(dim=dim, keepdim=True)
    else:
        return (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    if dim is not None:
        return (values * mask).sum(dim=dim, keepdim=True) / (mask.sum(dim=dim, keepdim=True) + 1e-8)
    else:
        return (values * mask).sum() / (mask.sum() + 1e-8)


def masked_normalize(values: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Normalize values with masked values."""
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    values = values * mask
    mean = masked_mean(values, mask, dim=dim)
    mean_centered = values - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def compute_logprobs_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes log probabilities of the taken actions given the logits.

    Args:
        logits (torch.Tensor): Logits from the model, with shape (batch_size, ..., vocab_size).
        targets (torch.Tensor): Actions taken, with shape (batch_size, ...).

    Returns:
        torch.Tensor: Log probabilities of the actions, with shape (batch_size, ...).
    """
    assert logits.dim() == 3, 'Logits should have at least three dimensions (batch_size, seq_len, vocab_size)'
    assert targets.dim() == 2, 'Targets should have at least two dimension (batch_size, seq_len)'
    assert logits.shape[:2] == targets.shape, f"Shape mismatch: logits shape {logits.shape} and labels shape {targets.shape}"

    logprobs = torch.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, dim=2, index=targets.unsqueeze(2)).squeeze(2)


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the entropy of the distribution from the logits.

    Args:
        logits (torch.Tensor): Logits from the model with shape (batch_size, ..., vocab_size).

    Returns:
        torch.Tensor: Entropy of the distribution for each token, with shape (batch_size, ...).
    """
    assert logits.dim() == 3, 'Logits should have at least three dimensions (batch_size, seq_len, vocab_size)'

    pd = torch.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
