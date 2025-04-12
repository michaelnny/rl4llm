import gc
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class TrainingMixin:
    """
    A mixin class providing common utility functions for training loops,
    like gradient norm computation, masked operations, and whitening.
    """

    @staticmethod
    def clean_up():
        """Clean GPU memory"""
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def compute_grad_norm(model: torch.nn.Module) -> torch.Tensor:
        """
        Computes the L2 norm of gradients for the model attached to this trainer.

        Requires the inheriting class to have a `self.model` attribute
        which is a `torch.nn.Module`.

        Returns:
            torch.Tensor: A scalar tensor representing the total gradient norm.
        """
        total_norm = torch.tensor(0.0)
        for p in model.parameters():
            if p.grad is not None:
                grad_detached = p.grad.detach()
                local_norm = torch.linalg.vector_norm(
                    grad_detached, dtype=p.dtype
                )
                if total_norm.device != local_norm.device:
                    total_norm = total_norm.to(local_norm.device)
                total_norm += local_norm**2
        return total_norm.sqrt()

    @staticmethod
    def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: Optional[Union[int, Tuple]] = None,
    ) -> torch.Tensor:
        """
        Computes the sum of tensor elements where the mask is True.

        Args:
            values: The tensor whose elements are to be summed.
            mask: A boolean tensor of the same shape as `values`.
            dim: The dimension or dimensions to reduce. If None, sums all masked elements.

        Returns:
            torch.Tensor: The sum of masked elements.
        """
        assert (
            torch.is_tensor(mask) and mask.dtype == torch.bool
        ), 'Mask must be a boolean tensor'
        assert (
            torch.is_tensor(values) and values.shape == mask.shape
        ), 'Values and mask must have the same shape'

        masked_values = values * mask  # Zero out masked-out elements

        if dim is not None:
            return masked_values.sum(dim=dim)
        else:
            return masked_values.sum()

    @staticmethod
    def masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: Optional[Union[int, Tuple]] = None,
    ) -> torch.Tensor:
        """
        Computes the mean of tensor elements where the mask is True.

        Args:
            values: The tensor whose elements are to be averaged.
            mask: A boolean tensor of the same shape as `values`.
            dim: The dimension or dimensions to reduce. If None, averages all masked elements.

        Returns:
            torch.Tensor: The mean of masked elements.
        """
        assert (
            torch.is_tensor(mask) and mask.dtype == torch.bool
        ), 'Mask must be a boolean tensor'
        assert (
            torch.is_tensor(values) and values.shape == mask.shape
        ), 'Values and mask must have the same shape'

        masked_values = values * mask  # Zero out masked-out elements
        num_valid = mask.sum(
            dim=dim, keepdim=dim is not None
        )  # Keep dim for broadcasting if dim is specified

        # Add epsilon to prevent division by zero
        epsilon = 1e-8
        mean = masked_values.sum(dim=dim, keepdim=dim is not None) / (
            num_valid + epsilon
        )

        # If dim was specified, keepdim=True was used. If not, we get a scalar.
        # If dim was specified but resulted in 0 valid elements along that dim,
        # the result would be 0/epsilon = 0, which is reasonable.
        return mean

    @staticmethod
    def whiten(
        values: torch.Tensor,
        shift_mean: bool = True,
        dim: int = -1,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Whitens the input tensor along a specified dimension.
        Converts values to mean 0 and variance 1.

        Args:
            values: Input tensor (expected to be float).
            shift_mean: If True (default), shift the mean to 0. If False, keep original mean.
            dim: The dimension along which to compute mean and variance.
            epsilon: Small value added to variance for numerical stability.

        Returns:
            torch.Tensor: The whitened tensor.
        """
        if not torch.is_floating_point(values):
            # Promote integer types to float for mean/var calculation
            values = values.float()

        # Compute the mean and variance along the specified dimension
        mean = values.mean(dim=dim, keepdim=True)
        var = values.var(dim=dim, unbiased=False, keepdim=True)

        # Perform whitening (normalize)
        whitened = (values - mean) * torch.rsqrt(var + epsilon)

        # If shift_mean is False, add back the mean
        if not shift_mean:
            whitened += mean
        return whitened

    @staticmethod
    def masked_whiten(
        values: torch.Tensor,
        mask: torch.Tensor,
        shift_mean: bool = True,
        dim: int = -1,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Whitens tensor elements where the mask is True, along a specified dimension.
        Elements where mask is False remain unchanged.

        IMPORTANT NOTE: This implementation calculates the mean and variance *only* from
        the masked (valid) elements along the specified dimension, and then applies
        whitening *only* to those valid elements. This differs from the original provided
        `masked_whiten` which flattened all valid elements globally before whitening.
        This implementation is generally more useful for sequence or batch data.

        Args:
            values: Input tensor (expected to be float).
            mask: A boolean tensor of the same shape as `values`.
            shift_mean: If True (default), shift the mean of valid elements to 0.
            dim: The dimension along which to compute mean/variance *using only masked elements*.
            epsilon: Small value added to variance for numerical stability.


        Returns:
            torch.Tensor: The tensor with masked elements whitened.
        """
        assert (
            torch.is_tensor(mask) and mask.dtype == torch.bool
        ), 'Mask must be a boolean tensor'
        assert (
            torch.is_tensor(values) and values.shape == mask.shape
        ), 'Values and mask must have the same shape'
        if not torch.is_floating_point(values):
            # Promote integer types to float for mean/var calculation
            values = values.float()

        # Calculate mean and variance using only masked values
        num_valid = mask.sum(dim=dim, keepdim=True).float()
        masked_values_for_stats = torch.where(
            mask, values, torch.zeros_like(values)
        )  # Use 0 where mask is False for sum

        mean = masked_values_for_stats.sum(dim=dim, keepdim=True) / (
            num_valid + epsilon
        )

        # Variance: E[X^2] - (E[X])^2 for masked values
        var = (masked_values_for_stats**2).sum(dim=dim, keepdim=True) / (
            num_valid + epsilon
        ) - mean**2
        # Ensure variance is non-negative
        var = torch.clamp(var, min=0.0)

        # Whiten only the valid values
        whitened_values = (values - mean) * torch.rsqrt(var + epsilon)

        # If not shifting mean, add back the calculated mean (of valid elements)
        if not shift_mean:
            whitened_values += mean

        # Combine whitened valid values with original invalid values
        output = torch.where(mask, whitened_values, values)

        return output

    @staticmethod
    def compute_logprobs_from_logits(
        logits: torch.Tensor,
        actions: torch.LongTensor,
        loss_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log probabilities of actions.

        Args:
            logits (torch.Tensor): Raw logits for sequence token ids, shape [batch_size, seq_len, vocab_size]
            actions (torch.LongTensor): Action token ids, shape [batch_size, seq_len]
            loss_masks (Optional[torch.Tensor]): Loss mask corresponding to the sequence, shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Log probabilities of actions, shape [batch_size, seq_len]
        """

        assert (
            logits.dim() == 3
        ), 'Logits tensor must have 3 dimensions: [batch_size, seq_len, vocab_size]'
        assert (
            actions.dim() == 2
        ), 'Actions tensor must have 2 dimensions: [batch_size, seq_len]'
        assert (
            logits.shape[:2] == actions.shape
        ), 'Logits tensor shape must match actions shape for batch_size and seq_len'
        if loss_masks is not None:
            assert (
                loss_masks.dim() == 2
            ), 'Loss masks tensor must have 2 dimensions: [batch_size, seq_len]'
            assert (
                loss_masks.shape == actions.shape
            ), 'Loss masks shape must match logits shape for batch_size and seq_len'

        # Process log_softmax and gather operations one sample at a time to avoid CUDA OOM
        batch_size = logits.shape[0]
        sample_logprobs = []

        for i in range(batch_size):
            # Process single sample
            sample_logits = logits[i, ...].float()
            sample_logprobs_all = torch.log_softmax(sample_logits, dim=-1)
            sample_actions = actions[i, ...].unsqueeze(1)
            sample_logprob = torch.gather(
                sample_logprobs_all, dim=1, index=sample_actions
            ).squeeze(1)
            sample_logprobs.append(sample_logprob)

        # Concatenate results
        logrobs = torch.stack(sample_logprobs, dim=0)
        if loss_masks is not None:
            logrobs = logrobs * loss_masks.float()

        return logrobs

    @staticmethod
    def compute_entropy_from_logits(
        logits: torch.Tensor, loss_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute entropy.

        Args:
            logits (torch.Tensor): Raw logits for sequence token ids, shape [batch_size, seq_len, vocab_size]
            loss_masks (Optional[torch.Tensor]): Loss mask corresponding to the sequence, shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Entropy, shape [batch_size, seq_len]
        """

        # Check input dimensions
        assert (
            logits.dim() == 3
        ), 'Logits tensor must have 3 dimensions: [batch_size, seq_len, vocab_size]'
        if loss_masks is not None:
            assert (
                loss_masks.dim() == 2
            ), 'Loss masks tensor must have 2 dimensions: [batch_size, seq_len]'
            assert (
                loss_masks.shape == logits.shape[:2]
            ), 'Loss masks shape must match logits shape for batch_size and seq_len'

        # Compute log probabilities in a numerically stable way
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Convert log probabilities to probabilities
        probs = torch.exp(log_probs)

        # Compute entropy: sum over the class dimension
        entropy = -(probs * log_probs).sum(dim=-1)

        # Apply loss mask if provided
        if loss_masks is not None:
            entropy = entropy * loss_masks.float()

        return entropy

    @staticmethod
    def compute_masked_monte_carlo_returns(
        rewards: torch.Tensor, mask: torch.Tensor, gamma: float
    ) -> torch.FloatTensor:
        """
        Computes monte carlo returns considering only assistant turns.

        Args:
            rewards (torch.Tensor): Float tensor with rewards (0 for user), shape [seq_len]
            mask (torch.Tensor): Binary mask (0 for user, 1 for assistant), shape [seq_len]
            gamma (float): Discount factor

        Returns:
            torch.Tensor: Tensor of the original shape, with discounted returns
                for assistant turns and zeros for user turns
        """
        # Input validation
        assert rewards.dim() == mask.dim() == 1, 'Inputs must be 1-dimensional'
        assert rewards.size(0) == mask.size(
            0
        ), 'Rewards and mask must have same length'
        assert gamma > 0.0 and gamma <= 1.0, 'Discount factor must be in (0, 1]'

        # Initialize returns tensor
        returns = torch.zeros_like(mask, dtype=rewards.dtype)
        seq_len = len(rewards)

        g = 0.0
        # Calculate returns from t=T-1, T-2, ..., 1, 0
        for t in reversed(range(0, seq_len)):
            delta = gamma * g if mask[t] and t < seq_len - 1 else 0.0
            g = rewards[t] + delta
            returns[t] = g

        returns *= mask.float()
        return returns

    @staticmethod
    def compute_masked_gae_advantage(
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> torch.Tensor:
        """Computes masked generalized advantage estimates for a sequence length k considering only assistant turns

        The advantages are computed in a backwards fashion according to the equation:
        Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
        where δₜ = rₜ + γ * V(sₜ₊₁) - V(sₜ).

        Advantages are zeroed out for steps where mask is 0.

        See Proximal Policy Optimization Algorithms, Schulman et al.:
        https://arxiv.org/abs/1707.06347

        Args:
            rewards (torch.Tensor): Float tensor with rewards, shape [seq_len]
            values (torch.Tensor): Float tensor with value estimate, shape [seq_len]
            mask (torch.Tensor): Binary mask (0 for user/prompt, 1 for assistant/generation), shape [seq_len]
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter.

        Returns:
            torch.Tensor: Multi-step truncated generalized advantage estimation, shape [seq_len].
                         Advantages are zero at positions where mask is 0.
        """

        assert (
            rewards.dim() == values.dim() == mask.dim() == 1
        ), 'Inputs must be 1D tensors'
        assert (
            rewards.shape == values.shape == mask.shape
        ), 'Input shapes must match'
        seq_len = rewards.shape[0]

        advantages = torch.zeros_like(rewards, dtype=rewards.dtype)
        gae_cumulative = 0.0

        # Convert mask to float for multiplication
        mask_float = mask.float()

        # Iterate backwards through the sequence
        for t in reversed(range(seq_len)):
            # Determine V(s_{t+1}). If t is the last step, V(s_{t+1}) = 0.
            if t == seq_len - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # Calculate delta: δₜ = rₜ + γ * V(sₜ₊₁) - V(sₜ)
            delta = rewards[t] + gamma * next_value - values[t]

            # Calculate GAE for step t: Aₜ = δₜ + γ * λ * Aₜ₊₁
            gae_cumulative = (
                delta + gamma * gae_lambda * gae_cumulative * mask_float[t]
            )

            advantages[t] = gae_cumulative * mask_float[t]

        return advantages
