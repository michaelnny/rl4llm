"""Implements common features for model training"""

import gc
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

from rl4llm.core.distributed import DistributedOps


class TrainingMixin:
    """
    A mixin class providing common utility functions for training loops,
    like masked operations, whitening, and their distributed counterparts.
    """

    DimType = Optional[Union[int, Tuple[int, ...]]]

    def __init__(self, dist_ops: Optional[DistributedOps] = None):
        """
        Initialize with TrainingMixin instance.
        """
        self.dist_ops = dist_ops or DistributedOps.get_instance()

    @staticmethod
    def clean_up():
        """Clean GPU memory"""
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _validate_mask(values: torch.Tensor, mask: torch.Tensor):
        """Validate mask is boolean and broadcastable to values' shape."""
        assert (
            torch.is_tensor(mask) and mask.dtype == torch.bool
        ), 'Mask must be a boolean tensor'
        assert torch.is_tensor(values), 'Values must be a tensor'
        try:
            # Check if mask shape can be broadcast to values shape
            torch.broadcast_shapes(mask.shape, values.shape)
        except RuntimeError as e:
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to values shape {values.shape}"
            ) from e

    @staticmethod
    def _ensure_float(values: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is float type."""
        if not torch.is_floating_point(values):
            return values.float()
        return values

    @staticmethod
    def _calculate_masked_stats(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType = None,
        keepdim: bool = False,
        epsilon: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes sum, count, mean, and variance for elements where mask is True.
        Handles broadcasting of mask and potential division by zero.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                masked_sum, count, mean, variance
        """
        TrainingMixin._validate_mask(values, mask)
        values = TrainingMixin._ensure_float(values)
        # Broadcast mask to values shape for element-wise operations
        try:
            broadcast_mask = torch.broadcast_to(mask, values.shape)
        except (
            RuntimeError
        ) as e:  # Should be caught by _validate_mask, but double-check
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to values shape {values.shape}"
            ) from e

        count = broadcast_mask.sum(dim=dim, keepdim=keepdim).float()
        safe_count = count + epsilon

        masked_values = torch.where(
            broadcast_mask, values, torch.zeros_like(values)
        )

        masked_sum = masked_values.sum(dim=dim, keepdim=keepdim)
        mean = masked_sum / safe_count

        masked_values_sq_sum = (masked_values**2).sum(dim=dim, keepdim=keepdim)
        # Var(X) = E[X^2] - (E[X])^2
        var = (masked_values_sq_sum / safe_count) - mean**2
        var = torch.clamp(var, min=0.0)  # Ensure non-negative variance

        # Handle cases where count is 0
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        var = torch.where(count > 0, var, torch.zeros_like(var))

        return masked_sum, count, mean, var

    @staticmethod
    def whiten(
        values: torch.Tensor,
        shift_mean: bool = True,
        dim: DimType = None,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Whitens the input tensor (mean 0, variance 1).

        Args:
            values: Input tensor.
            shift_mean: If True, shift mean to 0. Otherwise, keep original mean.
            dim: Dimension(s) along which to compute statistics. If None, use all elements.
            epsilon: Small value for numerical stability.

        Returns:
            The whitened tensor, with the same shape as input.
        """
        values = TrainingMixin._ensure_float(values)
        # Calculate stats keeping dim for broadcasting
        mean = torch.mean(values, dim=dim, keepdim=True)
        var = torch.var(values, dim=dim, unbiased=False, keepdim=True)

        whitened = (values - mean) * torch.rsqrt(var + epsilon)

        if not shift_mean:
            whitened += mean  # Add back the original mean
        return whitened

    # --- Masked Operations (Local) ---

    @staticmethod
    def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """
        Computes the sum of tensor elements where the mask is True.

        Args:
            values: The tensor whose elements are to be summed.
            mask: A boolean tensor broadcastable to `values`.
            dim: Dimension(s) to reduce. If None, sums all masked elements.
            keepdim: Whether the output tensor has `dim` retained or not.

        Returns:
            The sum of masked elements.
        """
        TrainingMixin._validate_mask(values, mask)
        try:
            broadcast_mask = torch.broadcast_to(mask, values.shape)
        except RuntimeError as e:
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to values shape {values.shape}"
            ) from e
        masked_values = torch.where(
            broadcast_mask, values, torch.zeros_like(values)
        )
        return masked_values.sum(dim=dim, keepdim=keepdim)

    @staticmethod
    def masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType = None,
        keepdim: bool = False,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Computes the mean of tensor elements where the mask is True.

        Args:
            values: The tensor whose elements are to be averaged.
            mask: A boolean tensor broadcastable to `values`.
            dim: Dimension(s) to reduce. If None, averages all masked elements.
            keepdim: Whether the output tensor has `dim` retained or not.
            epsilon: Small value for numerical stability.

        Returns:
            The mean of masked elements.
        """
        _, _, mean, _ = TrainingMixin._calculate_masked_stats(
            values, mask, dim, keepdim, epsilon
        )
        return mean

    @staticmethod
    def masked_var(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType = None,
        keepdim: bool = False,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Computes the variance of tensor elements where the mask is True.

        Args:
            values: The tensor whose elements are used for variance.
            mask: A boolean tensor broadcastable to `values`.
            dim: Dimension(s) to reduce. If None, variance of all masked elements.
            keepdim: Whether the output tensor has `dim` retained or not.
            epsilon: Small value for numerical stability.

        Returns:
            The variance of masked elements.
        """
        _, _, _, var = TrainingMixin._calculate_masked_stats(
            values, mask, dim, keepdim, epsilon
        )
        return var

    @staticmethod
    def masked_whiten(
        values: torch.Tensor,
        mask: torch.Tensor,
        shift_mean: bool = True,
        dim: DimType = None,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Whitens tensor elements where mask is True, using masked statistics.
        Elements where mask is False remain unchanged.

        Args:
            values: Input tensor.
            mask: A boolean tensor broadcastable to `values`.
            shift_mean: If True, shift mean of masked elements to 0.
            dim: Dimension(s) along which to compute masked statistics. If None, use all masked elements.
            epsilon: Small value for numerical stability.

        Returns:
            The tensor with masked elements whitened, same shape as input.
        """
        # Calculate masked stats, keeping dim for broadcasting during whitening
        _, _, mean, var = TrainingMixin._calculate_masked_stats(
            values, mask, dim, keepdim=True, epsilon=epsilon
        )

        # Whiten using broadcasted stats
        whitened_values = (values - mean) * torch.rsqrt(var + epsilon)

        if not shift_mean:
            whitened_values += mean  # Add back the masked mean

        # Combine whitened valid values with original invalid values
        try:
            broadcast_mask = torch.broadcast_to(mask, values.shape)
        except RuntimeError as e:
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to values shape {values.shape}"
            ) from e
        output = torch.where(broadcast_mask, whitened_values, values)
        return output

    # --- Distributed Operations Helper ---

    def _dist_aggregate_masked_stats(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType,
        keepdim: bool,
        calculate_sum: bool = True,  # Control calculation/reduction
        calculate_count: bool = True,  # Control calculation/reduction
        calculate_sum_sq: bool = False,  # Control calculation/reduction
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Helper: Calculates local masked stats (sum, count, sum_sq based on flags),
        then aggregates globally ONLY the requested stats.
        Handles varying tensor sizes across ranks.

        Returns:
            Tuple: Optional[global_sum], Optional[global_count], Optional[global_sum_sq]
                   Elements are None if not requested via flags.
        """
        TrainingMixin._validate_mask(values, mask)  # Ensures broadcastable
        float_values = TrainingMixin._ensure_float(values)
        try:
            broadcast_mask = torch.broadcast_to(mask, float_values.shape)
        except RuntimeError as e:
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to values shape {values.shape}"
            ) from e

        # Calculate local statistics ONLY if needed
        local_sum = None
        local_count = None
        local_sum_sq = None

        if calculate_count:
            local_count = broadcast_mask.sum(dim=dim, keepdim=keepdim).float()

        if (
            calculate_sum or calculate_sum_sq
        ):  # Need masked_values if either sum or sum_sq needed
            masked_values = torch.where(
                broadcast_mask, float_values, torch.zeros_like(float_values)
            )
            if calculate_sum:
                local_sum = masked_values.sum(dim=dim, keepdim=keepdim)
            if calculate_sum_sq:
                local_sum_sq = (masked_values**2).sum(dim=dim, keepdim=keepdim)

        # Aggregate across processes ONLY if needed and world_size > 1
        global_sum = local_sum
        global_count = local_count
        global_sum_sq = local_sum_sq

        if self.dist_ops.world_size > 1:
            if calculate_sum and local_sum is not None:
                global_sum = self.dist_ops.all_reduce_tensor(
                    local_sum, op=dist.ReduceOp.SUM
                )
            if calculate_count and local_count is not None:
                global_count = self.dist_ops.all_reduce_tensor(
                    local_count, op=dist.ReduceOp.SUM
                )
            if calculate_sum_sq and local_sum_sq is not None:
                global_sum_sq = self.dist_ops.all_reduce_tensor(
                    local_sum_sq, op=dist.ReduceOp.SUM
                )

        return global_sum, global_count, global_sum_sq

    # --- Distributed Masked Operations ---

    def dist_masked_sum(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """
        Computes masked sum, aggregated across ranks. Handles varying tensor sizes.

        Args:
            values: Local tensor shard.
            mask: Local mask, broadcastable to `values`.
            dim: Dimension(s) for local reduction before aggregation. If None, reduces all dims.
            keepdim: Whether the output tensor has `dim` retained or not.

        Returns:
            The globally summed tensor. Available on all ranks.
        """
        # OPTIMIZED: Calculate local sum directly and reduce only that.
        local_sum = TrainingMixin.masked_sum(
            values, mask, dim=dim, keepdim=keepdim
        )

        if self.dist_ops.world_size == 1:
            return local_sum
        else:
            global_sum = self.dist_ops.all_reduce_tensor(
                local_sum, op=dist.ReduceOp.SUM
            )
            return global_sum

    def dist_masked_mean(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType = None,
        keepdim: bool = False,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Computes masked mean, aggregated across ranks. Handles varying tensor sizes.

        Args:
            values: Local tensor shard.
            mask: Local mask, broadcastable to `values`.
            dim: Dimension(s) for local reduction before aggregation. If None, reduces all dims.
            keepdim: Whether the output tensor has `dim` retained or not.
            epsilon: Small value for numerical stability.

        Returns:
            The globally averaged tensor. Available on all ranks.
        """
        # Need global sum and global count
        global_sum, global_count, _ = self._dist_aggregate_masked_stats(
            values,
            mask,
            dim,
            keepdim,
            calculate_sum=True,
            calculate_count=True,
            calculate_sum_sq=False,
        )
        assert global_sum is not None and global_count is not None

        global_mean = global_sum / (global_count + epsilon)
        global_mean = torch.where(
            global_count > 0, global_mean, torch.zeros_like(global_mean)
        )
        return global_mean

    def dist_masked_var(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: DimType = None,
        keepdim: bool = False,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Computes masked variance, aggregated across ranks. Handles varying tensor sizes.

        Args:
            values: Local tensor shard.
            mask: Local mask, broadcastable to `values`.
            dim: Dimension(s) for local reduction before aggregation. If None, reduces all dims.
            keepdim: Whether the output tensor has `dim` retained or not.
            epsilon: Small value for numerical stability.

        Returns:
            The globally computed variance tensor. Available on all ranks.
        """
        # Need all three global stats
        global_sum, global_count, global_sum_sq = (
            self._dist_aggregate_masked_stats(
                values,
                mask,
                dim,
                keepdim,
                calculate_sum=True,
                calculate_count=True,
                calculate_sum_sq=True,
            )
        )
        assert (
            global_sum is not None
            and global_count is not None
            and global_sum_sq is not None
        )

        safe_global_count = global_count + epsilon
        global_mean = global_sum / safe_global_count
        global_e_x_sq = global_sum_sq / safe_global_count
        global_var = global_e_x_sq - global_mean**2
        global_var = torch.clamp(global_var, min=0.0)
        global_var = torch.where(
            global_count > 0, global_var, torch.zeros_like(global_var)
        )
        return global_var

    def dist_masked_whiten(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        shift_mean: bool = True,
        dim: DimType = None,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Whitens masked elements using global masked statistics. Handles varying tensor sizes.

        Args:
            values: Local tensor shard.
            mask: Local mask, broadcastable to `values`.
            shift_mean: If True, shift mean of masked elements to 0 globally.
            dim: Dimension(s) for global statistics calculation. If None, uses all elements.
            epsilon: Small value for numerical stability.

        Returns:
            Local tensor with masked elements whitened using global stats, same shape as input.
        """
        float_values = TrainingMixin._ensure_float(values)
        TrainingMixin._validate_mask(
            float_values, mask
        )  # Ensures broadcastable

        # Get global stats, keeping dim for local broadcasting
        global_sum, global_count, global_sum_sq = (
            self._dist_aggregate_masked_stats(
                float_values,
                mask,
                dim,
                keepdim=True,  # MUST keepdim for broadcast
                calculate_sum=True,
                calculate_count=True,
                calculate_sum_sq=True,
            )
        )
        assert (
            global_sum is not None
            and global_count is not None
            and global_sum_sq is not None
        )

        # Calculate global mean/var for whitening (broadcastable)
        safe_global_count = global_count + epsilon
        global_mean = global_sum / safe_global_count
        global_e_x_sq = global_sum_sq / safe_global_count
        global_var = global_e_x_sq - global_mean**2
        global_var = torch.clamp(global_var, min=0.0)

        # Handle count=0 cases for broadcastable stats
        global_mean = torch.where(
            global_count > 0, global_mean, torch.zeros_like(global_mean)
        )
        global_var = torch.where(
            global_count > 0, global_var, torch.zeros_like(global_var)
        )

        # Whiten local values using global stats
        whitened_values = (float_values - global_mean) * torch.rsqrt(
            global_var + epsilon
        )

        if not shift_mean:
            whitened_values += global_mean  # Add back global masked mean

        # Combine using local mask
        try:
            broadcast_mask = torch.broadcast_to(mask, float_values.shape)
        except RuntimeError as e:
            raise ValueError(
                f"Mask shape {mask.shape} cannot be broadcast to values shape {values.shape}"
            ) from e
        output = torch.where(broadcast_mask, whitened_values, float_values)
        return output

    # --- Distributed Standard Operations ---

    def dist_whiten(
        self,
        values: torch.Tensor,
        shift_mean: bool = True,
        dim: DimType = None,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Whitens tensor using global statistics. Handles varying tensor sizes.

        Args:
            values: Local tensor shard.
            shift_mean: If True, shift mean to 0 globally.
            dim: Dimension(s) for global statistics calculation. If None, uses all elements.
            epsilon: Small value for numerical stability.

        Returns:
            Local tensor whitened using global stats, same shape as input.
        """
        # Equivalent to dist_masked_whiten with a mask of all True
        mask = torch.ones_like(values, dtype=torch.bool)
        return self.dist_masked_whiten(
            values=values,
            mask=mask,
            shift_mean=shift_mean,
            dim=dim,
            epsilon=epsilon,
        )

    # --- RL Algorithm Common Operations ---

    @staticmethod
    def compute_logprobs_from_logits(
        logits: torch.Tensor,
        actions: torch.LongTensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log probabilities of actions.

        Args:
            logits (torch.Tensor): Raw logits for sequence token ids, shape [batch_size, seq_len, vocab_size]
            actions (torch.LongTensor): Action token ids, shape [batch_size, seq_len]
            loss_mask (Optional[torch.Tensor]): Loss mask corresponding to the sequence, shape [batch_size, seq_len]

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
        if loss_mask is not None:
            assert (
                loss_mask.dim() == 2
            ), 'Loss masks tensor must have 2 dimensions: [batch_size, seq_len]'
            assert (
                loss_mask.shape == actions.shape
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
        logprobs = torch.stack(sample_logprobs, dim=0)
        if loss_mask is not None:
            logprobs = logprobs * loss_mask.float()

        return logprobs

    @staticmethod
    def compute_entropy_from_logits(
        logits: torch.Tensor, loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute entropy.

        Args:
            logits (torch.Tensor): Raw logits for sequence token ids, shape [batch_size, seq_len, vocab_size]
            loss_mask (Optional[torch.Tensor]): Loss mask corresponding to the sequence, shape [batch_size, seq_len]

        Returns:
            torch.Tensor: Entropy, shape [batch_size, seq_len]
        """

        # Check input dimensions
        assert (
            logits.dim() == 3
        ), 'Logits tensor must have 3 dimensions: [batch_size, seq_len, vocab_size]'
        if loss_mask is not None:
            assert (
                loss_mask.dim() == 2
            ), 'Loss masks tensor must have 2 dimensions: [batch_size, seq_len]'
            assert (
                loss_mask.shape == logits.shape[:2]
            ), 'Loss masks shape must match logits shape for batch_size and seq_len'

        # # Compute log probabilities in a numerically stable way
        # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # # Convert log probabilities to probabilities
        # probs = torch.exp(log_probs)

        # # Compute entropy: sum over the class dimension
        # entropy = -(probs * log_probs).sum(dim=-1)

        # Working on one item at a time to avoid CUDA OOM
        batch_size = logits.size(0)
        entropies = []

        for i in range(batch_size):
            # grab a single [seq_len, vocab_size] slice
            sample_logits = logits[i].float()
            # log‑softmax → [L, V]
            logp = torch.nn.functional.log_softmax(sample_logits, dim=-1)
            # p = exp(logp) → [L, V]
            p = logp.exp()
            # entropy = –Σ p * log p along V → [L]
            sample_entropy = -(p * logp).sum(dim=-1)
            entropies.append(sample_entropy)

        # stack back to [B, L]
        entropy = torch.stack(entropies, dim=0)

        # Apply loss mask if provided
        if loss_mask is not None:
            entropy = entropy * loss_mask.float()

        return entropy

    @staticmethod
    def masked_monte_carlo_returns(
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
    def masked_returns_and_gae_advantages(
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and GAE advantages for a single episode sequence. This is often used in PPO.

        Args:
            rewards (torch.Tensor): A tensor containing the reward for both prompt and completion sequence
                                    from t=0,1,...,T-1, shape [seq_len].
            values (torch.Tensor): A tensor containing the state value estimate for both prompt and completion
                                sequence from t=0,1,...,T-1, shape [seq_len].
            mask (torch.Tensor): A tensor where 0s for prompt position and 1s for completion position
                                for sequence from t=0,1,...,T-1, shape [seq_len].
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors containing the returns and advantage estimates.
        """

        assert (
            rewards.shape == values.shape == mask.shape
        ), 'Tensors have mismatched shapes'
        assert rewards.dtype == values.dtype, 'Tensors have mismatched dtypes'
        assert rewards.dim() == 1, 'Tensors must be 1D'
        assert mask.dtype == torch.bool, 'Mask must be a bool tensor'
        assert 0 < gamma <= 1.0, 'Invalid gamma, must be (0, 10]'
        assert 0 < gae_lambda <= 1.0, 'Invalid gae_lambda, must be (0, 10]'

        device = rewards.device
        torch_dtype = rewards.dtype

        # Extract only the completion sequence (where mask == 1)
        r_t = rewards[mask]
        v_t = values[mask]

        # Handle empty completion sequence case
        if r_t.numel() == 0:
            return torch.zeros_like(
                rewards, dtype=torch_dtype
            ), torch.zeros_like(rewards, dtype=torch_dtype)

        # Pad value at terminal step T to have zero
        v_tp1 = torch.zeros_like(v_t, dtype=torch_dtype, device=device)
        v_tp1[:-1] = v_t[1:]

        # Mark terminal step
        done_tp1 = torch.zeros_like(v_tp1, dtype=torch.bool, device=device)
        done_tp1[-1] = True
        discount_tp1 = (~done_tp1).float() * gamma

        adv_t = truncated_generalized_advantage_estimation(
            r_t, v_t, v_tp1, discount_tp1, gae_lambda
        )
        return_t = adv_t + v_t

        # Create the full tensors with zero values for prompt tokens
        returns = torch.zeros_like(rewards, dtype=torch_dtype, device=device)
        advantages = torch.zeros_like(rewards, dtype=torch_dtype, device=device)
        returns[mask] = return_t
        advantages[mask] = adv_t

        return returns, advantages


def truncated_generalized_advantage_estimation(
    reward_t: torch.Tensor,
    value_t: torch.Tensor,
    value_tp1: torch.Tensor,
    discount_tp1: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ + γₜ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Args:
      reward_t: Sequence of rewards at times [0, k]
      value_t: Sequence of values under π at times [0, k]
      value_tp1: Sequence of values under π at times [1, k+1]
      discount_tp1: Sequence of discounts at times [1, k+1]
      lambda_: a scalar

    Returns:
      Multistep truncated generalized advantage estimation at times [0, k].
    """

    assert len(reward_t.shape) == 1
    assert len(value_t.shape) == 1
    assert len(value_tp1.shape) == 1
    assert len(discount_tp1.shape) == 1
    _dtype = reward_t.dtype
    # Ensure lambda_ is a tensor of the same shape as discount_tp1
    lambda_ = torch.ones_like(discount_tp1, dtype=_dtype) * lambda_
    delta_t = reward_t + discount_tp1 * value_tp1 - value_t
    advantage_t = torch.zeros_like(delta_t, dtype=_dtype)

    # Compute advantages in reverse order
    gae_t = 0
    for i in reversed(range(len(delta_t))):
        gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
        advantage_t[i] = gae_t

    return advantage_t
