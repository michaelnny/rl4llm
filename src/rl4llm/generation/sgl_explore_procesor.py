from typing import List, Optional

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class SglExploreLogitProcessor(CustomLogitProcessor):
    """Initializes the SglExploreLogitProcessor.

    Applies exploration sampling and/or token replacement based on configuration.
    Replacement forces sampling from target_tokens if a source_token is detected
    within the top logits.

    Args:
        explore_steps: Number of steps for exploration sampling.
        explore_skip_n: Number of initial steps to skip before exploration.
        explore_top_k: Top-k value for exploration sampling.
        explore_decay: Decay rate for explore_top_k during exploration.
        replace_source_tokens: Token IDs to check for in top logits to trigger replacement.
        replace_target_tokens: Token IDs to force sample from when replacement occurs.
        replace_check_top_k: Check if any source token is within this top-k logits.
        replace_max_count: Max replacements per sequence (0 disables replacement).
    """

    def __init__(
        self,
        explore_steps: int = 0,
        explore_skip_n: int = 0,
        explore_top_k: int = 20,
        explore_decay: float = 0.9,
        replace_source_tokens: Optional[List[int]] = None,
        replace_target_tokens: Optional[List[int]] = None,
        replace_check_top_k: int = 5,
        replace_max_count: int = 3,
    ):
        super().__init__()

        if not isinstance(explore_steps, int) or explore_steps < 0:
            raise ValueError('explore_steps must be a non-negative integer.')
        if not isinstance(explore_skip_n, int) or explore_skip_n < 0:
            raise ValueError('explore_skip_n must be a non-negative integer.')
        if not isinstance(explore_top_k, int) or explore_top_k <= 0:
            raise ValueError('explore_top_k must be a positive integer.')
        if not isinstance(explore_decay, float) or not (
            0.0 < explore_decay <= 1.0
        ):
            raise ValueError(
                'explore_decay must be a float between 0.0 (exclusive) and 1.0 (inclusive).'
            )
        if replace_source_tokens is not None:
            if not isinstance(replace_source_tokens, list) or not all(
                isinstance(t, int) for t in replace_source_tokens
            ):
                raise TypeError(
                    'replace_source_tokens must be a list of integers or None.'
                )
            if not replace_source_tokens:
                raise ValueError(
                    'replace_source_tokens cannot be an empty list if provided.'
                )
        if replace_target_tokens is not None:
            if not isinstance(replace_target_tokens, list) or not all(
                isinstance(t, int) for t in replace_target_tokens
            ):
                raise TypeError(
                    'replace_target_tokens must be a list of integers or None.'
                )
            if not replace_target_tokens:
                raise ValueError(
                    'replace_target_tokens cannot be an empty list if provided.'
                )
        if not isinstance(replace_check_top_k, int) or replace_check_top_k <= 0:
            raise ValueError('replace_check_top_k must be a positive integer.')
        if not isinstance(replace_max_count, int) or replace_max_count < 0:
            raise ValueError(
                'replace_max_count must be a non-negative integer.'
            )
        if replace_source_tokens and not replace_target_tokens:
            raise ValueError(
                'replace_target_tokens must be provided if replace_source_tokens is set.'
            )
        if replace_target_tokens and not replace_source_tokens:
            raise ValueError(
                'replace_source_tokens must be provided if replace_target_tokens is set.'
            )

        self.explore_steps: int = explore_steps
        self.explore_skip_n: int = explore_skip_n
        self.explore_top_k: int = explore_top_k
        self.explore_decay: float = explore_decay
        self.replace_check_top_k: int = replace_check_top_k
        self.replace_max_count: int = (
            replace_max_count
            if replace_source_tokens
            and replace_target_tokens
            and replace_max_count > 0
            else 0
        )
        self.source_tokens_tensor: Optional[torch.Tensor] = (
            torch.tensor(list(set(replace_source_tokens)), dtype=torch.long)
            if replace_source_tokens
            else None
        )
        self.target_tokens_tensor: Optional[torch.Tensor] = (
            torch.tensor(list(set(replace_target_tokens)), dtype=torch.long)
            if replace_target_tokens
            else None
        )

        if (
            self.source_tokens_tensor is not None
            and self.target_tokens_tensor is not None
        ):
            overlap = torch.isin(
                self.source_tokens_tensor, self.target_tokens_tensor
            )
            if torch.any(overlap):
                print(
                    f"Warning: Some tokens are present in both replace_source_tokens and replace_target_tokens: {self.source_tokens_tensor[overlap].tolist()}"
                )

    def __call__(self, logits, custom_param_list):
        """Processes logits for exploration sampling and token replacement.

        Args:
            logits: Input logits tensor of shape (batch_size, vocab_size).
            custom_param_list: List of dictionaries containing per-sequence parameters
                (e.g., step, replace_prob, replace_count, temperature).

        Returns:
            Modified logits tensor after applying exploration and/or replacement logic.
        """
        import random

        import torch

        assert logits.shape[0] == len(custom_param_list)

        bsz, vocab = logits.shape
        device = logits.device

        source_tokens_dev = None
        if self.source_tokens_tensor is not None:
            if self.source_tokens_tensor.device != device:
                self.source_tokens_tensor = self.source_tokens_tensor.to(device)
            source_tokens_dev = self.source_tokens_tensor

        target_tokens_dev = None
        if self.target_tokens_tensor is not None:
            if self.target_tokens_tensor.device != device:
                self.target_tokens_tensor = self.target_tokens_tensor.to(device)
            target_tokens_dev = self.target_tokens_tensor

        temps = []
        for row in range(bsz):
            cfg = custom_param_list[row]
            step = int(cfg.get('step', 0))
            replace_prob = float(cfg.get('replace_prob', 0.0))
            replace_count = int(cfg.get('replace_count', 0))

            if (
                self.explore_steps > 0
                and self.explore_skip_n
                <= step
                < self.explore_skip_n + self.explore_steps
            ):
                k = max(
                    2,
                    int(
                        self.explore_top_k
                        * self.explore_decay ** (step - self.explore_skip_n)
                    ),
                )
                k = min(k, vocab)
                top_k_indices = torch.topk(logits[row], k=k).indices
                mask = torch.full_like(logits[row], -1e6)
                mask[top_k_indices] = 100.0
                logits[row] = mask

            if (
                source_tokens_dev is not None
                and target_tokens_dev is not None
                and self.replace_max_count > 0
                and replace_count < self.replace_max_count
                and replace_prob > 0
                and step > self.explore_skip_n + self.explore_steps
                and random.random() < replace_prob
            ):
                check_k = min(self.replace_check_top_k, vocab)
                _, top_k_indices = torch.topk(logits[row], k=check_k)
                is_source_in_top_k = torch.isin(
                    top_k_indices, source_tokens_dev
                )

                if torch.any(is_source_in_top_k):
                    logits[row].fill_(-1e6)
                    logits[row, target_tokens_dev] = 100.0
                    cfg['replace_count'] = replace_count + 1

            cfg['step'] = step + 1
            temps.append(float(cfg.get('temperature', 1.0)))

        temps_tensor = torch.tensor(
            temps, dtype=logits.dtype, device=device
        ).unsqueeze(1)
        temps_tensor = torch.clamp(temps_tensor, min=1e-6)
        finite_mask = torch.isfinite(logits)
        logits[finite_mask] = (
            logits[finite_mask] / temps_tensor.expand_as(logits)[finite_mask]
        )

        return logits
