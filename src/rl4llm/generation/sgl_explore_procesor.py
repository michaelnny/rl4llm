from typing import List, Optional

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class SglExploreLogitProcessor(CustomLogitProcessor):
    """Initializes the SglExploreLogitProcessor.

    Applies exploration sampling and/or token replacement based on configuration.
    Replacement forces sampling from target_tokens if a source_token is detected
    within the top logits.

    Args:
        random_start_steps: Number of steps for exploration sampling.
        random_start_skip_n: Number of initial steps to skip before exploration.
        random_start_top_k: Top-k value for exploration sampling.

        replace_source_tokens: Token IDs to check for in top logits to trigger replacement.
        replace_target_tokens: Token IDs to force sample from when replacement occurs.
        replace_top_k: Check if any source token is within this top-k logits.
        replace_max_count: Max replacements per sequence (0 disables replacement).
    """

    def __init__(
        self,
        random_start_steps: int = 0,
        random_start_skip_n: int = 0,
        random_start_top_k: int = 20,
        replace_source_tokens: Optional[List[int]] = None,
        replace_target_tokens: Optional[List[int]] = None,
        replace_top_k: int = 5,
        replace_max_count: int = 3,
    ):
        super().__init__()

        if not isinstance(random_start_steps, int) or random_start_steps < 0:
            raise ValueError(
                'random_start_steps must be a non-negative integer.'
            )
        if not isinstance(random_start_skip_n, int) or random_start_skip_n < 0:
            raise ValueError(
                'random_start_skip_n must be a non-negative integer.'
            )
        if not isinstance(random_start_top_k, int) or random_start_top_k <= 0:
            raise ValueError('random_start_top_k must be a positive integer.')

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
        if not isinstance(replace_top_k, int) or replace_top_k <= 0:
            raise ValueError('replace_top_k must be a positive integer.')
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

        self.random_start_steps: int = random_start_steps
        self.random_start_skip_n: int = random_start_skip_n
        self.random_start_top_k: int = random_start_top_k

        self.replace_top_k: int = replace_top_k
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

        for row in range(bsz):
            cfg = custom_param_list[row]
            step = int(cfg.get('step', 0))
            replace_prob = float(cfg.get('replace_prob', 0.0))
            replace_count = int(cfg.get('replace_count', 0))

            # Exploring start
            if (
                self.random_start_steps > 0
                and self.random_start_skip_n
                <= step
                < self.random_start_skip_n + self.random_start_steps
            ):
                k = min(self.random_start_top_k, vocab)
                top_k_indices = torch.topk(logits[row], k=k).indices
                mask = torch.full_like(logits[row], -1e6)
                mask[top_k_indices] = 100.0
                logits[row] = mask

            # Special token 'replacement'
            if (
                source_tokens_dev is not None
                and target_tokens_dev is not None
                and self.replace_max_count > 0
                and replace_count < self.replace_max_count
                and replace_prob > 0
                and step > self.random_start_skip_n + self.random_start_steps
                and random.random() < replace_prob
            ):
                check_k = min(self.replace_top_k, vocab)
                _, top_k_indices = torch.topk(logits[row], k=check_k)
                is_source_in_top_k = torch.isin(
                    top_k_indices, source_tokens_dev
                )

                if torch.any(is_source_in_top_k):
                    logits[row].fill_(-1e6)
                    logits[row, target_tokens_dev] = 100.0
                    cfg['replace_count'] = replace_count + 1

            cfg['step'] = step + 1

        return logits
