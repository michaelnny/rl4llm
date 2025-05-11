from typing import List, Optional

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class SglExploreLogitProcessor(CustomLogitProcessor):
    """Initializes the SglExploreLogitProcessor.

    Applies exploration sampling.

    Args:
        random_start_steps: Number of steps for exploration sampling.
        random_start_skip_n: Number of initial steps to skip before exploration.
        random_start_top_k: Top-k value for exploration sampling.
    """

    def __init__(
        self,
        random_start_steps: int = 0,
        random_start_skip_n: int = 0,
        random_start_top_k: int = 20,
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

        self.random_start_steps: int = random_start_steps
        self.random_start_skip_n: int = random_start_skip_n
        self.random_start_top_k: int = random_start_top_k

    def __call__(self, logits, custom_param_list):
        """Processes logits for exploration sampling and token replacement.

        Args:
            logits: Input logits tensor of shape (batch_size, vocab_size).
            custom_param_list: List of dictionaries containing per-sequence parameters
                (e.g., step).

        Returns:
            Modified logits tensor after applying exploration and/or replacement logic.
        """

        import torch

        assert logits.shape[0] == len(custom_param_list)

        bsz, vocab = logits.shape

        for row in range(bsz):
            cfg = custom_param_list[row]
            step = int(cfg.get('step', 0))

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

            cfg['step'] = step + 1

        return logits
