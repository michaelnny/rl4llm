import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class SglExploreLogitProcessor(CustomLogitProcessor):
    """A simple logits processor to implement group temperature and exploring start for sampling."""

    def __init__(
        self,
        temperatures: torch.Tensor,
        explore_steps: int = 0,
        explore_skip_n: int = 0,
        explore_top_k: int = 20,
        explore_decay_rate: float = 0.9,
    ):
        """
        Initializes the HfExploreLogitsProcessor.

        Args:
            temperatures: Temperature for logits scaling (tensor).
            group_size: The number of sequences in the original batch request.
            device: The torch device where tensors should be placed.
            explore_steps: Number of steps for exploration sampling.
            explore_skip_n: Number of initial steps to skip before exploration.
            explore_top_k: Top-k value for exploration sampling.
            explore_decay_rate: Decay rate for explore_top_k during exploration.
        """
        if not isinstance(temperatures, torch.Tensor):
            raise ValueError('temperature must be a tensor')
        if any(t < 0 for t in temperatures):
            raise ValueError('temperature values cannot be negative')
        if not isinstance(explore_steps, int) or explore_steps < 0:
            raise ValueError('explore_steps must be a non-negative integer.')
        if not isinstance(explore_skip_n, int) or explore_skip_n < 0:
            raise ValueError('explore_skip_n must be a non-negative integer.')
        if not isinstance(explore_top_k, int) or explore_top_k <= 0:
            raise ValueError('explore_top_k must be a positive integer.')
        if not isinstance(explore_decay_rate, float) or not (
            0.0 < explore_decay_rate <= 1.0
        ):
            raise ValueError(
                'explore_decay_rate must be a float between 0.0 (exclusive) and 1.0 (inclusive).'
            )

        self.temperatures = temperatures
        self.explore_steps: int = explore_steps
        self.explore_skip_n: int = explore_skip_n
        self.explore_top_k: int = explore_top_k
        self.explore_decay_rate: float = explore_decay_rate

        self.step_t: int = 0
        self._initialized = False

    def __call__(self, logits: torch.Tensor, custom_param_list) -> torch.Tensor:
        """Apply exploring random start and group temperature"""
        import torch

        assert logits.dim() == 2
        bsz, vocab_size = logits.shape

        device = logits.device

        # --- Exploration Logic ---
        is_explore = (
            self.explore_steps > 0
            and self.explore_skip_n
            <= self.step_t
            < self.explore_skip_n + self.explore_steps
        )
        if is_explore:
            effective_steps = self.step_t - self.explore_skip_n
            current_explore_top_k = max(
                2,
                int(
                    self.explore_top_k
                    * (self.explore_decay_rate**effective_steps)
                ),
            )
            k = min(current_explore_top_k, vocab_size)

            # Apply uniform sampling within the top-k for exploration
            _, top_k_indices = torch.topk(logits, k=k, dim=-1)
            logits.fill_(float('-inf'))
            logits.scatter_(dim=-1, index=top_k_indices, value=100.0)
        else:
            # --- Group Temperature Logic ---
            if not self._initialized:
                self.temperatures = self.temperatures.to(device)

            # Important, SGLang might dynamic batch the requests during the first few steps
            # so the input logits not necessary have the full batch
            curr_temps = self.temperatures[:bsz]  # Shape: (bsz,)

            # Apply temperature scaling where temp > 0
            non_zero_temp_mask = curr_temps > 0

            safe_temp = torch.clamp(curr_temps, min=1e-8)
            # Reshape temperature and mask for broadcasting with logits
            # (bsz,) -> (bsz, 1)
            safe_temp_reshaped = safe_temp.unsqueeze(1)
            non_zero_temp_mask_reshaped = non_zero_temp_mask.unsqueeze(1)
            scaled_logits = logits / safe_temp_reshaped

            logits = torch.where(
                non_zero_temp_mask_reshaped, scaled_logits, logits
            )

        self.step_t += 1
        return logits
