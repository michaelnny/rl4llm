import random
from typing import Any, Callable, Dict, List, Optional, Union

import sglang as sgl
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
        explore_percentage: Decay rate for random_start_top_k during exploration.
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
        explore_percentage: float = 0.9,
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
        if not isinstance(explore_percentage, float) or not (
            0.0 < explore_percentage <= 1.0
        ):
            raise ValueError(
                'explore_percentage must be a float between 0.0 (exclusive) and 1.0 (inclusive).'
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
        self.explore_percentage: float = explore_percentage
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

        temps = []
        for row in range(bsz):
            cfg = custom_param_list[row]
            step = int(cfg.get('step', 0))
            replace_prob = float(cfg.get('replace_prob', 0.0))
            replace_count = int(cfg.get('replace_count', 0))

            if (
                self.random_start_steps > 0
                and self.random_start_skip_n
                <= step
                < self.random_start_skip_n + self.random_start_steps
            ):
                k = max(
                    2,
                    int(
                        self.random_start_top_k
                        * self.explore_percentage
                        ** (step - self.random_start_skip_n)
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


# Example Usage (similar to original main, but adding replacement params)
def main():
    # Make sure to have the tokenizer for the model to get token IDs
    from transformers import AutoTokenizer

    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        source_tokens = [tokenizer.eos_token_id]
        target_tokens = [tokenizer.encode(kwd)[0] for kwd in [' Wait', ' Hmm']]
        print(f"Using Source Tokens (IDs): {source_tokens}")
        print(f"Using Target Tokens (IDs): {target_tokens}")
    except KeyError as e:
        print(f"Error: Token not found in tokenizer vocabulary: {e}")
        print('Please verify the tokens used for source/target exist.')
        return
    except Exception as e:
        print(f"An error occurred getting token IDs: {e}")
        return

    # --- SGLang Engine Setup ---
    llm = sgl.Engine(
        model_path=model_name,
        enable_custom_logit_processor=True,
        # enable_memory_saver=True, # Optional
        # tp_size=1, # Adjust tensor parallelism if needed
    )

    prompts = [
        '<|im_start|>user\nDescribe a perfect day<|im_end|>\n<|im_start|>assistant',
        '<|im_start|>user\nWhat is the meaning of life is<|im_end|>\n<|im_start|>assistant',
        '<|im_start|>user\nRecipe for chocolate chip cookies<|im_end|>\n<|im_start|>assistant',
        '<|im_start|>user\nWrite a short story about a lost robot<|im_end|>\n<|im_start|>assistant',
    ]

    # --- Sampling Parameters ---
    sampling_params = []
    base_temp = 0.7
    replace_probability = 0.99

    for i, p in enumerate(prompts):
        sp = {
            'temperature': 1.0,  # Set to 1.0 here, actual temp applied in processor
            'top_p': 0.95,
            'top_k': -1,  # Use -1 if top_p is used, or set a value like 50
            'max_new_tokens': 250,
            'custom_params': {
                'temperature': base_temp,
                'step': 0,
                'replace_prob': (replace_probability if i % 2 == 0 else 0.0),
                'replace_count': 0,
            },
        }
        sampling_params.append(sp)

    # --- Logit Processor Setup ---
    logit_processor = SglExploreLogitProcessor(
        random_start_steps=5,
        random_start_top_k=100,
        random_start_skip_n=0,
        explore_percentage=0.9,
        replace_source_tokens=source_tokens,
        replace_target_tokens=target_tokens,
        replace_top_k=10,
        replace_max_count=3,
    )

    # --- Generation ---
    print('Generating text...')
    outputs = llm.generate(
        prompts,
        sampling_params,
        custom_logit_processor=logit_processor.to_str(),
    )

    # --- Print Results ---
    print('\n--- Generation Results ---')
    for i, (prompt, output, params) in enumerate(
        zip(prompts, outputs, sampling_params)
    ):
        print('===============================')
        applied_replacement = params['custom_params']['replace_prob'] > 0
        print(
            f"Prompt {i + 1} (Replacement Active: {applied_replacement}): {prompt}"
        )
        print(f"Generated text: {output['text']}")
        # You might need to inspect internal state or logs to see replacement counts if needed
        # print(f"Final custom_params state: {output.get('custom_params_state', 'N/A')}") # If sglang returns final state

    # --- Cleanup ---
    print('\nReleasing resources...')
    # llm.shutdown() # Use shutdown or release_memory_occupation depending on sglang version/needs
    llm.release_memory_occupation()
    torch.cuda.empty_cache()
    print(
        f"CUDA Memory Allocated after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB"
    )


if __name__ == '__main__':
    # Make sure to install transformers: pip install transformers
    main()
