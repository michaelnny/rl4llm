from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import deepspeed
import torch
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel


class DeepSpeedUtilsMixin:
    """Mixin providing DeepSpeed utilities that can work with any specified engine."""

    @contextmanager
    def with_unwrapped_model(
        self, engine: DeepSpeedEngine
    ) -> Generator[PreTrainedModel, None, None]:
        """Returns the unwrapped model from the specified DeepSpeed engine."""
        if self.is_zero3_enabled(engine):
            with deepspeed.zero.GatheredParameters(engine.parameters()):
                yield engine.module
        else:
            yield engine.module

    def is_zero3_enabled(self, engine: DeepSpeedEngine) -> bool:
        """Checks if ZeRO-3 is enabled for the specified engine."""
        return engine.zero_optimization_stage() == 3

    def is_zero2_enabled(self, engine: DeepSpeedEngine) -> bool:
        """Checks if ZeRO-2 is enabled for the specified engine."""
        return engine.zero_optimization_stage() == 2

    def is_params_offload_enabled(self, engine: DeepSpeedEngine) -> bool:
        """Checks if model parameters offload is enabled for the specified engine."""
        offload_param = engine.zero_offload_param()
        return (
            self.is_zero3_enabled(engine)
            and offload_param is not None
            and offload_param.device in ['cpu', 'nvme']
        )

    def is_optimizer_offload_enabled(self, engine: DeepSpeedEngine) -> bool:
        """Checks if optimizer parameters offload is enabled for the specified engine."""
        offload_optimizer = engine.zero_offload_optimizer()
        return (
            self.is_zero3_enabled(engine)
            and offload_optimizer is not None
            and offload_optimizer.device in ['cpu', 'nvme']
        )

    def get_torch_dtype(self, engine: DeepSpeedEngine) -> torch.dtype:
        """Determines appropriate torch dtype from the specified engine config."""
        if engine.bfloat16_enabled():
            return torch.bfloat16
        elif engine.fp16_enabled():
            return torch.float16
        return torch.float32

    def save_weights_hf_pretrained(
        self, engine: DeepSpeedEngine, output_dir: str
    ) -> None:
        """Saves the model weights with HF pretrained format."""

        if self.is_zero3_enabled(engine):
            if torch.distributed.get_rank() == 0:
                print('ZeRO-3: Gathering parameters for saving...')
                # Ensure model is on CPU or has enough GPU memory on rank 0 for consolidated weights
                # model_engine.module.cpu() # Optional: Move to CPU first if GPU memory is tight

            # Gather parameters context. modifier_rank=0 specifies rank 0 gathers.
            # requires_grad=False is important for saving, prevents unnecessary grad tracking.
            with deepspeed.zero.GatheredParameters(
                engine.parameters(), modifier_rank=0
            ):
                if torch.distributed.get_rank() == 0:
                    print('Rank 0: Saving gathered model...')
                    model_to_save = engine.module
                    model_to_save.save_pretrained(output_dir)
                    print(f"Model saved by rank 0 to {output_dir}")

            # Barrier ensures all ranks wait until rank 0 is done saving.
            torch.distributed.barrier()

        # The else block for ZeRO-1/2 remains the same
        else:
            if torch.distributed.get_rank() == 0:
                # print(f"ZeRO stage {args.zero_stage}: Saving model and tokenizer on rank 0...")
                model_to_save = engine.module
                model_to_save.save_pretrained(output_dir)
                print(f"Model saved by rank 0 to {output_dir}")
            torch.distributed.barrier()  # Good practice to include barrier here too
