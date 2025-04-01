import random
import numpy as np
import logging
from copy import deepcopy
import torch
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from pydantic import BaseModel, Field, constr, field_validator, model_validator

from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class EpisodeData(BaseModel):
    """LLM ENV rollout episode"""

    prompt_tokens: List[int] = Field(..., description="Prompt token ids")
    prompt_text: str = Field(..., description="Prompt full text")
    prompt_length: int = Field(..., description="Prompt token size")
    completion_tokens: List[int] = Field(..., description="Completion token ids")
    completion_text: str = Field(..., description="Completion full text")
    completion_length: int = Field(..., description="Completion token size")
    reward_dict: Dict[str, float] = Field(..., description="Rewards for the episode")
    raw_data: Optional[Dict] = Field(None, description="Raw sample data")


class EnvState(BaseModel):
    """Environment state for LLM generation"""

    prompt: str = Field(..., description="Prompt full text")
    input_ids: torch.Tensor = Field(..., description="Prompt token ids")
    attention_mask: torch.Tensor = Field(
        ..., description="Attention mask for the prompt token ids"
    )
    ground_truth: str | float | int = Field(
        ..., description="Ground truth to the problem"
    )
    # question: Optional[str] = Field(None, description="Question text")
    # task_type: Optional[str] = Field(None, description="Task type")
    raw_data: Optional[Dict] = Field(None, description="Raw sample data")


class BaseRewardFunction(BaseModel):
    """Base reward function for an environment."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique name using only letters, numbers, hyphens and underscores",
        examples=["bleu_score", "toxicity-check", "reward_model_v2"],
        regex=r"^[a-zA-Z0-9_\-]+$",
    )

    def __call__(
        self,
        completions: List[str],
        ground_truths: List[Union[str | float | int]],
        **kwargs: Dict[str, Any],
    ) -> List[float]:
        """Implements the reward function.

        Args:
            completions (List[str]): LLM generated completion texts.
            ground_truths (List[Union[str | float | int]]): Ground truth for the problem.
            **kwargs (Dict[str, Any]): Any additional data.

        Returns:
            List[float]: A list of scalar rewards.
        """
        raise NotImplementedError(
            "Reward functions must implement the __call__ method."
        )


# class EnvAction(BaseModel):
#     """Environment action from LLM generation"""

#     completion_ids: torch.Tensor = Field(..., description="Completion token ids")
#     completion_text: str = Field(..., description="Completion full text")


class LLMEnv:
    """
    Environment for generating LLM training samples.

    Manages interaction flow, prompt generation, response processing,
    and reward calculation based on configuration.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        reward_functions: List[BaseRewardFunction],
        seed: int = 42,
    ):
        assert batch_size >= 1, "Batch size must be greater than 0"
        assert reward_functions and all(
            isinstance(fn, BaseRewardFunction) for fn in reward_functions
        )
        self._seed = seed if seed is not None else 131
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)

        self._tokenizer = tokenizer

        self._loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._reward_functions = reward_functions

    def reset(self) -> EnvState:
        """ """
        if self._dataset_iterator is None:
            self._dataset_iterator = iter(self._loader)
        try:
            item = next(self._dataset_iterator)
            return self.__prepare_initial_state(item)
        except StopIteration:
            logger.info("Dataset iterator exhausted. Resetting DataLoader.")
            self._dataset_iterator = iter(self._loader)
            item = next(self._dataset_iterator)
            return self.__prepare_initial_state(item)

    def __prepare_initial_state(self, item: Dict) -> EnvState:
        """ """
        if (
            not isinstance(item, dict)
            or "prompt" not in item
            or "ground_truth" not in item
        ):
            raise ValueError(
                f"Invalid data, must have 'prompt' and 'ground_truth', got {item}"
            )

        inputs = self._tokenizer(
            item["prompt"],
            return_tensors="pt",
            truncation=False,
            padding=False,
            max_length=self._tokenizer.model_max_length,
        )

        state = EnvState(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            ground_truth=item["ground_truth"],
            raw_data=deepcopy(item),
        )

        return state

    def rollout(self, llm: PreTrainedModel, gen_args: Dict) -> List[EpisodeData]:
        """ """

        device = llm.device
        s_t = self.reset()

        output = llm.generate(
            input_ids=s_t.input_ids.to(device), attention_mask=s_t.attention_mask.to(device), **gen_args
        )

        full_sequences = output.sequences

        batch_size = full_sequences.size(0)
        prompt_length = s_t.input_ids.size(1)
        ground_truths = [s_t.ground_truth] * batch_size

        completion_ids = full_sequences[:, prompt_length:].cpu()
        completion_lengths = (
            (completion_ids != self._tokenizer.pad_token_id).sum(dim=1).cpu()
        )
        completion_texts = self._tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        reward_dict = {}

        for reward_fn in self._reward_functions:
            reward_dict[reward_fn.name] = reward_fn(
                completion_texts, ground_truths, **s_t.raw_data
            )

        results = []

        for i in range(batch_size):
            sample = EpisodeData(
                prompt_text=s_t.prompt,
                prompt_tokens=s_t.input_ids,
                prompt_length=prompt_length,
                completion_text=completion_texts[i],
                completion_tokens=completion_ids[i],
                completion_length=completion_lengths[i],
                reward_dict=reward_dict,
                raw_data=s_t.raw_data,
            )

            results.append(sample)

        return results
