import logging
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from rl4llm.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class EpisodeData(BaseModel):
    """LLM ENV rollout episode"""

    prompt_tokens: List[int] = Field(..., description='Prompt token ids')
    prompt_text: str = Field(..., description='Prompt full text')
    prompt_length: int = Field(..., description='Prompt token size')
    completion_tokens: List[int] = Field(
        ..., description='Completion token ids'
    )
    completion_text: str = Field(..., description='Completion full text')
    completion_length: int = Field(..., description='Completion token size')
    reward_dict: Dict[str, float] = Field(
        ..., description='Rewards for the episode'
    )
    raw_data: Optional[Dict] = Field(None, description='Raw sample data')

    class Config:
        arbitrary_types_allowed = True


class EnvState(BaseModel):
    """Environment state for LLM generation"""

    prompt: List[str] = Field(..., description='Prompt full text')
    input_ids: torch.Tensor = Field(..., description='Prompt token ids')
    attention_mask: torch.Tensor = Field(
        ..., description='Attention mask for the prompt token ids'
    )
    ground_truth: List[str | float | int] = Field(
        ..., description='Ground truth to the problem'
    )
    # question: Optional[str] = Field(None, description="Question text")
    # task_type: Optional[str] = Field(None, description="Task type")
    raw_data: Optional[Dict] = Field(None, description='Raw sample data')

    class Config:
        arbitrary_types_allowed = True


class BaseRewardFunction(BaseModel):
    """Base reward function for an environment."""

    name: str = Field(
        ...,
        min_length=5,
        max_length=50,
        description='Unique name using only letters, numbers, hyphens and underscores',
        pattern=r'^[a-zA-Z0-9_\-]+$',
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
            'Reward functions must implement the __call__ method.'
        )

    class Config:
        arbitrary_types_allowed = True


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
        assert batch_size >= 1, 'Batch size must be greater than 0'
        assert reward_functions and all(
            isinstance(fn, BaseRewardFunction) for fn in reward_functions
        )
        self._seed = seed if seed is not None else 131
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)

        self._reward_functions = reward_functions
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._dataset_iterator = iter(self._loader)

    def reset(self) -> EnvState:
        """ """

        try:
            item = next(self._dataset_iterator)
            return self.__prepare_initial_state(item)
        except StopIteration:
            logger.info('Dataset iterator exhausted. Resetting DataLoader.')
            self._dataset_iterator = iter(self._loader)
            item = next(self._dataset_iterator)
            return self.__prepare_initial_state(item)

    def __prepare_initial_state(self, item: Dict) -> EnvState:
        """ """
        if (
            not isinstance(item, dict)
            or 'prompt' not in item
            or 'ground_truth' not in item
        ):
            raise ValueError(
                f"Invalid data, must have 'prompt' and 'ground_truth', got {item}"
            )

        inputs = self._tokenizer(
            item['prompt'],
            return_tensors='pt',
            truncation=False,
            padding=False,
            max_length=self._tokenizer.model_max_length,
        )

        state = EnvState(
            prompt=item['prompt'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            ground_truth=item['ground_truth'],
            raw_data=deepcopy(item),
        )

        return state

    def rollout(
        self, llm: PreTrainedModel, gen_args: Dict
    ) -> List[EpisodeData]:
        """ """

        group_size = gen_args.get('num_return_sequences', 1)

        device = llm.device
        s_t = self.reset()

        input_ids = s_t.input_ids
        attention_mask = s_t.attention_mask

        # if group_size > 1:
        #     # pad the data first???

        output = llm.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            **gen_args,
        )

        full_sequences = output.sequences

        batch_size = full_sequences.size(0)

        if self._batch_size == 1:
            ground_truths = [s_t.ground_truth[0]] * batch_size
            prompt_texts = [s_t.prompt[0]] * batch_size

        elif batch_size == self._batch_size:
            ground_truths = s_t.ground_truth
            prompt_texts = s_t.prompt
        else:
            # we are not sure???
            raise ValueError('Unsupported scenario')

        prompt_length = s_t.input_ids.size(1)

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
                prompt_text=prompt_texts[i],
                prompt_tokens=(
                    s_t.input_ids[i]
                    if self._batch_size == batch_size
                    else s_t.input_ids[0]
                ),
                prompt_length=prompt_length,
                completion_text=completion_texts[i],
                completion_tokens=completion_ids[i],
                completion_length=completion_lengths[i],
                reward_dict={k: v[i] for k, v in reward_dict.items()},
                raw_data=s_t.raw_data,
            )

            results.append(sample)

        return results
