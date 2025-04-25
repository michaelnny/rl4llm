"""Implements MDP ENV for collect samples using SGLang inference server with a custom HTTP client"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rl4llm.core.base_env import BaseEnv, EnvState, EpisodeData, EpisodeMetadata
from rl4llm.core.base_inference_client import InferenceClient

logger = logging.getLogger(__name__)


class SglMDPEnv(BaseEnv):
    """
    Environment for generating samples using SGLang inference server with a custom HTTP client.
    """

    def _process_single_output_item(
        self, item: Dict[str, Any]
    ) -> Union[str, torch.Tensor]:
        """Convert text to token IDs, ensuring EOS token if appropriate."""
        text = item['text']

        if not text:
            # Use some default text to ensure code will not break
            text = "I can't help with this question."

        meta_info = item.get('meta_info')
        token_ids = list(
            self.tokenizer(
                text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )['input_ids']
        )

        # Add EOS token if needed
        if meta_info and 'finish_reason' in meta_info:
            finish_reason = meta_info['finish_reason']
            if (
                'type' in finish_reason
                and finish_reason['type'] != 'length'
                and token_ids[-1] != self.tokenizer.eos_token_id
            ):
                token_ids.append(self.tokenizer.eos_token_id)

        return text, torch.tensor(token_ids, dtype=torch.long)

    def _process_llm_output(
        self, llm_output: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Processes raw LLM output into texts and token tensors, handling EOS."""
        texts = []
        token_ids_list = []

        for item in llm_output:
            text, token_ids = self._process_single_output_item(item)
            texts.append(text)
            token_ids_list.append(token_ids)

        return texts, token_ids_list

    def _generate_completions(
        self,
        llm: InferenceClient,
        sampling_params: Dict[str, Any],
        state: EnvState,
        **kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generates completions using the LLM for the current state.

        Args:
            llm: The custom inference client for engine.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens, do_sample).
            state: The current EnvState containing input_ids and attention_mask.
            **kwargs: Additional custom arguments.

        Returns:
            A tuple containing:
            - completion_texts: List of decoded completion strings (up to, but not including, EOS).
            - completion_tokens: List of completion token tensors (unpadded, includes EOS if present).
        """
        output = llm.generate(
            prompts=state.prompt,
            sampling_params=sampling_params,
        )

        # Unpack completions
        completion_texts, completion_ids = self._process_llm_output(output)

        return completion_texts, completion_ids

    @torch.inference_mode()
    def rollout(
        self,
        llm: InferenceClient,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[EpisodeData]:
        """
        Performs a rollout step: gets a batch, generates completions, calculates rewards,
        and returns structured episode data.

        Args:
            llm: The custom inference client for engine.
            sampling_params: Dictionary of generation arguments (e.g., max_new_tokens).
            **kwargs: Additional custom arguments.

        Returns:
            A list of EpisodeData objects, one for each generated sample in the batch
            (batch_size * group_size samples). Returns an empty list if the dataset is exhausted.
        """
        if sampling_params.get('n', 1) > 1:
            raise ValueError(
                'Set group_size during initialization instead of using n.'
            )

        state = self._reset()
        if state is None:
            logger.warning(
                f"Rank {self.rank}: Reset returned None; dataset exhausted."
            )
            return []

        completions, completion_tokens = self._generate_completions(
            llm, sampling_params, state, **kwargs
        )

        return self._to_episodes(state, completions, completion_tokens)

        # rewards_dict = self._calculate_rewards(completions, state.ground_truth)

        # # transform multiple rewards (e.g accuracy, format etc) into a single scalar for the same output
        # terminal_rewards = self._transform_rewards(rewards_dict)
        # prompt_tokens = [
        #     state.input_ids[i][state.attention_mask[i] == 1].cpu()
        #     for i in range(len(state.input_ids))
        # ]

        # full_sequences = [
        #     torch.concat([prompt_toks, completion_toks]).long()
        #     for prompt_toks, completion_toks in zip(
        #         prompt_tokens, completion_tokens
        #     )
        # ]

        # # States: tokens 0 to N-1; Actions: tokens 1 to N
        # state_sequences = [seq[:-1] for seq in full_sequences]
        # action_sequences = [seq[1:] for seq in full_sequences]

        # results = []

        # for i in range(len(completions)):
        #     states = state_sequences[i]
        #     actions = action_sequences[i]
        #     prompt_len = len(prompt_tokens[i])
        #     completion_len = len(completion_tokens[i])

        #     # Do not include the prompt tokens in the loss
        #     # for example, if we have a sequence token ids: [1, 2, 3, 4, 5, 6, 7]
        #     # where [1, 2, 3, 4] are the prompt tokens
        #     # and [5, 6, 7] are the completion tokens
        #     # the, the loss mask will be [0, 0, 0, 1, 1, 1]

        #     loss_mask = torch.zeros_like(actions, dtype=torch.bool)
        #     loss_mask[prompt_len - 1 :] = True

        #     assert loss_mask.sum().item() == completion_len

        #     meta = EpisodeMetadata(
        #         prompt=state.prompt[i],
        #         prompt_length=prompt_len,
        #         completion=completions[i],
        #         completion_length=completion_len,
        #         reward_dict={k: v[i] for k, v in rewards_dict.items()},
        #         ground_truth=state.ground_truth[i]
        #     )

        #     ep = EpisodeData(
        #         states=states,
        #         actions=actions,
        #         loss_mask=loss_mask,
        #         terminal_reward=terminal_rewards[i],
        #         metadata=meta,
        #     )

        #     results.append(ep)

        # return results
