# import pytest
# import torch
# import datasets
# from unittest.mock import MagicMock, patch

# from typing import Dict, Any, List, Tuple, Union, Optional

# from rl4llm.envs.sgl_env import (
#     EnvState,
#     EpisodeData,
#     EpisodeMetadata,
#     SglMDPEnv,
#     InferenceClient,
# )


# # Mock logger
# class MockLogger:
#     def warning(self, msg):
#         print(f"WARN: {msg}")  # Or pass


# logger = MockLogger()


# # --- Fixtures ---


# @pytest.fixture
# def dummy_dataset():
#     """Returns a dummy dataset."""
#     data = {
#         'prompt': ['hello world', 'foo bar'],
#         'ground_truth': ['expected output one', 'expected output two'],
#     }
#     return datasets.Dataset.from_dict(data)


# @pytest.fixture
# def sgl_mdp_env(dummy_dataset, mock_tokenizer, mock_reward_function):
#     """Provides an instance of SglMDPEnv with a mock tokenizer."""
#     # Mock the necessary parts of BaseMDPEnv initialization if needed
#     env = SglMDPEnv(
#         dataset=dummy_dataset,
#         tokenizer=mock_tokenizer,
#         reward_functions=[mock_reward_function],
#         batch_size=1,
#         group_size=1,
#     )
#     # Mock methods inherited or composed that are not under test
#     env._reset = MagicMock()
#     env._to_episodes = MagicMock(return_value=["episode_data"])
#     env._calculate_rewards = MagicMock()
#     env._transform_rewards = MagicMock()
#     return env


# @pytest.fixture
# def mock_llm():
#     """Provides a mock InferenceClient."""
#     llm = MagicMock(spec=InferenceClient)
#     return llm


# # --- Test Functions ---


# @pytest.mark.parametrize(
#     "batch_size, prompt_lens, completion_lens",
#     [
#         (1, [5], [3]),  # Single item batch
#         (2, [4, 6], [2, 4]),  # Multi-item batch with different lengths
#         (1, [5], [0]),  # Single item with empty completion
#         (
#             2,
#             [3, 0],
#             [2, 4],
#         ),  # Batch including an empty prompt (handle edge case)
#         (1, [0], [3]),  # Single item with empty prompt
#     ],
# )
# def test_to_episodes_structure_and_content(
#     sgl_mdp_env: SglMDPEnv,
#     batch_size: int,
#     prompt_lens: List[int],
#     completion_lens: List[int],
# ):
#     """Tests that _to_episodes correctly structures output and calculates masks/sequences."""
#     # --- Arrange ---
#     prompts = [f"prompt_{i}" for i in range(batch_size)]
#     completions = [
#         f"completion_{i}" * (completion_lens[i] > 0) for i in range(batch_size)
#     ]
#     ground_truths = [f"gt_{i}" for i in range(batch_size)]

#     prompt_token_lists = [
#         torch.arange(i * 100 + 1, i * 100 + p_len + 1)
#         for i, p_len in enumerate(prompt_lens)
#     ]

#     max_prompt_len = max(prompt_lens) if prompt_lens else 0
#     if max_prompt_len == 0 and batch_size > 0:
#         max_prompt_len = 1  # Avoid zero dimension if batch exists

#     padded_input_ids = torch.zeros(
#         (batch_size, max_prompt_len), dtype=torch.long
#     )
#     padded_attn_mask = torch.zeros(
#         (batch_size, max_prompt_len), dtype=torch.long
#     )
#     for i, tokens in enumerate(prompt_token_lists):
#         seq_len = len(tokens)
#         if seq_len > 0:
#             padded_input_ids[i, :seq_len] = tokens
#             padded_attn_mask[i, :seq_len] = 1

#     completion_tokens = [
#         torch.arange(i * 100 + p_len + 1, i * 100 + p_len + c_len + 1)
#         for i, (p_len, c_len) in enumerate(zip(prompt_lens, completion_lens))
#     ]

#     state = EnvState(
#         input_ids=padded_input_ids,
#         attention_mask=padded_attn_mask,
#         prompt=prompts,
#         ground_truth=ground_truths,
#     )

#     mock_rewards = {
#         f"reward_{k}": [float(k + j) for j in range(batch_size)]
#         for k in range(2)
#     }
#     mock_terminal_rewards = [float(10 + i) for i in range(batch_size)]

#     # --- Act ---
#     # Call the method on the instance created by the fixture
#     results = sgl_mdp_env._to_episodes(state, completions, completion_tokens)

#     # --- Assert ---
#     # Check mocks were called inside the 'with' block
#     sgl_mdp_env._calculate_rewards.assert_called_once_with(
#         completions, ground_truths
#     )
#     sgl_mdp_env._transform_rewards.assert_called_once_with(mock_rewards)

#     # Continue with the rest of the assertions outside the 'with' block
#     assert isinstance(results, list)
#     assert len(results) == batch_size

#     for i in range(batch_size):
#         ep_data = results[i]
#         prompt_len = prompt_lens[i]
#         completion_len = completion_lens[i]
#         full_len = prompt_len + completion_len

#         assert isinstance(ep_data, EpisodeData)

#         expected_prompt_tokens = prompt_token_lists[i]
#         expected_completion_tokens = completion_tokens[i]
#         expected_full_sequence = torch.cat(
#             [expected_prompt_tokens, expected_completion_tokens]
#         ).long()

#         expected_seq_len = full_len - 1 if full_len > 0 else 0
#         assert isinstance(ep_data.states, torch.Tensor)
#         assert ep_data.states.shape == (expected_seq_len,)
#         assert ep_data.states.dtype == torch.long

#         assert isinstance(ep_data.actions, torch.Tensor)
#         assert ep_data.actions.shape == (expected_seq_len,)
#         assert ep_data.actions.dtype == torch.long

#         assert isinstance(ep_data.loss_mask, torch.Tensor)
#         assert ep_data.loss_mask.shape == ep_data.actions.shape
#         assert ep_data.loss_mask.dtype == torch.bool

#         if full_len > 0:
#             torch.testing.assert_close(
#                 ep_data.states, expected_full_sequence[:-1]
#             )
#             torch.testing.assert_close(
#                 ep_data.actions, expected_full_sequence[1:]
#             )

#             expected_mask = torch.zeros_like(ep_data.actions, dtype=torch.bool)
#             # Mask should start at index `prompt_len - 1` if prompt_len > 0
#             # If prompt_len is 0, mask starts at index 0
#             mask_start_index = (
#                 max(0, prompt_len - 1) if completion_len > 0 else 0
#             )
#             if completion_len > 0:
#                 expected_mask[mask_start_index:] = True

#             torch.testing.assert_close(
#                 ep_data.loss_mask, expected_mask, msg=f"Item {i} mask mismatch"
#             )
#             assert (
#                 ep_data.loss_mask.sum().item() == completion_len
#             ), f"Item {i} mask sum mismatch"
#         else:
#             assert ep_data.loss_mask.sum().item() == 0

#         assert ep_data.terminal_reward == mock_terminal_rewards[i]

#         meta = ep_data.metadata
#         assert isinstance(meta, EpisodeMetadata)
#         assert meta.prompt == prompts[i]
#         assert meta.prompt_length == prompt_len
#         assert meta.completion == completions[i]
#         assert meta.completion_length == completion_len
#         assert meta.ground_truth == ground_truths[i]
#         expected_reward_dict = {k: v[i] for k, v in mock_rewards.items()}
#         assert meta.reward_dict == expected_reward_dict


# # @pytest.mark.parametrize(
# #     "item, expected_text, expected_tokens",
# #     [
# #         pytest.param(
# #             {'text': 'abc', 'meta_info': {'finish_reason': {'type': 'stop'}}},
# #             'abc',
# #             torch.tensor(
# #                 [197, 198, 199, 2], dtype=torch.long
# #             ),  # ord('a')+100=197, etc. + EOS
# #             id="normal_text_add_eos",
# #         ),
# #         pytest.param(
# #             {'text': 'def', 'meta_info': {'finish_reason': {'type': 'length'}}},
# #             'def',
# #             torch.tensor([200, 201, 202], dtype=torch.long),  # No EOS added
# #             id="finish_reason_length",
# #         ),
# #         pytest.param(
# #             {'text': 'ijk'},  # No meta_info
# #             'ijk',
# #             torch.tensor([205, 206, 207], dtype=torch.long),  # No EOS added
# #             id="no_meta_info",
# #         ),
# #         pytest.param(
# #             {
# #                 'text': 'lmn',
# #                 'meta_info': {},
# #             },  # meta_info exists, but no finish_reason
# #             'lmn',
# #             torch.tensor([208, 209, 210], dtype=torch.long),  # No EOS added
# #             id="meta_info_no_finish_reason",
# #         ),
# #         pytest.param(
# #             {
# #                 'text': 'opq',
# #                 'meta_info': {'finish_reason': {}},
# #             },  # finish_reason exists, but no type
# #             'opq',
# #             torch.tensor([211, 212, 213], dtype=torch.long),  # No EOS added
# #             id="finish_reason_no_type",
# #         ),
# #         pytest.param(
# #             {
# #                 'text': '',
# #                 'meta_info': {'finish_reason': {'type': 'stop'}},
# #             },  # Empty text results in default
# #             "I can't help with this question.",
# #             # Tokenize the default text and add EOS
# #             torch.tensor(
# #                 [ord(c) + 100 for c in "I can't help with this question."]
# #                 + [2],
# #                 dtype=torch.long,
# #             ),
# #             id="empty_text_input_generates_default_and_eos",
# #         ),
# #         pytest.param(
# #             {
# #                 'text': '',
# #                 'meta_info': {'finish_reason': {'type': 'length'}},
# #             },  # Empty text results in default
# #             "I can't help with this question.",
# #             # Tokenize the default text, NO EOS due to length
# #             torch.tensor(
# #                 [ord(c) + 100 for c in "I can't help with this question."],
# #                 dtype=torch.long,
# #             ),
# #             id="empty_text_input_generates_default_no_eos_for_length",
# #         ),
# #         pytest.param(
# #             # Simulate a case where tokenizer returns empty list for non-empty text (unlikely but possible)
# #             {'text': 'xyz', 'meta_info': {'finish_reason': {'type': 'stop'}}},
# #             'xyz',
# #             torch.tensor(
# #                 [2], dtype=torch.long
# #             ),  # Should still add EOS if reason is stop
# #             id="empty_tokens_add_eos",
# #         ),
# #     ],
# # )
# # def test_process_single_output_item(
# #     sgl_mdp_env, mock_tokenizer, item, expected_text, expected_tokens
# # ):
# #     """Tests the processing of a single LLM output item including EOS logic."""
# #     # Special handling for the empty_tokens_add_eos case to force empty token list
# #     if (
# #         item['text'] == 'xyz'
# #         and item['meta_info']['finish_reason']['type'] == 'stop'
# #     ):
# #         mock_tokenizer.side_effect = lambda text, **kwargs: {'input_ids': []}
# #     else:
# #         # Reset side effect for other tests
# #         def _tokenizer_logic(text, **kwargs):
# #             if not text:
# #                 return {'input_ids': []}
# #             return {'input_ids': [ord(c) + 100 for c in text]}

# #         mock_tokenizer.side_effect = _tokenizer_logic

# #     text, token_ids = sgl_mdp_env._process_single_output_item(item)
# #     assert text == expected_text
# #     assert torch.equal(token_ids, expected_tokens)

# #     # Reset side effect after the test using it
# #     def _tokenizer_logic(text, **kwargs):
# #         if not text:
# #             return {'input_ids': []}
# #         return {'input_ids': [ord(c) + 100 for c in text]}

# #     mock_tokenizer.side_effect = _tokenizer_logic


# # def test_process_llm_output(sgl_mdp_env):
# #     """Tests processing a list of LLM output items."""
# #     llm_output = [
# #         {'text': 'ab', 'meta_info': {'finish_reason': {'type': 'stop'}}},
# #         {'text': 'cd', 'meta_info': {'finish_reason': {'type': 'length'}}},
# #     ]
# #     expected_texts = ['ab', 'cd']
# #     expected_token_lists = [
# #         torch.tensor([197, 198, 2], dtype=torch.long),  # EOS added
# #         torch.tensor([199, 200], dtype=torch.long),  # No EOS added
# #     ]

# #     texts, token_ids_list = sgl_mdp_env._process_llm_output(llm_output)

# #     assert texts == expected_texts
# #     assert len(token_ids_list) == len(expected_token_lists)
# #     for i in range(len(token_ids_list)):
# #         assert torch.equal(token_ids_list[i], expected_token_lists[i])


# # def test_generate_completions(sgl_mdp_env, mock_llm):
# #     """Tests the generation of completions by mocking the LLM call."""
# #     mock_output = [
# #         {'text': 'response1', 'meta_info': {'finish_reason': {'type': 'stop'}}},
# #         {
# #             'text': 'response2',
# #             'meta_info': {'finish_reason': {'type': 'length'}},
# #         },
# #     ]
# #     mock_llm.generate.return_value = mock_output
# #     sampling_params = {'max_new_tokens': 10}
# #     state = EnvState(prompt="Test prompt")

# #     expected_texts = ['response1', 'response2']
# #     expected_tokens = [
# #         torch.tensor(
# #             [ord(c) + 100 for c in 'response1'] + [2], dtype=torch.long
# #         ),
# #         torch.tensor([ord(c) + 100 for c in 'response2'], dtype=torch.long),
# #     ]

# #     texts, tokens = sgl_mdp_env._generate_completions(
# #         mock_llm, sampling_params, state
# #     )

# #     mock_llm.generate.assert_called_once_with(
# #         prompts=state.prompt, sampling_params=sampling_params
# #     )
# #     assert texts == expected_texts
# #     assert len(tokens) == len(expected_tokens)
# #     assert torch.equal(tokens[0], expected_tokens[0])
# #     assert torch.equal(tokens[1], expected_tokens[1])


# # def test_rollout_success(sgl_mdp_env, mock_llm):
# #     """Tests a successful rollout sequence."""
# #     # Mock _reset to return a valid state
# #     mock_state = EnvState(prompt="Initial prompt")
# #     sgl_mdp_env._reset.return_value = mock_state

# #     # Mock the internal call to _generate_completions
# #     mock_completions = ['comp1']
# #     mock_tokens = [torch.tensor([1, 2, 3])]  # Example tokens
# #     sgl_mdp_env._generate_completions = MagicMock(
# #         return_value=(mock_completions, mock_tokens)
# #     )

# #     sampling_params = {'max_new_tokens': 5}
# #     result = sgl_mdp_env.rollout(mock_llm, sampling_params)

# #     sgl_mdp_env._reset.assert_called_once()
# #     sgl_mdp_env._generate_completions.assert_called_once_with(
# #         mock_llm, sampling_params, mock_state
# #     )
# #     sgl_mdp_env._to_episodes.assert_called_once_with(
# #         mock_state, mock_completions, mock_tokens
# #     )
# #     assert result == [
# #         "episode_data"
# #     ]  # Matches the mock return value of _to_episodes


# def test_rollout_dataset_exhausted(sgl_mdp_env, mock_llm):
#     """Tests rollout when the dataset is exhausted (_reset returns None)."""
#     sgl_mdp_env._reset.return_value = None
#     sampling_params = {'max_new_tokens': 5}

#     result = sgl_mdp_env.rollout(mock_llm, sampling_params)

#     sgl_mdp_env._reset.assert_called_once()
#     # Ensure _generate_completions is not called if reset returns None
#     if hasattr(sgl_mdp_env, '_generate_completions') and isinstance(
#         sgl_mdp_env._generate_completions, MagicMock
#     ):
#         sgl_mdp_env._generate_completions.assert_not_called()
#     mock_llm.generate.assert_not_called()  # Should not proceed if reset fails
#     sgl_mdp_env._to_episodes.assert_not_called()
#     assert result == []


# def test_rollout_invalid_n_param(sgl_mdp_env, mock_llm):
#     """Tests that rollout raises ValueError if 'n' > 1 in sampling_params."""
#     sampling_params = {'max_new_tokens': 5, 'n': 2}
#     with pytest.raises(
#         ValueError,
#         match='Set group_size during initialization instead of using n.',
#     ):
#         sgl_mdp_env.rollout(mock_llm, sampling_params)
