import logging
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset

# Objects under test
from rl4llm.core.base_env import (
    BaseMDPEnv,
    ChatMessage,
    EnvState,
    SampleState,
)
from rl4llm.core.base_inference_client import InferenceClient
from rl4llm.envs.sgl_env import SglMDPEnv

# --- Fixtures Specific to SglMDPEnv ---


@pytest.fixture
def mock_inference_client():
    """Provides a mock InferenceClient."""
    client = MagicMock(spec=InferenceClient)

    def mock_generate(prompts: List[str], sampling_params: Dict[str, Any]):
        # Simple mock: return a response based on the prompt index
        outputs = []
        for i, prompt in enumerate(prompts):
            outputs.append({'text': f"Generated response {i + 1}"})
        return outputs

    client.generate = MagicMock(side_effect=mock_generate)
    return client


@pytest.fixture
def sgl_mdp_env(mock_dataset, mock_tokenizer, mock_reward_function):
    """Provides an instance of SglMDPEnv."""
    return SglMDPEnv(
        dataset=mock_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=2,
        group_size=1,
        max_steps=1,  # SglMDPEnv is typically single-step
        rank=0,
        world_size=1,
    )


@pytest.fixture
def env_state_for_sgl():
    """Provides an initial EnvState with multiple not-done samples."""
    sample1 = SampleState(
        messages=[ChatMessage(role='user', content='Prompt 1')],
        ground_truth='GT1',
        init_msg_size=1,
        current_step=0,
        done=False,
    )
    sample2 = SampleState(
        messages=[ChatMessage(role='user', content='Prompt 2')],
        ground_truth='GT2',
        init_msg_size=1,
        current_step=0,
        done=False,
    )
    return EnvState(sample_states=[sample1, sample2])


# --- Tests for SglMDPEnv ---


def test_sgl_mdp_env_inheritance(sgl_mdp_env):
    """Tests that SglMDPEnv inherits from BaseMDPEnv."""
    assert isinstance(sgl_mdp_env, BaseMDPEnv)


@patch.object(SglMDPEnv, '_convert_to_batch_prompts')
def test_run_interaction_loop_success(
    mock_convert_prompts,
    sgl_mdp_env,
    mock_inference_client,
    env_state_for_sgl,
):
    """Tests the successful execution of the interaction loop."""
    # Arrange
    mock_prompts = ['Formatted Prompt 1', 'Formatted Prompt 2']
    mock_convert_prompts.return_value = mock_prompts
    sampling_params = {'temperature': 0.7, 'max_new_tokens': 50}
    initial_state_copy = env_state_for_sgl.model_copy(
        deep=True
    )  # Avoid modifying fixture

    # Act
    final_env_state = sgl_mdp_env._run_interaction_loop(
        initial_state_copy, mock_inference_client, sampling_params
    )

    # Assert
    # Check that prompt conversion was called correctly
    mock_convert_prompts.assert_called_once_with(
        [s.messages for s in initial_state_copy.sample_states]
    )

    # Check that inference client was called correctly
    mock_inference_client.generate.assert_called_once_with(
        prompts=mock_prompts, sampling_params=sampling_params
    )

    # Check that the returned state is the same object (modified in-place)
    assert final_env_state is initial_state_copy

    # Check state updates for each sample
    assert len(final_env_state.sample_states) == 2
    # Sample 1
    sample1 = final_env_state.sample_states[0]
    assert sample1.done is True
    assert sample1.current_step == 1
    assert len(sample1.messages) == 2  # Initial user + generated assistant
    assert sample1.messages[1].role == 'assistant'
    assert (
        sample1.messages[1].content == 'Generated response 1'
    )  # From mock_generate
    # Sample 2
    sample2 = final_env_state.sample_states[1]
    assert sample2.done is True
    assert sample2.current_step == 1
    assert len(sample2.messages) == 2
    assert sample2.messages[1].role == 'assistant'
    assert sample2.messages[1].content == 'Generated response 2'


@patch.object(SglMDPEnv, '_convert_to_batch_prompts')
def test_run_interaction_loop_llm_error(
    mock_convert_prompts,
    sgl_mdp_env,
    mock_inference_client,
    env_state_for_sgl,
    caplog,
):
    """Tests the interaction loop when the LLM call fails."""
    # Arrange
    mock_prompts = ['Formatted Prompt 1', 'Formatted Prompt 2']
    mock_convert_prompts.return_value = mock_prompts
    sampling_params = {'temperature': 0.7, 'max_new_tokens': 50}
    error_message = 'LLM generation failed'
    mock_inference_client.generate.side_effect = Exception(error_message)
    initial_state_copy = env_state_for_sgl.model_copy(deep=True)

    # Act
    with caplog.at_level(logging.ERROR):
        final_env_state = sgl_mdp_env._run_interaction_loop(
            initial_state_copy, mock_inference_client, sampling_params
        )

    # Assert
    # Check that the original state is returned
    assert final_env_state is initial_state_copy
    # Check that samples are NOT marked as done and messages are not added
    sample1 = final_env_state.sample_states[0]
    assert sample1.done is False
    assert sample1.current_step == 0
    assert len(sample1.messages) == 1  # Only initial user message
    sample2 = final_env_state.sample_states[1]
    assert sample2.done is False
    assert sample2.current_step == 0
    assert len(sample2.messages) == 1

    # Check logs
    assert (
        f"Rank {sgl_mdp_env.rank}: Error during LLM generation" in caplog.text
    )
    assert error_message in caplog.text


def test_run_interaction_loop_empty_state(
    sgl_mdp_env,
    mock_inference_client,
):
    """Tests the interaction loop with an empty initial state."""
    # Arrange
    empty_state = EnvState(sample_states=[])
    sampling_params = {'temperature': 0.7, 'max_new_tokens': 50}

    # Act
    final_env_state = sgl_mdp_env._run_interaction_loop(
        empty_state, mock_inference_client, sampling_params
    )

    # Assert
    assert final_env_state is empty_state
    assert len(final_env_state.sample_states) == 0
    mock_inference_client.generate.assert_not_called()


@patch.object(SglMDPEnv, '_convert_to_batch_prompts')
def test_run_interaction_loop_skips_done_state(
    mock_convert_prompts,
    sgl_mdp_env,
    mock_inference_client,
):
    """Tests that already 'done' samples are skipped."""
    # Arrange
    sample_done = SampleState(
        messages=[ChatMessage(role='user', content='Prompt Done')],
        ground_truth='GT_Done',
        init_msg_size=1,
        current_step=1,
        done=True,  # Already done
    )
    sample_not_done = SampleState(
        messages=[ChatMessage(role='user', content='Prompt Not Done')],
        ground_truth='GT_NotDone',
        init_msg_size=1,
        current_step=0,
        done=False,
    )
    initial_state = EnvState(sample_states=[sample_done, sample_not_done])
    initial_state_copy = initial_state.model_copy(deep=True)

    # Mock prompt conversion only for the not-done sample
    mock_convert_prompts.return_value = ['Formatted Prompt Not Done']
    sampling_params = {'temperature': 0.7, 'max_new_tokens': 50}

    # Mock LLM generate to only expect one prompt
    mock_inference_client.generate.side_effect = (
        lambda prompts, sampling_params: (
            [{'text': 'Generated for Not Done'}]
            if len(prompts) == 1
            else pytest.fail('LLM called with wrong number of prompts')
        )
    )

    # Act
    final_env_state = sgl_mdp_env._run_interaction_loop(
        initial_state_copy, mock_inference_client, sampling_params
    )

    # Assert
    # Check prompt conversion was called only with the not-done messages
    mock_convert_prompts.assert_called_once_with([sample_not_done.messages])

    # Check LLM call
    mock_inference_client.generate.assert_called_once_with(
        prompts=['Formatted Prompt Not Done'], sampling_params=sampling_params
    )

    # Check states
    final_sample_done = final_env_state.sample_states[0]
    final_sample_not_done = final_env_state.sample_states[1]

    # Done sample should remain unchanged
    assert final_sample_done.done is True
    assert final_sample_done.current_step == 1
    assert len(final_sample_done.messages) == 1  # No new message added

    # Not-done sample should be updated
    assert final_sample_not_done.done is True
    assert final_sample_not_done.current_step == 1
    assert len(final_sample_not_done.messages) == 2
    assert final_sample_not_done.messages[1].role == 'assistant'
    assert final_sample_not_done.messages[1].content == 'Generated for Not Done'
