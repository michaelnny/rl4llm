import re
import uuid
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from datasets import Dataset
from pydantic import ValidationError
from transformers import PreTrainedTokenizer

# Objects under test
from rl4llm.core.base_env import (
    BaseMDPEnv,
    BaseRewardFunction,
    ChatMessage,
    EnvState,
    EpisodeData,
    SampleState,
    find_subsequence,
)

# --- Fixtures ---


# Consistent Mock Tokenizer Fixture
@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '<eos>'
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1  # EOS

    # These are reset by fixture scope if 'function'
    tokenizer.word_map = {}
    tokenizer.next_word_id = 100

    tokenizer.role_map_tok_ids = {
        'user': 10,
        'assistant': 11,
        'system': 12,
        'tool': 13,
    }
    # Text markers for roles (used by tokenize=False and parsed by encode)
    tokenizer.role_markers_text = {
        role_id: role_name + ':'
        for role_name, role_id in tokenizer.role_map_tok_ids.items()
    }
    tokenizer.text_to_role_id = {
        text_marker: role_id
        for role_id, text_marker in tokenizer.role_markers_text.items()
    }

    def _get_word_id(word):
        if word not in tokenizer.word_map:
            tokenizer.word_map[word] = tokenizer.next_word_id
            tokenizer.next_word_id += 1
        return tokenizer.word_map[word]

    tokenizer._get_word_id = _get_word_id

    def _mock_apply_chat_template_consistent(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        if tokenize:
            output_tokens = []
            for i, msg_dict in enumerate(messages):
                role = msg_dict.get('role')
                content = msg_dict.get('content')
                output_tokens.append(
                    tokenizer.role_map_tok_ids.get(role, 99)
                )  # Role ID
                if content:
                    for word in content.split():
                        output_tokens.append(_get_word_id(word))  # Word IDs

                is_last_msg = i == len(messages) - 1
                if not (is_last_msg and continue_final_message):
                    output_tokens.append(tokenizer.eos_token_id)  # EOS ID

            if (
                add_generation_prompt
            ):  # Not used by _convert_to_episodes' calls to this
                output_tokens.append(tokenizer.role_map_tok_ids['assistant'])
            return output_tokens
        else:  # tokenize=False
            parts = []
            for i, msg_dict in enumerate(messages):
                role = msg_dict.get('role')
                content = msg_dict.get('content')

                parts.append(
                    tokenizer.role_markers_text[
                        tokenizer.role_map_tok_ids.get(role, 99)
                    ]
                )
                if content:
                    parts.append(content)
                    for word in content.split():
                        _get_word_id(word)  # Ensure in map

                is_last_msg = i == len(messages) - 1
                if not (is_last_msg and continue_final_message):
                    parts.append(tokenizer.eos_token)

            if (
                add_generation_prompt
            ):  # Not used by _convert_to_episodes' calls to this
                parts.append(
                    tokenizer.role_markers_text[
                        tokenizer.role_map_tok_ids['assistant']
                    ]
                )
            return ' '.join(parts)

    def _mock_encode_consistent(text, add_special_tokens=False):
        tokens = []
        if not text:
            return tokens

        words = text.split()
        for word in words:
            if word == tokenizer.eos_token:
                tokens.append(tokenizer.eos_token_id)
            elif word == tokenizer.pad_token:
                tokens.append(tokenizer.pad_token_id)
            elif word in tokenizer.text_to_role_id:  # e.g. "user:"
                tokens.append(tokenizer.text_to_role_id[word])
            else:  # Content word
                tokens.append(_get_word_id(word))
        return tokens

    tokenizer.apply_chat_template = MagicMock(
        side_effect=_mock_apply_chat_template_consistent
    )
    tokenizer.encode = MagicMock(side_effect=_mock_encode_consistent)
    tokenizer.chat_template = 'mock_template'  # For _setup_tokenizer check

    return tokenizer


@pytest.fixture
def mock_reward_function():
    """Provides a simple mock BaseRewardFunction (single-sample processing)."""

    class MockReward(BaseRewardFunction):
        def __init__(self, name='mock_reward', reward_value=1.0):
            super().__init__(name)
            self.reward_value = reward_value

        def __call__(
            self,
            messages: List[ChatMessage],  # Single sample's messages
            ground_truth: Union[str, float, int],  # Single sample's GT
            **kwargs: Any,
        ) -> float:  # Returns a single float
            return self.reward_value  # Return the value for this one sample

    return MockReward()


@pytest.fixture
def mock_reward_function_alt():
    """Provides a second mock BaseRewardFunction with a different value (single-sample processing)."""

    class MockRewardAlt(BaseRewardFunction):
        def __init__(self, name='mock_reward_alt', reward_value=0.5):
            super().__init__(name)
            self.reward_value = reward_value

        def __call__(
            self,
            messages: List[ChatMessage],  # Single sample's messages
            ground_truth: Union[str, float, int],  # Single sample's GT
            **kwargs: Any,
        ) -> float:  # Returns a single float
            return self.reward_value

    return MockRewardAlt()


@pytest.fixture
def sample_raw_data():
    """Provides sample raw data mimicking dataset rows."""
    return [
        {
            'messages': [{'role': 'user', 'content': 'Hello there'}],
            'ground_truth': 'General Kenobi',
        },
        {
            'messages': [
                {'role': 'user', 'content': 'Explain RLHF'},
                {'role': 'system', 'content': 'Be concise'},
            ],
            'ground_truth': 'Reinforcement Learning from Human Feedback',
        },
    ]


@pytest.fixture
def mock_dataset(sample_raw_data):
    """Provides a mock datasets.Dataset."""
    return Dataset.from_list(sample_raw_data)


# Minimal concrete subclass for testing BaseMDPEnv methods
class MockMDPEnv(BaseMDPEnv):
    def _run_interaction_loop(
        self,
        env_state: EnvState,
        llm: Any,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        # Simple mock: adds a fixed assistant message based on ground truth
        for sample_state in env_state.sample_states:
            if not sample_state.done:
                response_content = (
                    f"Assistant response for {sample_state.ground_truth}"
                )
                sample_state.messages.append(
                    ChatMessage(role='assistant', content=response_content)
                )
                sample_state.current_step += 1
                if sample_state.current_step >= self.max_steps:
                    sample_state.done = True
        return env_state


@pytest.fixture
def base_mdp_env(mock_dataset, mock_tokenizer, mock_reward_function):
    """Provides an instance of the minimal concrete MockMDPEnv."""
    return MockMDPEnv(
        dataset=mock_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=2,
        group_size=1,  # Keep group size 1 for simplicity in most tests
        max_steps=1,
        rank=0,
        world_size=1,
    )


@pytest.fixture
def sample_state_fixture():
    """Provides a sample SampleState object."""
    return SampleState(
        messages=[
            ChatMessage(role='user', content='Initial prompt'),
            ChatMessage(role='assistant', content='First response'),
        ],
        ground_truth='Expected outcome',
        init_msg_size=1,
        current_step=1,
        done=True,
    )


@pytest.fixture
def env_state_fixture(sample_state_fixture):
    """Provides a sample EnvState object."""
    # Create a copy to avoid modifying the original fixture instance
    state_copy = sample_state_fixture.model_copy(deep=True)
    state_copy.id = str(uuid.uuid4().hex)  # Give it a unique ID
    return EnvState(sample_states=[state_copy])


# --- Tests ---


# Test Pydantic Models (ChatMessage, EpisodeData, SampleState, EnvState)
def test_chat_message_creation():
    """Tests successful creation of a ChatMessage."""
    msg = ChatMessage(role='user', content='Hello')
    assert msg.role == 'user'
    assert msg.content == 'Hello'


@pytest.mark.parametrize('invalid_role', ['agent', '', None, 123])
def test_chat_message_invalid_role(invalid_role):
    """Tests that ChatMessage raises ValueError for invalid roles."""
    with pytest.raises(ValidationError):
        ChatMessage(role=invalid_role, content='Test')


def test_episode_data_creation():
    """Tests successful creation of EpisodeData with matching tensor shapes."""
    ep = EpisodeData(
        states=torch.tensor([1, 2]),
        actions=torch.tensor([2, 3]),
        loss_mask=torch.tensor([0, 1], dtype=torch.bool),
        terminal_reward=1.0,
        ground_truth='test',
        reward_dict={'r1': 1.0},
        chat_history=[ChatMessage(role='user', content='hi')],
        prompt_length=1,
        completion_length=1,
    )
    assert ep.states.shape == ep.actions.shape == ep.loss_mask.shape


def test_episode_data_shape_mismatch():
    """Tests that EpisodeData raises ValueError for tensor shape mismatch."""
    with pytest.raises(ValidationError, match='Tensor shape mismatch'):
        EpisodeData(
            states=torch.tensor([1, 2]),
            actions=torch.tensor([2, 3, 4]),  # Different shape
            loss_mask=torch.tensor([0, 1], dtype=torch.bool),
            terminal_reward=1.0,
            ground_truth='test',
            reward_dict={'r1': 1.0},
            chat_history=[ChatMessage(role='user', content='hi')],
            prompt_length=1,
            completion_length=1,
        )


def test_sample_state_creation(sample_state_fixture):
    """Tests successful creation of a SampleState."""
    assert isinstance(sample_state_fixture, SampleState)
    assert len(sample_state_fixture.id) > 0  # Default ID is generated
    assert sample_state_fixture.init_msg_size == 1
    assert sample_state_fixture.done is True


def test_env_state_creation(env_state_fixture):
    """Tests successful creation of an EnvState."""
    assert isinstance(env_state_fixture, EnvState)
    assert len(env_state_fixture.sample_states) == 1
    assert isinstance(env_state_fixture.sample_states[0], SampleState)


# Test BaseRewardFunction
def test_base_reward_function_creation():
    """Tests successful creation of a BaseRewardFunction subclass instance."""

    class MyReward(BaseRewardFunction):
        def __call__(self, *args, **kwargs):
            return [0.0]  # Implement abstract method

    reward_fn = MyReward(name='my-reward-123')
    assert reward_fn.name == 'my-reward-123'


@pytest.mark.parametrize(
    'invalid_name', ['', 'invalid name', 'name!', None, 123]
)
def test_base_reward_function_invalid_name(invalid_name):
    """Tests that BaseRewardFunction raises ValueError for invalid names."""

    class MyReward(BaseRewardFunction):
        def __call__(self, *args, **kwargs):
            return [0.0]

    with pytest.raises((ValueError, TypeError)):
        MyReward(name=invalid_name)


def test_base_reward_function_abstract_call():
    """Tests that calling __call__ on the abstract base class raises NotImplementedError."""
    with pytest.raises(TypeError):  # Cannot instantiate abstract class
        reward_fn = BaseRewardFunction('test')

    class MyIncompleteReward(BaseRewardFunction):  # Missing __call__ impl
        pass

    with pytest.raises(TypeError):  # Cannot instantiate abstract class
        reward_fn = MyIncompleteReward('test')

    class MyReward(BaseRewardFunction):  # Has __call__
        def __call__(self, *args, **kwargs):
            return [0.0]

    reward_fn = MyReward('test')  # Instantiation works
    # Provide dummy args matching signature
    assert reward_fn(batch_messages=[[]], batch_ground_truths=['gt']) == [
        0.0
    ]  # Call works


# Test find_subsequence helper
@pytest.mark.parametrize(
    'main_list, sub_list, expected_index',
    [
        ([1, 2, 3, 4, 5], [3, 4], 2),
        (['a', 'b', 'c', 'd'], ['b', 'c'], 1),
        ([1, 2, 1, 2, 3], [1, 2, 3], 2),
        ([1, 2, 3], [4], -1),
        ([1, 2, 3], [1, 2, 3, 4], -1),
        ([1, 2, 3], [1, 2, 3], 0),
        ([], [1], -1),
        ([1, 2, 3], [], -1),  # Defined behavior for empty sublist
        ([], [], -1),  # Defined behavior for empty lists
    ],
)
def test_find_subsequence(main_list, sub_list, expected_index):
    """Tests the find_subsequence helper function."""
    assert find_subsequence(main_list, sub_list) == expected_index


# Test BaseMDPEnv methods
def test_base_mdp_env_init_success(
    mock_dataset, mock_tokenizer, mock_reward_function
):
    """Tests successful initialization of BaseMDPEnv."""
    env = MockMDPEnv(
        dataset=mock_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=1,
        group_size=1,
        max_steps=1,
    )
    assert env.batch_size == 1
    assert env.tokenizer == mock_tokenizer
    assert len(env.reward_functions) == 1
    assert env.pad_token_id == 0


def test_base_mdp_env_init_tokenizer_setup(mock_dataset, mock_reward_function):
    """Tests tokenizer setup, including pad token fallback."""
    tokenizer_no_pad = MagicMock(spec=PreTrainedTokenizer)
    tokenizer_no_pad.padding_side = 'left'
    tokenizer_no_pad.pad_token = None  # No pad token initially
    tokenizer_no_pad.eos_token = '<eos>'
    tokenizer_no_pad.pad_token_id = None  # No pad token id initially
    tokenizer_no_pad.eos_token_id = 1
    # Mock apply_chat_template to simulate existence check
    tokenizer_no_pad.apply_chat_template = MagicMock(return_value='test')
    tokenizer_no_pad.chat_template = 'mock'  # Indicate a template exists

    env = MockMDPEnv(
        dataset=mock_dataset,
        tokenizer=tokenizer_no_pad,
        reward_functions=[mock_reward_function],
        batch_size=1,
        group_size=1,
        max_steps=1,
    )
    assert env.tokenizer.pad_token == '<eos>'  # Falls back to eos
    assert env.pad_token_id == 1  # Falls back to eos_token_id


def test_base_mdp_env_init_validation_errors(
    mock_dataset, mock_tokenizer, mock_reward_function, mock_reward_function_alt
):
    """Tests various validation errors during BaseMDPEnv initialization."""
    with pytest.raises(ValueError, match='world_size must be >= 1'):
        MockMDPEnv(
            mock_dataset,
            mock_tokenizer,
            [mock_reward_function],
            1,
            1,
            1,
            world_size=0,
        )
    with pytest.raises(ValueError, match='Rank must be less than world_size'):
        MockMDPEnv(
            mock_dataset,
            mock_tokenizer,
            [mock_reward_function],
            1,
            1,
            1,
            rank=1,
            world_size=1,
        )
    with pytest.raises(ValueError, match='Batch size must be >= 1'):
        MockMDPEnv(
            mock_dataset, mock_tokenizer, [mock_reward_function], 0, 1, 1
        )
    with pytest.raises(ValueError, match='Group size must be >= 1'):
        MockMDPEnv(
            mock_dataset, mock_tokenizer, [mock_reward_function], 1, 0, 1
        )
    with pytest.raises(ValueError, match='Max steps must be >= 1'):
        MockMDPEnv(
            mock_dataset, mock_tokenizer, [mock_reward_function], 1, 1, 0
        )
    with pytest.raises(ValueError, match='reward_functions cannot be empty'):
        MockMDPEnv(mock_dataset, mock_tokenizer, [], 1, 1, 1)
    with pytest.raises(
        ValueError, match='All reward_functions must be instances'
    ):
        MockMDPEnv(
            mock_dataset,
            mock_tokenizer,
            [mock_reward_function, lambda x: 0.0],
            1,
            1,
            1,
        )
    with pytest.raises(TypeError, match='dataset must be a datasets.Dataset'):
        MockMDPEnv(
            [{'messages': [], 'ground_truth': ''}],
            mock_tokenizer,
            [mock_reward_function],
            1,
            1,
            1,
        )
    with pytest.raises(
        ValueError, match="Dataset needs 'messages' and 'ground_truth' columns"
    ):
        bad_ds = Dataset.from_dict({'col1': [1], 'col2': [2]})
        MockMDPEnv(bad_ds, mock_tokenizer, [mock_reward_function], 1, 1, 1)
    with pytest.raises(
        ValueError, match="'messages' column should contain lists"
    ):
        bad_ds = Dataset.from_dict(
            {'messages': ['not a list'], 'ground_truth': ['gt']}
        )
        MockMDPEnv(bad_ds, mock_tokenizer, [mock_reward_function], 1, 1, 1)
    with pytest.raises(
        ValueError,
        match='Multiple reward functions provided without a reward_transform_fn',
    ):
        MockMDPEnv(
            mock_dataset,
            mock_tokenizer,
            [mock_reward_function, mock_reward_function_alt],
            1,
            1,
            1,
        )


def test_collate_fn(base_mdp_env, sample_raw_data):
    """Tests the _collate_fn."""
    collated = base_mdp_env._collate_fn(sample_raw_data)
    assert isinstance(collated, dict)
    assert 'messages' in collated
    assert 'ground_truth' in collated
    assert len(collated['messages']) == len(sample_raw_data)
    assert len(collated['ground_truth']) == len(sample_raw_data)
    assert collated['messages'][0] == sample_raw_data[0]['messages']
    assert collated['ground_truth'][1] == sample_raw_data[1]['ground_truth']


def test_collate_fn_empty(base_mdp_env):
    """Tests the _collate_fn with an empty list."""
    collated = base_mdp_env._collate_fn([])
    assert collated == {}


def test_prepare_initial_state(base_mdp_env, sample_raw_data):
    """Tests preparing the initial EnvState from a raw batch."""
    raw_batch = base_mdp_env._collate_fn(sample_raw_data)
    env_state = base_mdp_env._prepare_initial_state(raw_batch)

    assert isinstance(env_state, EnvState)
    expected_total_samples = len(sample_raw_data) * base_mdp_env.group_size
    assert len(env_state.sample_states) == expected_total_samples

    # Check first sample group (assuming group_size=1)
    sample_state_0 = env_state.sample_states[0]
    assert sample_state_0.ground_truth == sample_raw_data[0]['ground_truth']
    assert len(sample_state_0.messages) == len(sample_raw_data[0]['messages'])
    assert (
        sample_state_0.messages[0].content
        == sample_raw_data[0]['messages'][0]['content']
    )
    assert sample_state_0.init_msg_size == len(sample_raw_data[0]['messages'])
    assert sample_state_0.current_step == 0
    assert sample_state_0.done is False

    # Check second sample group (assuming group_size=1)
    sample_state_1 = env_state.sample_states[1]
    assert sample_state_1.ground_truth == sample_raw_data[1]['ground_truth']
    assert len(sample_state_1.messages) == len(sample_raw_data[1]['messages'])
    assert sample_state_1.init_msg_size == len(sample_raw_data[1]['messages'])


def test_prepare_initial_state_with_group_size(
    mock_dataset, mock_tokenizer, mock_reward_function, sample_raw_data
):
    """Tests preparing initial state with group_size > 1."""
    env = MockMDPEnv(
        dataset=mock_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function],
        batch_size=len(sample_raw_data),
        group_size=3,  # Test group size 3
        max_steps=1,
    )
    raw_batch = env._collate_fn(sample_raw_data)
    env_state = env._prepare_initial_state(raw_batch)

    assert isinstance(env_state, EnvState)
    expected_total_samples = len(sample_raw_data) * 3
    assert len(env_state.sample_states) == expected_total_samples

    # Check that the first 3 sample states correspond to the first raw sample
    for i in range(3):
        assert (
            env_state.sample_states[i].ground_truth
            == sample_raw_data[0]['ground_truth']
        )
        assert env_state.sample_states[i].init_msg_size == len(
            sample_raw_data[0]['messages']
        )
        # Ensure messages are copies, not the same object
        if i > 0:
            assert (
                env_state.sample_states[i].messages
                is not env_state.sample_states[0].messages
            )
            assert (
                env_state.sample_states[i].messages[0]
                is not env_state.sample_states[0].messages[0]
            )


def test_calculate_rewards_single_function(base_mdp_env, env_state_fixture):
    """Tests reward calculation with a single reward function."""

    terminal_rewards, rewards_dict = base_mdp_env._calculate_rewards(
        env_state_fixture.sample_states  # Pass list of SampleState
    )

    assert isinstance(terminal_rewards, torch.Tensor)
    # env_state_fixture has 1 sample by default
    assert terminal_rewards.shape == (len(env_state_fixture.sample_states),)
    assert terminal_rewards.tolist() == [1.0]  # From mock_reward_function
    assert 'mock_reward' in rewards_dict
    assert rewards_dict['mock_reward'] == [1.0]


def test_calculate_rewards_multiple_functions(
    mock_dataset,
    mock_tokenizer,
    mock_reward_function,
    mock_reward_function_alt,
    env_state_fixture,
):
    """Tests reward calculation with multiple functions and a transform."""

    def simple_sum_transform(r_dict: Dict[str, List[float]]) -> torch.Tensor:
        # Sum rewards for each sample
        num_samples = len(next(iter(r_dict.values())))
        sums = [
            sum(r_dict[key][i] for key in r_dict) for i in range(num_samples)
        ]
        return torch.tensor(sums, dtype=torch.float32)

    env = MockMDPEnv(
        dataset=mock_dataset,
        tokenizer=mock_tokenizer,
        reward_functions=[mock_reward_function, mock_reward_function_alt],
        batch_size=1,
        group_size=1,
        max_steps=1,
        reward_transform_fn=simple_sum_transform,
    )

    terminal_rewards, rewards_dict = env._calculate_rewards(
        env_state_fixture.sample_states
    )

    assert terminal_rewards.shape == (len(env_state_fixture.sample_states),)
    # mock_reward=1.0, mock_reward_alt=0.5 -> sum=1.5
    assert terminal_rewards.tolist() == [1.5]
    assert 'mock_reward' in rewards_dict
    assert 'mock_reward_alt' in rewards_dict
    assert rewards_dict['mock_reward'] == [1.0]
    assert rewards_dict['mock_reward_alt'] == [0.5]


def test_calculate_rewards_function_error(
    base_mdp_env, env_state_fixture, caplog
):
    """Tests that reward calculation handles errors in a reward function."""

    class ErrorReward(BaseRewardFunction):
        def __init__(self, name='error_reward'):
            super().__init__(name)

        def __call__(self, *args, **kwargs):
            raise ValueError('Simulated error')

    base_mdp_env.reward_functions = [ErrorReward()]
    with pytest.raises(ValueError, match='Simulated error'):
        terminal_rewards, rewards_dict = base_mdp_env._calculate_rewards(
            env_state_fixture.sample_states
        )


def test_transform_rewards_fallback_on_error(base_mdp_env, caplog):
    """Tests that reward transformation falls back to the first reward if the transform function fails."""
    rewards_dict = {'reward1': [1.0, 2.0], 'reward2': [0.5, 0.6]}

    def error_transform(r_dict):
        raise ValueError('Transform error')

    base_mdp_env.reward_transform_fn = error_transform

    transformed = base_mdp_env._transform_rewards(rewards_dict)

    assert 'Reward transformation failed' in caplog.text
    assert "Falling back to using reward 'reward1'" in caplog.text
    assert torch.equal(
        transformed, torch.tensor([1.0, 2.0], dtype=torch.float32)
    )


def test_convert_to_batch_prompts(base_mdp_env, mock_tokenizer):
    """Tests converting batch messages to prompts using the chat template."""

    env_state = EnvState(
        sample_states=[
            SampleState(
                messages=[ChatMessage(role='user', content='Hello')],
                ground_truth='test',
                init_msg_size=1,
            ),
            SampleState(
                messages=[
                    ChatMessage(role='user', content='How are you'),
                    ChatMessage(role='assistant', content='I am fine'),
                ],
                ground_truth='test',
                init_msg_size=2,
            ),
        ]
    )
    # Expected dictionaries after model_dump()
    expected_msg_1 = [
        {
            'role': 'user',
            'content': 'Hello',
        }
    ]
    expected_msg_2 = [
        {
            'role': 'user',
            'content': 'How are you',
        },
        {
            'role': 'assistant',
            'content': 'I am fine',
        },
    ]

    prompts = base_mdp_env._convert_to_batch_prompts(env_state)

    assert len(prompts) == 2
    # Check that apply_chat_template was called correctly for each message list
    mock_tokenizer.apply_chat_template.assert_any_call(
        expected_msg_1,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
    mock_tokenizer.apply_chat_template.assert_any_call(
        expected_msg_2,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    assert mock_tokenizer.apply_chat_template.call_count >= 2


# --- Test Cases for _convert_to_episodes ---


def test_convert_to_episodes_empty_env_state(base_mdp_env):
    """Test with an EnvState containing no sample states."""
    empty_env_state = EnvState(sample_states=[])
    episodes = base_mdp_env._convert_to_episodes(empty_env_state)
    assert episodes == []


def test_convert_to_episodes_reward_mismatch(base_mdp_env):
    """Test scenario where reward calculation yields mismatched number of rewards."""
    sample_state = SampleState(
        messages=[
            ChatMessage(role='user', content='Q'),
            ChatMessage(role='assistant', content='A'),
        ],
        ground_truth='GT',
        init_msg_size=1,
        done=True,
    )
    env_state = EnvState(sample_states=[sample_state])

    # Mock _calculate_rewards to return an empty tensor for rewards
    with patch.object(
        base_mdp_env, '_calculate_rewards', return_value=(torch.empty(0), {})
    ):
        episodes = base_mdp_env._convert_to_episodes(env_state)

    assert episodes == []


def test_convert_to_episodes_simple_case(base_mdp_env):
    """Test a single sample with a simple user-assistant interaction."""
    tokenizer = base_mdp_env.tokenizer
    ID_U, ID_A = (
        tokenizer.role_map_tok_ids['user'],
        tokenizer.role_map_tok_ids['assistant'],
    )
    ID_Q, ID_R = tokenizer._get_word_id('Q'), tokenizer._get_word_id('R')
    ID_EOS = tokenizer.eos_token_id

    sample_messages = [
        ChatMessage(role='user', content='Q'),
        ChatMessage(
            role='assistant', content='R'
        ),  # This is the generated part
    ]
    sample_state = SampleState(
        messages=sample_messages, ground_truth='GT', init_msg_size=1, done=True
    )
    env_state = EnvState(sample_states=[sample_state])

    # Expected tokenization:
    # User: Q <eos> Assistant: R <eos_programmatic>
    # [ID_U, ID_Q, ID_EOS, ID_A, ID_R, ID_EOS]
    expected_full_ids = [ID_U, ID_Q, ID_EOS, ID_A, ID_R, ID_EOS]
    # states: [ID_U, ID_Q, ID_EOS, ID_A, ID_R]
    # actions: [ID_Q, ID_EOS, ID_A, ID_R, ID_EOS]
    # loss_mask (shifted): [0,0,0, 1 (for R), 1 (for final EOS)] (length 5)
    # Original loss_mask for full_sequence_ids: [0,0,0, 1(R),0, 1(final EOS)] - error in manual trace, content R is masked, then final EOS
    # Let's trace masking:
    # msg_idx=0 (user Q), init_msg_size=1. msg_idx != init_msg_size-1. Not assistant.
    #   prompt_token_len not set yet. current_pos updated.
    # msg_idx=1 (assistant R), init_msg_size=1. msg_idx >= init_msg_size. Is assistant.
    #   content="R", content_tokens=[ID_R]
    #   prefix_tokens for [U,A] = [ID_U, ID_Q, ID_EOS, ID_A, ID_R, ID_EOS] (cfm=F for apply_chat_template(tokenize=T))
    #   current_pos (after U msg) = 3 ([ID_U, ID_Q, ID_EOS])
    #   message_tokens (for A msg) = [ID_A, ID_R, ID_EOS]
    #   find_subsequence([ID_A,ID_R,ID_EOS], [ID_R]) -> returns 1 (index of ID_R in message_tokens)
    #   global_content_start = current_pos + 1 = 3 + 1 = 4. (index of ID_R in full_sequence_ids)
    #   loss_mask[4] = 1.
    # prompt_token_len is set when msg_idx == init_msg_size - 1 (0 == 0).
    #   prefix_tokens for [U] = [ID_U, ID_Q, ID_EOS]. prompt_token_len = 3.
    # Final loss_mask for full_sequence_ids: [0,0,0,0,1,0] initially. Then loss_mask[4]=1. Then loss_mask[-1]=1.
    # So, [0,0,0,0,1,1]. Shifted: [0,0,0,1,1]

    expected_states = torch.tensor(
        [ID_U, ID_Q, ID_EOS, ID_A, ID_R], dtype=torch.long
    )
    expected_actions = torch.tensor(
        [ID_Q, ID_EOS, ID_A, ID_R, ID_EOS], dtype=torch.long
    )
    expected_loss_mask = torch.tensor(
        [False, False, False, True, True], dtype=torch.bool
    )  # R and final EOS
    expected_prompt_len = 3  # [ID_U, ID_Q, ID_EOS]
    expected_completion_len = 2  # R, EOS

    episodes = base_mdp_env._convert_to_episodes(env_state)
    assert len(episodes) == 1
    ep = episodes[0]

    assert torch.equal(ep.states, expected_states)
    assert torch.equal(ep.actions, expected_actions)
    assert torch.equal(ep.loss_mask, expected_loss_mask)
    assert ep.prompt_length == expected_prompt_len
    assert ep.completion_length == expected_completion_len
    assert ep.terminal_reward == 1.0  # from mock_reward_function
    assert ep.reward_dict == {'mock_reward': 1.0}
    assert ep.ground_truth == 'GT'
    assert len(ep.chat_history) == 2


def test_convert_to_episodes_messages_too_short(base_mdp_env):
    """Test scenario where messages is too short."""

    sample_state = SampleState(
        messages=[
            ChatMessage(role='user', content='Q'),
        ],
        ground_truth='GT',
        init_msg_size=1,
        done=True,
    )
    env_state = EnvState(sample_states=[sample_state])

    with pytest.raises(
        RuntimeError, match='Sample resulted in messages length < 2'
    ):
        base_mdp_env._convert_to_episodes(env_state)


def test_convert_to_episodes_content_not_found(base_mdp_env):
    """Test when assistant content tokens cannot be found in message tokens."""
    tokenizer = base_mdp_env.tokenizer

    # Normal apply_chat_template behavior
    original_apply_chat_template = tokenizer.apply_chat_template

    def faulty_encode_for_content(text, add_special_tokens=False):
        if text == 'R':  # The assistant's content
            return [tokenizer._get_word_id('NonExistentToken')]
        # Fallback to original encode logic for other calls (like chat history string)
        # This is tricky, better to control find_subsequence directly or ensure mock is precise
        # For simplicity, let's assume the main encode for history works, but content encode is faulty

        # Simplified: just make content tokens for "R" different from what apply_chat_template would produce
        # The consistent_mock_tokenizer should already make this work if inputs are normal.
        # To force failure, we can patch find_subsequence.
        words = text.split()
        tokens = []
        for word in words:
            if word == tokenizer.eos_token:
                tokens.append(tokenizer.eos_token_id)
            elif word in tokenizer.text_to_role_id:
                tokens.append(tokenizer.text_to_role_id[word])
            else:
                tokens.append(tokenizer._get_word_id(word))
        return tokens

    tokenizer.encode = MagicMock(side_effect=faulty_encode_for_content)
    # This setup is a bit fragile. A more direct way:
    # Patch `find_subsequence` to return -1 for the specific call.

    sample_messages = [
        ChatMessage(role='user', content='Q'),
        ChatMessage(role='assistant', content='R'),
    ]
    sample_state = SampleState(
        messages=sample_messages, ground_truth='GT', init_msg_size=1, done=True
    )
    env_state = EnvState(sample_states=[sample_state])

    with patch(
        'rl4llm.core.base_env.find_subsequence', return_value=-1
    ) as mock_find_sub:
        with pytest.raises(
            RuntimeError,
            match='Could not precisely locate content tokens for assistant msg 1.',
        ):
            base_mdp_env._convert_to_episodes(env_state)
        # Ensure find_subsequence was actually called for the assistant message content
        # Expected content tokens for "R" would be [ID_R]
        # Expected message_tokens would be [ID_A, ID_R, ID_EOS]
        # mock_find_sub.assert_any_call(ANY, [tokenizer._get_word_id("R")]) # Check it was called with content tokens of "R"


def test_convert_to_episodes_masking_and_prompt_len_detailed(base_mdp_env):
    """Test more complex masking with system message and multiple turns."""
    tokenizer = base_mdp_env.tokenizer
    ID_S, ID_U, ID_A = (
        tokenizer.role_map_tok_ids['system'],
        tokenizer.role_map_tok_ids['user'],
        tokenizer.role_map_tok_ids['assistant'],
    )
    ID_CTX = tokenizer._get_word_id('context')
    ID_Q1 = tokenizer._get_word_id('Q1')
    ID_R1 = tokenizer._get_word_id('R1')  # Part of prompt
    ID_Q2 = tokenizer._get_word_id('Q2')
    ID_R2 = tokenizer._get_word_id('R2')  # Generated
    ID_EOS = tokenizer.eos_token_id

    # Messages: Sys, User, Asst (prompt), User, Asst (generated)
    # init_msg_size = 3 (Sys, User, Asst are the initial prompt)
    sample_messages = [
        ChatMessage(role='system', content='context'),  # Prompt
        ChatMessage(role='user', content='Q1'),  # Prompt
        ChatMessage(role='assistant', content='R1'),  # Prompt
        ChatMessage(role='user', content='Q2'),  # New User Query
        ChatMessage(role='assistant', content='R2'),  # Generated Response
    ]
    sample_state = SampleState(
        messages=sample_messages, ground_truth='GT', init_msg_size=3, done=True
    )
    env_state = EnvState(sample_states=[sample_state])

    # Expected full sequence:
    # S:context<E> U:Q1<E> A:R1<E> U:Q2<E> A:R2<E_prog>
    # [S, CTX, E,  U, Q1, E,  A, R1, E,  U, Q2, E,  A, R2, E]
    # Token IDs:
    # [12,100,1,  10,101,1,  11,102,1,  10,103,1,  11,104,1] (total 15 tokens)

    # Prompt part (init_msg_size=3): S:context<E> U:Q1<E> A:R1<E>
    # Tokens: [12,100,1, 10,101,1, 11,102,1] -> length 9. So, prompt_token_len = 9.

    # Loss mask for full sequence (before shift):
    # Indices:0  1   2   3  4   5   6  7   8   9 10  11  12 13  14
    # Tokens: S CTX  E   U  Q1  E   A  R1  E   U  Q2  E   A  R2  E_prog
    # Mask:   0  0   0   0  0   0   0  0   0   0  0   0   0  1   1  (R2 and final EOS)
    # (Because R1 is part of prompt, R2 is generated as msg_idx=4 >= init_msg_size=3)

    # Shifted loss_mask (length 14):
    # [0,0,0,0,0,0,0,0,0,0,0,0,1,1]

    expected_prompt_len = 9
    expected_completion_len = 2  # R2, EOS_prog
    # Last two elements of loss_mask should be True

    episodes = base_mdp_env._convert_to_episodes(env_state)
    assert len(episodes) == 1
    ep = episodes[0]

    assert ep.prompt_length == expected_prompt_len
    assert ep.completion_length == expected_completion_len

    # Check loss mask: only last two tokens (R2, final_EOS) should be True
    # ep.loss_mask is for actions (length N-1)
    # Expected: [F,F,F,F,F,F,F,F,F,F,F,F, T (R2), T (EOS_prog)]
    assert not torch.any(ep.loss_mask[:-2])  # All False until the last two
    assert torch.all(ep.loss_mask[-2:])  # Last two are True
    assert ep.loss_mask.sum().item() == expected_completion_len

    # Verify states and actions shapes
    assert ep.states.shape[0] == 14
    assert ep.actions.shape[0] == 14
    assert ep.loss_mask.shape[0] == 14


def test_convert_to_episodes_init_msg_size_zero(base_mdp_env):
    """Test when init_msg_size is 0, all assistant messages are masked."""
    tokenizer = base_mdp_env.tokenizer
    ID_U, ID_A = (
        tokenizer.role_map_tok_ids['user'],
        tokenizer.role_map_tok_ids['assistant'],
    )
    ID_Q = tokenizer._get_word_id('Q')
    ID_R = tokenizer._get_word_id('R')
    ID_EOS = tokenizer.eos_token_id

    sample_messages = [
        ChatMessage(role='user', content='Q'),
        ChatMessage(role='assistant', content='R'),
    ]
    # init_msg_size = 0 means the initial prompt was empty or not from these messages.
    # All assistant messages here are considered "generated".
    sample_state = SampleState(
        messages=sample_messages, ground_truth='GT', init_msg_size=0, done=True
    )
    env_state = EnvState(sample_states=[sample_state])

    # Expected full sequence: U:Q<E> A:R<E_prog>
    # Tokens: [ID_U, ID_Q, ID_EOS, ID_A, ID_R, ID_EOS] (len 6)
    # prompt_token_len should be 0 (as init_msg_size is 0)

    # Masking: msg_idx=0 (user Q). Not assistant.
    # msg_idx=1 (assistant R). msg_idx=1 >= init_msg_size=0. Is assistant.
    #   Content "R" is masked. global_content_start = index of R.
    #   Index of R is 4. loss_mask[4]=1.
    # Final EOS is masked: loss_mask[5]=1.
    # Original loss_mask: [0,0,0,0,1,1]
    # Shifted loss_mask: [0,0,0,1,1] (length 5)

    expected_prompt_len = 0
    expected_completion_len = 2  # R, EOS_prog

    episodes = base_mdp_env._convert_to_episodes(env_state)
    assert len(episodes) == 1
    ep = episodes[0]

    assert ep.prompt_length == expected_prompt_len
    assert ep.completion_length == expected_completion_len
    # Shifted loss mask: [F,F,F, T(R), T(EOS)]
    assert torch.equal(
        ep.loss_mask,
        torch.tensor([False, False, False, True, True], dtype=torch.bool),
    )


def test_convert_to_episodes_assistant_empty_content(base_mdp_env):
    """Test when an assistant message has empty content."""
    tokenizer = base_mdp_env.tokenizer
    ID_U, ID_A = (
        tokenizer.role_map_tok_ids['user'],
        tokenizer.role_map_tok_ids['assistant'],
    )
    ID_Q = tokenizer._get_word_id('Q')
    ID_EOS = tokenizer.eos_token_id

    sample_messages = [
        ChatMessage(role='user', content='Q'),
        ChatMessage(role='assistant', content=''),  # Empty content
    ]
    sample_state = SampleState(
        messages=sample_messages, ground_truth='GT', init_msg_size=1, done=True
    )
    env_state = EnvState(sample_states=[sample_state])

    # Expected full sequence: U:Q<E> A:<E_prog> (no content for A)
    # Tokens: [ID_U, ID_Q, ID_EOS, ID_A, ID_EOS] (len 5)
    # prompt_token_len for U:Q<E> is 3.

    # Masking: msg_idx=1 (assistant, empty content). msg_idx=1 >= init_msg_size=1.
    #   content="", content_tokens=[] (or depends on tokenizer, mock encode gives [])
    #   The `if content_tokens:` block is skipped. No content masking.
    # Final EOS is masked: loss_mask[4]=1.
    # Original loss_mask: [0,0,0,0,1]
    # Shifted loss_mask: [0,0,0,1] (length 4)

    expected_prompt_len = 3
    expected_completion_len = 1  # Only final EOS

    episodes = base_mdp_env._convert_to_episodes(env_state)
    assert len(episodes) == 1
    ep = episodes[0]

    assert ep.prompt_length == expected_prompt_len
    assert ep.completion_length == expected_completion_len
    # Shifted loss mask: [F,F,F, T(EOS)]
    assert torch.equal(
        ep.loss_mask,
        torch.tensor([False, False, False, True], dtype=torch.bool),
    )
    assert ep.states.shape[0] == 4  # [ID_U, ID_Q, ID_EOS, ID_A]
    assert ep.actions.shape[0] == 4  # [ID_Q, ID_EOS, ID_A, ID_EOS]


# Test Rollout Flow (requires MockMDPEnv)
@patch.object(MockMDPEnv, '_reset')
@patch.object(MockMDPEnv, '_run_interaction_loop')
@patch.object(MockMDPEnv, '_convert_to_episodes')
def test_rollout_flow(
    mock_convert, mock_interact, mock_reset, base_mdp_env, sample_state_fixture
):
    """Tests the basic flow of the rollout method."""
    # Setup mocks using valid SampleState
    initial_state = EnvState(
        sample_states=[sample_state_fixture.model_copy(deep=True)]
    )
    # Create a different final state for clarity
    final_state_sample = sample_state_fixture.model_copy(deep=True)
    final_state_sample.messages.append(
        ChatMessage(role='assistant', content='Final response')
    )
    final_state = EnvState(sample_states=[final_state_sample])
    expected_episodes = [
        MagicMock(spec=EpisodeData)
    ]  # Mocking the output is fine

    mock_reset.return_value = initial_state
    mock_interact.return_value = final_state
    mock_convert.return_value = expected_episodes

    # Call rollout
    llm_mock = MagicMock()
    sampling_params = {'max_new_tokens': 10}
    episodes = base_mdp_env.rollout(
        llm=llm_mock, sampling_params=sampling_params, custom_arg='test'
    )

    # Assertions
    mock_reset.assert_called_once()
    mock_interact.assert_called_once_with(
        initial_state, llm_mock, sampling_params, custom_arg='test'
    )
    mock_convert.assert_called_once_with(final_state)
    assert episodes == expected_episodes


@patch.object(MockMDPEnv, '_reset')
def test_rollout_handles_reset_none(mock_reset, base_mdp_env, caplog):
    """Tests that rollout returns an empty list if _reset returns None."""
    mock_reset.return_value = None
    llm_mock = MagicMock()
    sampling_params = {'max_new_tokens': 10}

    episodes = base_mdp_env.rollout(
        llm=llm_mock, sampling_params=sampling_params
    )

    assert episodes == []
    assert f"Rank {base_mdp_env.rank}: Reset returned None" in caplog.text


@patch.object(MockMDPEnv, '_reset')
@patch.object(MockMDPEnv, '_run_interaction_loop')
def test_rollout_handles_interaction_error(
    mock_interact, mock_reset, base_mdp_env, caplog, sample_state_fixture
):
    """Tests that rollout returns an empty list if _run_interaction_loop fails."""
    # Use a valid SampleState for the initial state
    initial_state = EnvState(
        sample_states=[sample_state_fixture.model_copy(deep=True)]
    )
    mock_reset.return_value = initial_state
    mock_interact.side_effect = ValueError('Interaction loop error')

    llm_mock = MagicMock()
    sampling_params = {'max_new_tokens': 10}

    episodes = base_mdp_env.rollout(
        llm=llm_mock, sampling_params=sampling_params
    )

    assert episodes == []
    assert (
        f"Rank {base_mdp_env.rank}: Error during _run_interaction_loop"
        in caplog.text
    )
    assert 'Interaction loop error' in caplog.text


@patch.object(MockMDPEnv, '_reset')
@patch.object(MockMDPEnv, '_run_interaction_loop')
@patch.object(MockMDPEnv, '_convert_to_episodes')
def test_rollout_handles_conversion_error(
    mock_convert,
    mock_interact,
    mock_reset,
    base_mdp_env,
    caplog,
    sample_state_fixture,
):
    """Tests that rollout returns an empty list if _convert_to_episodes fails."""
    # Use valid SampleStates
    initial_state = EnvState(
        sample_states=[sample_state_fixture.model_copy(deep=True)]
    )
    final_state_sample = sample_state_fixture.model_copy(deep=True)
    final_state_sample.messages.append(
        ChatMessage(role='assistant', content='Final response')
    )
    final_state = EnvState(sample_states=[final_state_sample])

    mock_reset.return_value = initial_state
    mock_interact.return_value = final_state
    mock_convert.side_effect = ValueError('Conversion error')

    llm_mock = MagicMock()
    sampling_params = {'max_new_tokens': 10}

    episodes = base_mdp_env.rollout(
        llm=llm_mock, sampling_params=sampling_params
    )

    assert episodes == []
    assert (
        f"Rank {base_mdp_env.rank}: Error during _convert_to_episodes"
        in caplog.text
    )
    assert 'Conversion error' in caplog.text
