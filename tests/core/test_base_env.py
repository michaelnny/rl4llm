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


@pytest.fixture
def mock_tokenizer():
    """Provides a mock PreTrainedTokenizer."""
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '<eos>'
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    # Store the word map globally within the mock instance for consistency
    tokenizer.word_map = {}
    tokenizer.next_word_id = 100

    def _get_word_id(word):
        if word not in tokenizer.word_map:
            tokenizer.word_map[word] = tokenizer.next_word_id
            tokenizer.next_word_id += 1
        return tokenizer.word_map[word]

    def _mock_apply_chat_template_v2(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        tokens = []
        full_text = ''
        role_map = {'user': 10, 'assistant': 11, 'system': 12, 'tool': 13}
        # Reset map for each call to simulate fresh tokenization *if needed*
        # For these tests, let's keep the map persistent across calls within a single test setup
        # unless explicitly reset by the test.

        for i, msg in enumerate(messages):
            # Use .get() for safer dictionary access if msg is a dict
            role = msg.get('role') if isinstance(msg, dict) else msg.role
            content = (
                msg.get('content') if isinstance(msg, dict) else msg.content
            )

            role_token = role_map.get(role, 99)
            tokens.append(role_token)
            full_text += f"{role}: {content}\n"
            content_tokens = []
            if content:  # Handle potential None content
                for word in content.split():
                    content_tokens.append(_get_word_id(word))
            tokens.extend(content_tokens)
            tokens.append(tokenizer.eos_token_id)  # Add EOS after each message

        if add_generation_prompt:
            tokens.append(role_map['assistant'])
            full_text += 'assistant:\n'

        if tokenize:
            return tokens
        else:
            return full_text

    def _mock_encode_v2(text, add_special_tokens=False):
        tokens = []
        if text:  # Handle potential None text
            for word in text.split():
                tokens.append(_get_word_id(word))
        return tokens

    tokenizer.apply_chat_template = MagicMock(
        side_effect=_mock_apply_chat_template_v2
    )
    tokenizer.encode = MagicMock(side_effect=_mock_encode_v2)

    # Needed for _setup_tokenizer check
    tokenizer.chat_template = 'mock_template'  # Indicate a template exists

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
    # Check default optional fields
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


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

    base_mdp_env.reward_functions = [ErrorReward()]  # Replace reward function
    terminal_rewards, rewards_dict = base_mdp_env._calculate_rewards(
        env_state_fixture.sample_states  # Pass list of SampleState
    )

    # env_state_fixture.sample_states[0].id will be part of the log message
    sample_id = env_state_fixture.sample_states[0].id
    assert (
        f"Reward func 'error_reward' for sample ID {sample_id} failed"
        in caplog.text
    )
    assert (
        'Simulated error' in caplog.text
    )  # Check for the specific error message
    assert terminal_rewards.tolist() == [0.0]  # Default reward on error
    assert rewards_dict['error_reward'] == [0.0]


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
            'tool_calls': None,
            'tool_call_id': None,
        }
    ]
    expected_msg_2 = [
        {
            'role': 'user',
            'content': 'How are you',
            'tool_calls': None,
            'tool_call_id': None,
        },
        {
            'role': 'assistant',
            'content': 'I am fine',
            'tool_calls': None,
            'tool_call_id': None,
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
    assert mock_tokenizer.apply_chat_template.call_count >= 2  # Keep this check


# --- Tests for _convert_to_episodes (High Priority) ---


def test_convert_to_episodes_simple_case(base_mdp_env, mock_tokenizer):
    """Tests converting a simple user-assistant interaction."""
    # Reset word map for this specific test
    mock_tokenizer.word_map = {'Hello': 100, 'Hi': 101, 'there': 102}
    mock_tokenizer.next_word_id = 103

    # Define expected token sequences based on the mock logic
    prompt_tokens = [10, 100, 1]  # U: Hello EOS
    full_tokens = [10, 100, 1, 11, 101, 102, 1]  # U: Hello EOS A: Hi there EOS
    content_tokens = [101, 102]  # Hi there

    # Configure side effects for apply_chat_template based on expected calls
    mock_tokenizer.apply_chat_template.side_effect = [
        full_tokens,  # Initial call for full sequence
        prompt_tokens,  # Loop call msg_idx=0 (prompt)
        full_tokens,  # Loop call msg_idx=1 (full again)
    ]
    # Configure side effect for encode (only called for assistant message content)
    mock_tokenizer.encode.side_effect = [
        content_tokens  # Tokens for "Hi there"
    ]

    sample_state = SampleState(
        messages=[
            ChatMessage(role='user', content='Hello'),
            ChatMessage(role='assistant', content='Hi there'),
        ],
        ground_truth='Greeting',
        init_msg_size=1,  # User message is the prompt
        current_step=1,
        done=True,
    )
    env_state = EnvState(sample_states=[sample_state])

    episodes = base_mdp_env._convert_to_episodes(env_state)

    assert len(episodes) == 1
    ep = episodes[0]

    expected_full_sequence = torch.tensor(full_tokens, dtype=torch.long)
    expected_states = expected_full_sequence[:-1]
    expected_actions = expected_full_sequence[1:]
    # Masking logic derived in thought process:
    # loss_mask = [0, 0, 0, 0, 1, 1, 0]
    # final_loss_mask = loss_mask[1:] = [0, 0, 0, 1, 1, 0]
    expected_loss_mask = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)

    assert torch.equal(ep.states, expected_states)
    assert torch.equal(ep.actions, expected_actions)
    assert torch.equal(ep.loss_mask, expected_loss_mask)
    assert ep.prompt_length == len(prompt_tokens)  # Should be 3
    assert ep.completion_length == len(content_tokens)  # Should be 2
    assert ep.terminal_reward == 1.0  # From mock reward
    assert ep.ground_truth == 'Greeting'
    assert ep.chat_history == sample_state.messages


def test_convert_to_episodes_multi_turn(base_mdp_env, mock_tokenizer):
    """Tests converting a multi-turn conversation with masking only the last assistant turn."""
    # Reset word map
    mock_tokenizer.word_map = {'Q1': 100, 'A1': 101, 'Q2': 102, 'A2': 103}
    mock_tokenizer.next_word_id = 104

    # Define expected token sequences
    msg0_tokens = [10, 100, 1]  # U: Q1 EOS
    msg1_tokens = [10, 100, 1, 11, 101, 1]  # U: Q1 EOS A: A1 EOS
    msg2_tokens = [10, 100, 1, 11, 101, 1, 10, 102, 1]  # ... U: Q2 EOS (prompt)
    msg3_tokens = [
        10,
        100,
        1,
        11,
        101,
        1,
        10,
        102,
        1,
        11,
        103,
        1,
    ]  # ... A: A2 EOS (full)
    content_a1_tokens = [101]  # A1
    content_a2_tokens = [103]  # A2

    # Configure side effects for apply_chat_template
    mock_tokenizer.apply_chat_template.side_effect = [
        msg3_tokens,  # Initial call for full sequence
        msg0_tokens,  # Loop msg_idx=0
        msg1_tokens,  # Loop msg_idx=1
        msg2_tokens,  # Loop msg_idx=2 (prompt)
        msg3_tokens,  # Loop msg_idx=3 (full again)
    ]
    # Configure side effect for encode (only called for assistant messages >= init_msg_size)
    mock_tokenizer.encode.side_effect = [
        content_a2_tokens,  # Only A2 content is encoded for masking
    ]

    sample_state = SampleState(
        messages=[
            ChatMessage(role='user', content='Q1'),
            ChatMessage(role='assistant', content='A1'),
            ChatMessage(role='user', content='Q2'),  # Prompt ends here
            ChatMessage(
                role='assistant', content='A2'
            ),  # This should be masked
        ],
        ground_truth='Multi-turn GT',
        init_msg_size=3,  # U:Q1, A:A1, U:Q2 form the prompt
        current_step=1,  # Assume one generation step produced A2
        done=True,
    )
    env_state = EnvState(sample_states=[sample_state])

    episodes = base_mdp_env._convert_to_episodes(env_state)

    assert len(episodes) == 1
    ep = episodes[0]

    expected_full_sequence = torch.tensor(msg3_tokens, dtype=torch.long)
    expected_states = expected_full_sequence[:-1]
    expected_actions = expected_full_sequence[1:]
    # Masking logic derived in thought process:
    # loss_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # final_loss_mask = loss_mask[1:] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    expected_loss_mask = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool
    )

    assert torch.equal(ep.states, expected_states)
    assert torch.equal(ep.actions, expected_actions)
    assert torch.equal(ep.loss_mask, expected_loss_mask)
    assert ep.prompt_length == len(msg2_tokens)  # Should be 9
    assert ep.completion_length == len(content_a2_tokens)  # Should be 1
    assert ep.terminal_reward == 1.0
    assert ep.ground_truth == 'Multi-turn GT'


def test_convert_to_episodes_no_assistant_response_after_prompt(
    base_mdp_env, mock_tokenizer
):
    """Tests conversion when only prompt messages exist (no generation happened)."""
    # Reset word map
    mock_tokenizer.word_map = {'Hello': 100}
    mock_tokenizer.next_word_id = 101

    # Define expected token sequences
    prompt_tokens = [10, 100, 1]  # U: Hello EOS

    # Configure side effects for apply_chat_template
    mock_tokenizer.apply_chat_template.side_effect = [
        prompt_tokens,  # Initial call for full sequence (which is just the prompt)
        prompt_tokens,  # Loop call msg_idx=0 (prompt)
    ]
    # Configure side effect for encode (should not be called)
    mock_tokenizer.encode.side_effect = []

    sample_state = SampleState(
        messages=[ChatMessage(role='user', content='Hello')],
        ground_truth='No response GT',
        init_msg_size=1,
        current_step=0,  # No steps taken
        done=False,  # Not technically done, but represents state before generation
    )
    env_state = EnvState(sample_states=[sample_state])

    episodes = base_mdp_env._convert_to_episodes(env_state)

    assert len(episodes) == 1
    ep = episodes[0]

    expected_full_sequence = torch.tensor(prompt_tokens, dtype=torch.long)
    expected_states = expected_full_sequence[:-1]  # [10, 100]
    expected_actions = expected_full_sequence[1:]  # [100, 1]
    # Masking logic derived in thought process:
    # loss_mask = [0, 0, 0]
    # final_loss_mask = loss_mask[1:] = [0, 0]
    expected_loss_mask = torch.tensor([0, 0], dtype=torch.bool)

    assert torch.equal(ep.states, expected_states)
    assert torch.equal(ep.actions, expected_actions)
    assert torch.equal(ep.loss_mask, expected_loss_mask)
    assert ep.prompt_length == len(prompt_tokens)  # Should be 3
    assert ep.completion_length == 0  # No assistant tokens masked
    assert ep.terminal_reward == 1.0
    assert ep.ground_truth == 'No response GT'


def test_convert_to_episodes_batch(base_mdp_env, mock_tokenizer):
    """Tests converting a batch of samples."""
    # Reset word map
    mock_tokenizer.word_map = {
        'Hi': 100,
        'Hey': 101,
        'Bye': 102,
        'See': 103,
        'ya': 104,
    }
    mock_tokenizer.next_word_id = 105

    # Define expected token sequences for Sample 0
    s0_prompt_tokens = [10, 100, 1]  # U: Hi EOS
    s0_full_tokens = [10, 100, 1, 11, 101, 1]  # U: Hi EOS A: Hey EOS
    s0_content_tokens = [101]  # Hey

    # Define expected token sequences for Sample 1
    s1_prompt_tokens = [10, 102, 1]  # U: Bye EOS
    s1_full_tokens = [10, 102, 1, 11, 103, 104, 1]  # U: Bye EOS A: See ya EOS
    s1_content_tokens = [103, 104]  # See ya

    # Configure side effects for apply_chat_template (interleaved calls)
    mock_tokenizer.apply_chat_template.side_effect = [
        # --- Sample 0 ---
        s0_full_tokens,  # Initial call for full sequence
        s0_prompt_tokens,  # Loop msg_idx=0 (prompt)
        s0_full_tokens,  # Loop msg_idx=1 (full again)
        # --- Sample 1 ---
        s1_full_tokens,  # Initial call for full sequence
        s1_prompt_tokens,  # Loop msg_idx=0 (prompt)
        s1_full_tokens,  # Loop msg_idx=1 (full again)
    ]
    # Configure side effect for encode
    mock_tokenizer.encode.side_effect = [
        s0_content_tokens,  # Sample 0, content "Hey"
        s1_content_tokens,  # Sample 1, content "See ya"
    ]

    sample_state_1 = SampleState(
        messages=[
            ChatMessage(role='user', content='Hi'),
            ChatMessage(role='assistant', content='Hey'),
        ],
        ground_truth='GT1',
        init_msg_size=1,
        done=True,
    )
    sample_state_2 = SampleState(
        messages=[
            ChatMessage(role='user', content='Bye'),
            ChatMessage(role='assistant', content='See ya'),
        ],
        ground_truth='GT2',
        init_msg_size=1,
        done=True,
    )
    env_state = EnvState(sample_states=[sample_state_1, sample_state_2])

    episodes = base_mdp_env._convert_to_episodes(env_state)

    assert len(episodes) == 2

    # Check episode 1
    ep1 = episodes[0]
    assert ep1.prompt_length == len(s0_prompt_tokens)  # 3
    assert ep1.completion_length == len(s0_content_tokens)  # 1
    # Trace: full=[10, 100, 1, 11, 101, 1], loss_mask=[0,0,0,0,1,0], final=[0,0,0,1,0]
    expected_loss_mask_1 = torch.tensor([0, 0, 0, 1, 0], dtype=torch.bool)
    assert torch.equal(ep1.loss_mask, expected_loss_mask_1)
    assert ep1.ground_truth == 'GT1'

    # Check episode 2
    ep2 = episodes[1]
    assert ep2.prompt_length == len(s1_prompt_tokens)  # 3
    assert ep2.completion_length == len(s1_content_tokens)  # 2
    # Trace: full=[10, 102, 1, 11, 103, 104, 1], loss_mask=[0,0,0,0,1,1,0], final=[0,0,0,1,1,0]
    expected_loss_mask_2 = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
    assert torch.equal(ep2.loss_mask, expected_loss_mask_2)
    assert ep2.ground_truth == 'GT2'


def test_convert_to_episodes_error_handling(
    base_mdp_env, mock_tokenizer, caplog
):
    """Tests that conversion skips samples where tokenization fails."""
    # Reset word map
    mock_tokenizer.word_map = {'OK': 100}
    mock_tokenizer.next_word_id = 101

    sample_state_ok = SampleState(
        messages=[ChatMessage(role='user', content='OK')],
        ground_truth='GT_OK',
        init_msg_size=1,
        done=True,
    )
    # Create a unique ID for the bad state to check log message
    bad_state_id = str(uuid.uuid4().hex)
    sample_state_bad = SampleState(
        id=bad_state_id,
        messages=[ChatMessage(role='user', content='BAD')],
        ground_truth='GT_BAD',
        init_msg_size=1,
        done=True,
    )
    env_state = EnvState(sample_states=[sample_state_ok, sample_state_bad])

    # Make apply_chat_template fail on the second sample's initial call
    ok_tokens = [10, 100, 1]

    def faulty_apply_template(*args, **kwargs):
        messages = args[0]
        # Check content of the first message for simplicity
        first_content = (
            messages[0].get('content')
            if isinstance(messages[0], dict)
            else messages[0].content
        )

        if first_content == 'BAD':
            raise ValueError('Simulated tokenization error')
        elif first_content == 'OK':
            return ok_tokens
        return [1]  # Default fallback

    mock_tokenizer.apply_chat_template.side_effect = faulty_apply_template
    mock_tokenizer.encode.return_value = []  # No assistant response needed

    episodes = base_mdp_env._convert_to_episodes(env_state)

    assert len(episodes) == 1  # Only the OK sample should be converted
    assert episodes[0].ground_truth == 'GT_OK'
    assert (
        f"Failed converting Sample ID {bad_state_id} to EpisodeData"
        in caplog.text
    )
    assert 'Simulated tokenization error' in caplog.text


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
