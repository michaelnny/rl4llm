# test_llm_env.py
import random
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from rl4llm.envs.llm_env import (
    BaseRewardFunction,
    EnvState,
    EpisodeData,
    LLMEnv,
)

# --- Test Fixtures ---


# Dummy Dataset
class SimpleDictDataset(TorchDataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture(scope='module')
def dummy_dataset_data():
    return [
        {'prompt': 'Q: What is 1+1?', 'ground_truth': 'A: 2', 'topic': 'math'},
        {
            'prompt': 'Q: Capital of France?',
            'ground_truth': 'A: Paris',
            'topic': 'geo',
        },
        {
            'prompt': 'Q: Meaning of life?',
            'ground_truth': 'A: 42',
            'topic': 'philosophy',
        },
        {
            'prompt': 'Q: Water boiling point?',
            'ground_truth': 'A: 100C',
            'topic': 'science',
        },
    ]


@pytest.fixture(scope='module')
def dummy_dataset(dummy_dataset_data):
    return SimpleDictDataset(dummy_dataset_data)


# Dummy Reward Function
class MockRewardFunction(BaseRewardFunction):
    def __init__(self, name='mock_reward', reward_value=0.5):
        super().__init__(name=name)
        self.reward_value = reward_value
        self.call_args_list = []  # To track calls

    def __call__(
        self, completions: List[str], ground_truths: List[str], **kwargs
    ) -> List[float]:
        self.call_args_list.append(
            {
                'completions': completions,
                'ground_truths': ground_truths,
                'kwargs': kwargs,
            }
        )
        # Return a fixed reward for simplicity, length matching completions
        return [
            float(self.reward_value + random.uniform(-0.1, 0.1))
            for _ in completions
        ]


@pytest.fixture
def mock_reward_fn():
    return MockRewardFunction(reward_value=0.75)


class MockLLM(PreTrainedModel):
    def __init__(self, config, tokenizer):
        super().__init__(config)  # Initialize the base class
        self.tokenizer = tokenizer
        # No self.device assignment here - rely on base class property
        self.generate_calls = []
        # Add a dummy parameter so self.device works correctly
        # This parameter will be moved by the inherited .to() method
        self.dummy_param = nn.Parameter(torch.tensor(1.0))

    # No custom 'to' method - inherit from PreTrainedModel

    # Keep the generate method, but ensure tensors are created on self.device
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_return_sequences: int = 1,
        max_new_tokens: int = 5,
        pad_token_id: int = 0,
        **kwargs,  # Capture other args passed
    ) -> GenerateDecoderOnlyOutput:
        """Mocks the generate function."""
        self.generate_calls.append(
            {
                'input_ids': input_ids.cpu(),  # Keep recording CPU version for simplicity
                'attention_mask': attention_mask.cpu(),
                'num_return_sequences': num_return_sequences,
                'max_new_tokens': max_new_tokens,
                'pad_token_id': pad_token_id,
                'kwargs': kwargs,
            }
        )

        # *** Use self.device (the property) when creating tensors ***
        current_device = self.device  # Read the device property from base class

        batch_size, seq_len = input_ids.shape
        output_batch_size = batch_size * num_return_sequences

        all_sequences = []
        for i in range(batch_size):
            # Ensure the input segment is on the correct device (it should be already if passed correctly)
            prompt_segment = input_ids[i : i + 1, :]  # Shape (1, seq_len)
            for j in range(num_return_sequences):
                new_token_val = 1000 + i * 100 + j * 10
                # Create new tensors on the model's current device
                new_tokens = torch.randint(
                    new_token_val,
                    new_token_val + 5,
                    (1, max_new_tokens),  # Shape (1, max_new_tokens)
                    device=current_device,  # Use the property here
                )
                # Combine prompt and new tokens
                full_seq = torch.cat([prompt_segment, new_tokens], dim=1)
                all_sequences.append(full_seq)

        # Stack sequences
        sequences_tensor = torch.cat(
            all_sequences, dim=0
        )  # Shape (output_batch_size, seq_len + max_new_tokens)

        return GenerateDecoderOnlyOutput(sequences=sequences_tensor)


# --- Fixture using the revised MockLLM ---


# Dummy Tokenizer
@pytest.fixture(scope='module')
def dummy_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def mock_llm(dummy_tokenizer):
    config = AutoConfig.from_pretrained('gpt2')  # Use config from a real model
    model = MockLLM(config, dummy_tokenizer)
    # The model will be on CPU by default after initialization
    # If you needed to test GPU interaction, you would call:
    # if torch.cuda.is_available():
    #     model.to('cuda')
    return model


# Fixture for the Environment itself
@pytest.fixture
def llm_env(dummy_dataset, dummy_tokenizer, mock_reward_fn):
    return LLMEnv(
        dataset=dummy_dataset,
        batch_size=2,  # Default test batch size
        tokenizer=dummy_tokenizer,
        reward_functions=[mock_reward_fn],
        rank=0,
        world_size=1,
        seed=42,
    )


@pytest.fixture
def llm_env_bs1(dummy_dataset, dummy_tokenizer, mock_reward_fn):
    # Specific instance with batch_size=1
    return LLMEnv(
        dataset=dummy_dataset,
        batch_size=1,
        tokenizer=dummy_tokenizer,
        reward_functions=[mock_reward_fn],
        seed=42,
    )


# --- Test Cases ---


def test_env_initialization(llm_env, dummy_tokenizer, mock_reward_fn):
    """Tests if the environment initializes correctly."""
    assert llm_env._batch_size == 2
    assert llm_env._tokenizer == dummy_tokenizer
    assert llm_env._reward_functions == [mock_reward_fn]
    assert llm_env._seed == 42
    assert hasattr(llm_env, '_loader')
    assert hasattr(llm_env, '_dataset_iterator')


def test_env_initialization_invalid_batch_size(
    dummy_dataset, dummy_tokenizer, mock_reward_fn
):
    """Tests error handling for invalid batch size."""
    with pytest.raises(ValueError, match='Batch size must be at least 1'):
        LLMEnv(dummy_dataset, 0, dummy_tokenizer, [mock_reward_fn])


def test_env_initialization_invalid_rewards(dummy_dataset, dummy_tokenizer):
    """Tests error handling for invalid reward functions."""
    with pytest.raises(
        ValueError, match='reward_functions must be a non-empty list'
    ):
        LLMEnv(dummy_dataset, 1, dummy_tokenizer, [])
    with pytest.raises(
        ValueError, match='reward_functions must be a non-empty list'
    ):
        LLMEnv(
            dummy_dataset, 1, dummy_tokenizer, [lambda x: x]
        )  # Not BaseRewardFunction


def test_env_reset(llm_env):
    """Tests the reset method and the structure of EnvState."""
    state = llm_env.reset()

    assert isinstance(state, EnvState)

    # Check batch size
    assert len(state.prompt) == llm_env._batch_size
    assert len(state.ground_truth) == llm_env._batch_size
    assert len(state.raw_data) == llm_env._batch_size
    assert state.input_ids.shape[0] == llm_env._batch_size
    assert state.attention_mask.shape[0] == llm_env._batch_size

    # Check tensor shapes match
    assert state.input_ids.shape == state.attention_mask.shape

    # Check content types
    assert isinstance(state.prompt[0], str)
    assert isinstance(state.ground_truth[0], str)
    assert isinstance(state.raw_data[0], dict)
    assert 'prompt' in state.raw_data[0]  # Check raw data structure
    assert 'ground_truth' in state.raw_data[0]
    assert 'topic' in state.raw_data[0]  # Check custom field


def test_env_reset_exhaustion(dummy_dataset, dummy_tokenizer, mock_reward_fn):
    """Tests if the dataloader iterator resets after exhaustion."""
    env = LLMEnv(
        dummy_dataset,
        batch_size=len(dummy_dataset),
        tokenizer=dummy_tokenizer,
        reward_functions=[mock_reward_fn],
    )
    state1 = env.reset()  # First batch (whole dataset)
    assert len(state1.prompt) == len(dummy_dataset)

    # Try resetting again, should re-initialize the iterator
    try:
        state2 = env.reset()
        assert len(state2.prompt) == len(dummy_dataset)
        # Check if data is potentially different due to shuffling (though seed is fixed)
        # or at least that it didn't crash
    except StopIteration:
        pytest.fail(
            'DataLoader iterator did not reset correctly after exhaustion.'
        )


# --- Rollout Tests ---


@pytest.mark.parametrize(
    'batch_size, num_return_sequences',
    [
        (1, 1),
        (2, 1),
        (1, 3),
        (2, 3),  # Test both batching and multiple returns
    ],
)
def test_rollout_output_structure(
    dummy_dataset,
    dummy_tokenizer,
    mock_reward_fn,
    mock_llm,
    batch_size,
    num_return_sequences,
):
    """Tests the overall structure and size of the rollout output."""
    env = LLMEnv(
        dataset=dummy_dataset,
        batch_size=batch_size,
        tokenizer=dummy_tokenizer,
        reward_functions=[mock_reward_fn],
        seed=42,
    )
    gen_args = {
        'num_return_sequences': num_return_sequences,
        'max_new_tokens': 10,
    }

    episodes = env.rollout(mock_llm, gen_args)

    expected_episodes = batch_size * num_return_sequences
    assert len(episodes) == expected_episodes
    assert all(isinstance(ep, EpisodeData) for ep in episodes)

    # Check model generate was called once
    assert len(mock_llm.generate_calls) == 1
    call_info = mock_llm.generate_calls[0]
    assert call_info['input_ids'].shape[0] == batch_size
    assert call_info['num_return_sequences'] == num_return_sequences
    assert call_info['max_new_tokens'] == 10

    # Check reward function was called once with the correct number of items
    assert len(mock_reward_fn.call_args_list) == 1
    reward_call_info = mock_reward_fn.call_args_list[0]
    assert len(reward_call_info['completions']) == expected_episodes
    assert len(reward_call_info['ground_truths']) == expected_episodes


def test_rollout_data_association(
    llm_env_bs1, mock_llm, mock_reward_fn, dummy_dataset_data
):
    """Tests if generated data is correctly associated with the original prompt."""
    batch_size = 1
    num_return_sequences = 3
    gen_args = {
        'num_return_sequences': num_return_sequences,
        'max_new_tokens': 8,
    }

    # Manually get the first expected item from the dataset (due to fixed seed)
    # Note: DataLoader with shuffle=True and fixed seed gives predictable order
    # If shuffle=False, it would just be dummy_dataset_data[0]
    # We need to simulate the dataloader's first fetch with shuffle=True, seed=42
    # For simplicity, let's assume the first item fetched IS dummy_dataset_data[0]
    # (In reality, you might need to run the dataloader once to confirm the first item)
    # Let's reset the seed locally to be sure of the dataset order
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    temp_loader = torch.utils.data.DataLoader(
        llm_env_bs1._loader.dataset,
        batch_size=llm_env_bs1._batch_size,
        shuffle=True,
    )
    expected_first_item = next(iter(temp_loader))

    original_prompt = expected_first_item['prompt'][0]
    original_gt = expected_first_item['ground_truth'][0]
    original_raw = expected_first_item  # The collated dict for the first item

    # Perform rollout
    episodes = llm_env_bs1.rollout(mock_llm, gen_args)

    assert len(episodes) == batch_size * num_return_sequences

    for i in range(num_return_sequences):
        episode = episodes[i]
        assert isinstance(episode, EpisodeData)

        # Check prompt association
        assert episode.prompt_text == original_prompt

        # Check raw data association (compare dicts)
        # The raw_data in EpisodeData should be the dict for the *single* item
        # from the original batch it corresponds to.
        assert episode.raw_data['prompt'] == original_raw['prompt'][0]
        assert (
            episode.raw_data['ground_truth'] == original_raw['ground_truth'][0]
        )
        assert episode.raw_data['topic'] == original_raw['topic'][0]

        # Check reward calculation inputs (via mock)
        reward_call_info = mock_reward_fn.call_args_list[0]
        assert reward_call_info['ground_truths'][i] == original_gt

        # Check completion details
        assert isinstance(episode.completion_text, str)
        assert len(episode.completion_text) > 0  # Basic check it's not empty
        assert isinstance(episode.completion_tokens, torch.Tensor)
        assert episode.completion_length == episode.completion_tokens.shape[0]
        assert episode.completion_length > 0

        # Check reward structure
        assert isinstance(episode.reward_dict, dict)
        assert mock_reward_fn.name in episode.reward_dict
        assert isinstance(episode.reward_dict[mock_reward_fn.name], float)


def test_rollout_batch_data_association(
    llm_env: LLMEnv,  # Type hint for clarity
    mock_llm,
    mock_reward_fn,
    dummy_dataset_data,
    mocker,  # Inject the pytest-mock fixture
):
    """Tests data association by mocking the reset method."""
    batch_size = llm_env._batch_size  # Use batch_size from the fixture
    num_return_sequences = 2
    gen_args = {
        'num_return_sequences': num_return_sequences,
        'max_new_tokens': 6,
    }

    # 1. Determine the data reset *should* return (based on seed)
    #    We still need to know what the first batch *would* be.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Use the actual loader from the env instance to be precise
    temp_loader = DataLoader(
        llm_env._loader.dataset,
        batch_size=llm_env._batch_size,
        shuffle=True,  # Match env's shuffle setting
        collate_fn=llm_env._collate_fn,  # Match env's collate_fn
    )
    expected_batch_dict = next(iter(temp_loader))

    # 2. Manually prepare the EnvState that reset should return
    #    (Mimicking the logic within _prepare_initial_state)
    prompts_to_use = expected_batch_dict['prompt']
    gts_to_use = expected_batch_dict['ground_truth']
    raw_data_list_to_use = [
        {key: expected_batch_dict[key][i] for key in expected_batch_dict}
        for i in range(len(prompts_to_use))
    ]

    inputs = llm_env._tokenizer(
        prompts_to_use,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=llm_env._max_prompt_length,
        return_attention_mask=True,
    )
    prompt_length = inputs['input_ids'].shape[1]

    # Create the exact EnvState object we want reset to return
    mock_state = EnvState(
        prompt=prompts_to_use,
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        ground_truth=gts_to_use,
        raw_data=raw_data_list_to_use,
        prompt_length=prompt_length,
    )

    # 3. Mock the llm_env.reset method
    mocker.patch.object(llm_env, 'reset', return_value=mock_state)

    # 4. Perform rollout - now it will use mock_state instead of fetching data
    episodes = llm_env.rollout(mock_llm, gen_args)

    # 5. Assertions - Compare against the data used to create mock_state
    assert len(episodes) == batch_size * num_return_sequences

    for i in range(len(episodes)):
        episode = episodes[i]
        original_item_index = (
            i // num_return_sequences
        )  # 0, 0, 1, 1 for bs=2, nrs=2

        # Check prompt association
        assert episode.prompt_text == prompts_to_use[original_item_index]

        # Check raw data association
        assert episode.raw_data == raw_data_list_to_use[original_item_index]

        # Check reward calculation inputs (via mock)
        # Ensure mock_reward_fn was called (it should have been if rollout ran)
        assert len(mock_reward_fn.call_args_list) == 1
        reward_call_info = mock_reward_fn.call_args_list[0]
        assert (
            reward_call_info['ground_truths'][i]
            == gts_to_use[original_item_index]
        )

        # Check reward value assignment
        assert mock_reward_fn.name in episode.reward_dict

    # Verify reset was called exactly once by rollout
    llm_env.reset.assert_called_once()


def test_rollout_tokenization_details(
    llm_env_bs1,  # Env with bs=1
    mock_llm,
    dummy_tokenizer,
    mocker,  # Inject mocker
):
    """Verify token lengths and padding in EpisodeData by mocking reset."""
    num_return_sequences = 1
    max_new_tokens = 7
    gen_args = {
        'num_return_sequences': num_return_sequences,
        'max_new_tokens': max_new_tokens,
    }

    # 1. Call reset ONCE to get the target state we want rollout to use
    #    (This also determines the expected prompt length/tokens)
    target_state = llm_env_bs1.reset()
    prompt_len_from_state = target_state.input_ids.shape[1]
    prompt_tokens_from_state = target_state.input_ids[0]  # Batch size is 1

    # 2. Mock the env's reset method to return this specific state
    mocker.patch.object(llm_env_bs1, 'reset', return_value=target_state)

    # 3. Perform rollout - it will now use target_state internally
    episodes = llm_env_bs1.rollout(mock_llm, gen_args)
    assert len(episodes) == 1
    episode = episodes[0]

    # 4. Assertions: Now comparing values derived from the SAME state
    # Check prompt length stored in episode matches the target state's length
    assert episode.prompt_length == prompt_len_from_state

    # Check prompt tokens stored match target state's tokens
    assert torch.equal(
        episode.prompt_tokens.cpu(), prompt_tokens_from_state.cpu()
    )

    # Check completion length and tokens
    # MockLLM generates exactly max_new_tokens (unless changed)
    expected_completion_len = max_new_tokens
    assert episode.completion_length == expected_completion_len
    assert episode.completion_tokens.shape[0] == expected_completion_len

    # Decode completion tokens and check against text (optional/approximate)
    decoded_from_tokens = dummy_tokenizer.decode(
        episode.completion_tokens, skip_special_tokens=True
    )
    assert isinstance(episode.completion_text, str)
    # print(f"\nDecoded: '{decoded_from_tokens}' vs Text: '{episode.completion_text}'") # Debug print

    # Verify reset was called exactly once by rollout
    llm_env_bs1.reset.assert_called_once()


@pytest.mark.parametrize(
    'batch_size, num_return_sequences, dataset',
    [
        # Case 1: Single prompt, single sequence
        (
            1,
            1,
            [{'prompt': 'Hello', 'ground_truth': 'World'}],
        ),
        # Case 2: Multiple prompts of different lengths, single sequence
        (
            2,
            1,
            [
                {'prompt': 'Hi', 'ground_truth': 'There'},
                {'prompt': 'This is a test', 'ground_truth': 'Indeed'},
            ],
        ),
        # Case 3: Single prompt, multiple sequences
        (
            1,
            3,
            [{'prompt': 'Test', 'ground_truth': 'Case'}],
        ),
        # Case 4: Multiple prompts, multiple sequences
        (
            2,
            3,
            [
                {'prompt': 'Short', 'ground_truth': 'One'},
                {'prompt': 'Longer prompt here', 'ground_truth': 'Two'},
            ],
        ),
        # Case 5: Prompt requiring truncation
        (
            1,
            1,
            [
                {
                    'prompt': 'This is a very long prompt to truncate',
                    'ground_truth': 'Truncated',
                }
            ],
        ),
    ],
)
def test_unpadded_prompt_tokens(
    batch_size,
    num_return_sequences,
    dataset,
    dummy_tokenizer,
    mock_reward_fn,
    mock_llm,
):
    """Test that prompt_tokens in EpisodeData are unpadded and match expected tokenization."""
    # Set a specific max_prompt_length for testing truncation
    max_prompt_length = 5  # Small value to force truncation in Case 6

    # Create LLMEnv with the parameterized dataset
    env = LLMEnv(
        dataset=dataset,
        batch_size=batch_size,
        tokenizer=dummy_tokenizer,
        reward_functions=[mock_reward_fn],
        max_prompt_length=max_prompt_length,
        seed=42,
    )

    gen_args = {
        'num_return_sequences': num_return_sequences,
        'max_new_tokens': 10,
    }

    # Run rollout
    episodes = env.rollout(mock_llm, gen_args)

    # Verify the number of episodes
    expected_episodes = batch_size * num_return_sequences
    assert (
        len(episodes) == expected_episodes
    ), f"Expected {expected_episodes} episodes, got {len(episodes)}"

    # Check each episode
    for idx, episode in enumerate(episodes):
        prompt_text = episode.prompt_text
        prompt_tokens = episode.prompt_tokens
        prompt_length = episode.prompt_length

        # Tokenize the prompt text with the same settings as in _prepare_initial_state
        expected_inputs = dummy_tokenizer(
            prompt_text,
            max_length=max_prompt_length,
            truncation=True,
            padding=False,
            return_tensors='pt',
        )
        expected_tokens = expected_inputs['input_ids'][0].tolist()

        # Assertion 1: prompt_tokens match the expected tokenization
        assert prompt_tokens.tolist() == expected_tokens, (
            f"Prompt tokens mismatch for prompt '{prompt_text}':\n"
            f"Expected: {expected_tokens}\n"
            f"Got: {prompt_tokens.tolist()}"
        )

        # Assertion 2: prompt_length matches the length of prompt_tokens
        assert len(prompt_tokens) == prompt_length, (
            f"Prompt length mismatch for prompt '{prompt_text}':\n"
            f"Expected length: {len(prompt_tokens)}, Got: {prompt_length}"
        )

        # Assertion 3: No padding tokens in prompt_tokens (except for empty case)
        pad_token_id = dummy_tokenizer.pad_token_id
        if len(prompt_tokens) > 0:
            assert (
                pad_token_id not in prompt_tokens
            ), f"Padding token {pad_token_id} found in prompt_tokens for prompt '{prompt_text}': {prompt_tokens.tolist()}"
        else:
            # For empty prompt, ensure prompt_tokens is empty
            assert (
                prompt_text == ''
            ), 'Non-empty prompt text with empty prompt_tokens'
            assert (
                len(prompt_tokens) == 0
            ), f"Expected empty prompt_tokens, got {prompt_tokens.tolist()}"
