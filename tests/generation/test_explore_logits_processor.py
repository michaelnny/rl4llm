from typing import List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizer

from rl4llm.generation.explore_processor import ExploreLogitsProcessor


class MockTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_map, **kwargs):
        self._vocab_map = vocab_map
        self._inv_vocab_map = {v: k for k, v in vocab_map.items()}
        self.pad_token_id = vocab_map.get('<pad>', 0)
        self.eos_token_id = vocab_map.get('<eos>', 1)
        self.unk_token_id = vocab_map.get('<unk>', 2)
        super().__init__(
            pad_token='<pad>',
            eos_token='<eos>',
            unk_token='<unk>',
            vocab_file=None,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_map)

    def _convert_token_to_id(self, token):
        return self._vocab_map.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        return self._inv_vocab_map.get(index, '<unk>')

    def get_vocab(self):
        return self._vocab_map.copy()

    def _tokenize(self, text, **kwargs):
        return text.split()

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        decoded = []
        special_tokens = (
            {self.pad_token_id, self.eos_token_id, self.unk_token_id}
            if skip_special_tokens
            else set()
        )
        for seq in sequences:
            tokens = [
                self._convert_id_to_token(
                    idx.item() if isinstance(idx, torch.Tensor) else idx
                )
                for idx in seq
                if not (skip_special_tokens and idx in special_tokens)
            ]
            decoded.append(' '.join(tokens))
        return decoded

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        return (f"{save_directory}/{filename_prefix or ''}vocab.txt",)


@pytest.fixture(scope='session')
def mock_tokenizer():
    """Provides a mock tokenizer with a predefined vocabulary for testing."""
    vocab = {
        '<pad>': 0,
        '<eos>': 1,
        '<unk>': 2,
        'hello': 3,
        'world': 4,
        'test': 5,
        'a': 6,
        'pattern': 7,
        'prevent': 8,
        'correct': 9,
        'incorrect': 10,
        'source': 11,
        'target1': 12,
        'target2': 13,
        'another': 14,
        'token': 15,
        'skip': 16,
        'explore': 17,
        'replace': 18,
        'max': 19,
        'count': 20,
        'always': 21,
        'fail': 22,
        'AAA': 23,
        'BBB': 24,
        'CCC': 25,
        'DDD': 26,
        'EEE': 27,
        'FFF': 28,
        'GGG': 29,
        'HHH': 30,
    }
    return MockTokenizer(vocab)


@pytest.fixture
def initial_input_ids():
    """Returns a tensor of initial input IDs for testing with batch size 3 and sequence length 5."""
    return torch.tensor(
        [[3, 4, 0, 0, 0], [6, 5, 1, 0, 0], [14, 15, 15, 0, 0]], dtype=torch.long
    )


@pytest.fixture
def initial_scores(mock_tokenizer):
    """Generates initial scores for testing with controlled high-score tokens."""
    batch_size, vocab_size = 3, mock_tokenizer.vocab_size
    scores = torch.randn(batch_size, vocab_size) * 2
    scores[0, [3, 4, 11, 23, 24]] = torch.tensor(
        [10, 9, 8, 7, 6], dtype=torch.float
    )
    scores[1, [5, 6, 11, 25, 26]] = torch.tensor(
        [10, 9, 8, 7, 6], dtype=torch.float
    )
    scores[2, [15, 14, 11, 27, 28]] = torch.tensor(
        [10, 9, 8, 7, 6], dtype=torch.float
    )
    return scores.float()


def correctness_always_fail(texts: List[str]) -> List[float]:
    """Marks all sequences as incorrect."""
    return [0.0] * len(texts)


def correctness_always_pass(texts: List[str]) -> List[float]:
    """Marks all sequences as correct."""
    return [1.0] * len(texts)


def correctness_if_contains_fail(texts: List[str]) -> List[float]:
    """Marks sequences containing 'fail' as incorrect."""
    return [0.0 if 'fail' in text else 1.0 for text in texts]


def simulate_steps(processor, initial_ids, initial_scores, num_steps):
    """Simulates multiple generation steps and returns processed scores."""
    all_scores, current_ids = [], initial_ids.clone()
    step_scores = initial_scores.clone()
    for _ in range(num_steps):
        processed_scores = processor(current_ids, step_scores.clone())
        all_scores.append(processed_scores.clone())
        next_tokens = torch.argmax(processed_scores, dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_tokens], dim=1)
    return all_scores


# Initialization Tests
@pytest.mark.parametrize(
    'temp, is_batch',
    [(0.7, False), ([0.7, 0.8], True), (torch.tensor([0.7, 0.8]), True)],
)
def test_init_temperature(mock_tokenizer, temp, is_batch):
    """Tests initialization with different temperature types."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=5, tokenizer=mock_tokenizer, temperature=temp
    )
    assert processor._is_batch_temp == is_batch
    assert processor.temperature is None
    if is_batch:
        assert torch.equal(
            processor._initial_temperature_val,
            (
                torch.tensor(temp, dtype=torch.float32)
                if isinstance(temp, list)
                else temp
            ),
        )
    else:
        assert processor._initial_temperature_val == temp


def test_init_invalid_temp_type(mock_tokenizer):
    """Tests that an invalid temperature type raises TypeError."""
    with pytest.raises(TypeError):
        ExploreLogitsProcessor(
            initial_seq_len=5, tokenizer=mock_tokenizer, temperature='invalid'
        )


def test_init_state_variables(mock_tokenizer):
    """Tests initial state variables are set correctly."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=5, tokenizer=mock_tokenizer, temperature=1.0
    )
    assert processor.temperature is None
    assert processor.source_tokens_tensor is None
    assert processor.target_tokens_tensor is None
    assert processor.current_step == 0
    assert processor.replacement_counts is None


# State Management Tests
def test_state_initialization_first_call(
    mock_tokenizer, initial_input_ids, initial_scores
):
    """Tests state initialization on first processor call."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=[0.5, 1.0, 1.5],
        replace_source_tokens=[11],
        replace_target_tokens=[12],
    )
    processor(initial_input_ids, initial_scores)
    assert processor.current_step == 0
    assert torch.equal(
        processor.temperature,
        torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32),
    )
    assert torch.equal(
        processor.source_tokens_tensor, torch.tensor([11], dtype=torch.long)
    )
    assert torch.equal(
        processor.target_tokens_tensor, torch.tensor([12], dtype=torch.long)
    )
    assert torch.equal(
        processor.replacement_counts, torch.zeros(3, dtype=torch.long)
    )


def test_state_increment_step(
    mock_tokenizer, initial_input_ids, initial_scores
):
    """Tests that current_step increments with each call."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=1.0,
    )
    processor(initial_input_ids, initial_scores)
    assert processor.current_step == 0
    processor(
        torch.cat(
            [initial_input_ids, torch.zeros((3, 1), dtype=torch.long)], dim=1
        ),
        initial_scores,
    )
    assert processor.current_step == 1


def test_state_reset_on_initial_len(
    mock_tokenizer, initial_input_ids, initial_scores
):
    """Tests state reset when input length matches initial_seq_len."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=1.0,
        replace_source_tokens=[11],
        replace_target_tokens=[12],
        replace_max_per_seq=1,
        replace_prob=1.0,
        correctness_callback=correctness_always_fail,
        explore_steps=1,
        explore_skip=0,
    )
    processor(initial_input_ids, initial_scores)
    processor(
        torch.cat([initial_input_ids, torch.tensor([[10], [10], [10]])], dim=1),
        initial_scores,
    )
    processor.replacement_counts = torch.ones(3, dtype=torch.long)
    processor(initial_input_ids, initial_scores)
    assert processor.current_step == 0
    assert torch.equal(
        processor.replacement_counts, torch.zeros(3, dtype=torch.long)
    )


# Temperature Tests
@pytest.mark.parametrize('temp_val', [1.0, 0.7, 1.5])
def test_scalar_temperature(
    mock_tokenizer, initial_input_ids, initial_scores, temp_val
):
    """Tests scalar temperature application to scores."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=temp_val,
    )
    processed_scores = processor(initial_input_ids, initial_scores.clone())
    expected_scores = (
        initial_scores if temp_val == 1.0 else initial_scores / temp_val
    )
    torch.testing.assert_close(processed_scores, expected_scores)
    if temp_val != 1.0:
        assert torch.all(
            torch.argmax(processed_scores, dim=1)
            == torch.argmax(initial_scores, dim=1)
        )


def test_batch_temperature(mock_tokenizer, initial_input_ids, initial_scores):
    """Tests batch-specific temperature application."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=[0.5, 1.0, 2.0],
    )
    processed_scores = processor(initial_input_ids, initial_scores.clone())
    expected_scores = initial_scores.clone()
    expected_scores[0] /= 0.5
    expected_scores[1] /= 1.0
    expected_scores[2] /= 2.0
    torch.testing.assert_close(processed_scores, expected_scores)


def test_zero_temperature(mock_tokenizer, initial_input_ids, initial_scores):
    """Tests zero temperature forces greedy selection."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=0.0,
    )
    processed_scores = processor(initial_input_ids, initial_scores.clone())
    expected_argmax = torch.argmax(initial_scores, dim=-1)
    for i in range(3):
        assert (
            torch.isneginf(processed_scores[i]).sum()
            == initial_scores.shape[1] - 1
        )
        assert processed_scores[i, expected_argmax[i]] == 100.0


def test_mixed_zero_batch_temperature(
    mock_tokenizer, initial_input_ids, initial_scores
):
    """Tests mixed batch temperature with zero values."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=[0.5, 0.0, 1.5],
    )
    processed_scores = processor(initial_input_ids, initial_scores.clone())
    torch.testing.assert_close(processed_scores[0], initial_scores[0] / 0.5)
    assert (
        torch.isneginf(processed_scores[1]).sum() == initial_scores.shape[1] - 1
    )
    assert processed_scores[1, torch.argmax(initial_scores[1])] == 100.0
    torch.testing.assert_close(processed_scores[2], initial_scores[2] / 1.5)


# Exploration Tests
@pytest.mark.parametrize(
    'step, explore_skip, explore_steps, should_explore',
    [
        (0, 2, 5, False),
        (2, 2, 5, True),
        (7, 2, 5, False),
        (0, 0, 3, True),
        (3, 0, 3, False),
    ],
)
def test_exploration_activation(
    mock_tokenizer,
    initial_input_ids,
    initial_scores,
    step,
    explore_skip,
    explore_steps,
    should_explore,
):
    """Tests exploration activation based on step conditions."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=initial_input_ids.shape[1],
        tokenizer=mock_tokenizer,
        temperature=1.0,
        explore_steps=explore_steps,
        explore_skip=explore_skip,
        explore_top_k=5,
    )
    processed_scores = simulate_steps(
        processor, initial_input_ids, initial_scores, step + 1
    )[step]
    if should_explore:
        k = min(
            max(2, int(5 * (0.9 ** max(0, step - explore_skip)))),
            initial_scores.shape[-1],
        )
        _, top_k_indices = torch.topk(initial_scores, k=k, dim=-1)
        for i in range(3):
            expected_non_inf = torch.zeros_like(
                processed_scores[i], dtype=torch.bool
            )
            expected_non_inf[top_k_indices[i]] = True
            assert torch.equal(
                ~torch.isneginf(processed_scores[i]), expected_non_inf
            )
            assert torch.all(processed_scores[i][expected_non_inf] == 0.0)
    else:
        torch.testing.assert_close(processed_scores, initial_scores)


# Replacement Tests
@pytest.fixture
def replacement_processor(mock_tokenizer):
    """Provides a processor configured for replacement testing."""
    return ExploreLogitsProcessor(
        initial_seq_len=5,
        tokenizer=mock_tokenizer,
        temperature=1.0,
        explore_steps=1,
        explore_skip=0,
        replace_source_tokens=[11],
        replace_target_tokens=[12, 13],
        replace_prevent_patterns=[[8, 7]],
        replace_prob=1.0,
        replace_max_per_seq=2,
        correctness_callback=correctness_always_fail,
    )


def test_replacement_eligible_basic(
    replacement_processor, initial_input_ids, initial_scores
):
    """Tests basic replacement eligibility and application."""
    replacement_processor.current_step = 1
    replacement_processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    scores = initial_scores.clone()
    scores[:, [11, 12, 13]] = torch.tensor([5.0, 4.0, 3.0], dtype=torch.float)
    input_ids = torch.cat(
        [initial_input_ids, torch.tensor([[3], [5], [14]], dtype=torch.long)],
        dim=1,
    )
    processed_scores = replacement_processor(input_ids, scores)
    assert torch.all(torch.isneginf(processed_scores[:, 11]))
    assert torch.all(
        (processed_scores[:, 12] == scores[:, 12] + 100.0)
        ^ (processed_scores[:, 13] == scores[:, 13] + 100.0)
    )
    assert torch.all(replacement_processor.replacement_counts == 1)


def test_replacement_probability_zero(
    replacement_processor, initial_input_ids, initial_scores
):
    """Tests no replacement occurs with zero probability."""
    replacement_processor.replace_prob = 0.0
    replacement_processor.current_step = 1
    replacement_processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    scores = initial_scores.clone()
    input_ids = torch.cat(
        [initial_input_ids, torch.tensor([[3], [5], [14]], dtype=torch.long)],
        dim=1,
    )
    processed_scores = replacement_processor(input_ids, scores)
    torch.testing.assert_close(processed_scores, scores)
    assert torch.all(replacement_processor.replacement_counts == 0)


@patch('torch.rand')
def test_replacement_probability_half(
    mock_rand, replacement_processor, initial_input_ids, initial_scores
):
    """Tests replacement with 0.5 probability using mocked randomness."""
    replacement_processor.replace_prob = 0.5
    replacement_processor.current_step = 1
    replacement_processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    mock_rand.return_value = torch.tensor(
        [0.2, 0.6, 0.7], device=initial_input_ids.device
    )
    scores = initial_scores.clone()
    scores[:, [11, 12, 13]] = torch.tensor([5.0, 4.0, 3.0], dtype=torch.float)
    input_ids = torch.cat(
        [initial_input_ids, torch.tensor([[3], [5], [14]], dtype=torch.long)],
        dim=1,
    )
    processed_scores = replacement_processor(input_ids, scores)
    assert torch.isneginf(processed_scores[0, 11])
    assert replacement_processor.replacement_counts[0] == 1
    torch.testing.assert_close(processed_scores[1:], scores[1:])
    assert torch.all(replacement_processor.replacement_counts[1:] == 0)


def test_replacement_max_count(
    replacement_processor, initial_input_ids, initial_scores
):
    """Tests replacement stops after reaching max count."""
    replacement_processor.replace_max_per_seq = 1
    replacement_processor.current_step = 1
    replacement_processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    scores = initial_scores.clone()
    scores[:, [11, 12, 13]] = torch.tensor([5.0, 4.0, 3.0], dtype=torch.float)
    input_ids = torch.cat(
        [initial_input_ids, torch.tensor([[3], [5], [14]], dtype=torch.long)],
        dim=1,
    )
    replacement_processor(input_ids, scores)
    replacement_processor.current_step = 2
    processed_scores = replacement_processor(
        torch.cat(
            [input_ids, torch.tensor([[3], [5], [14]], dtype=torch.long)], dim=1
        ),
        scores,
    )
    assert torch.all(replacement_processor.replacement_counts == 1)
    torch.testing.assert_close(processed_scores, scores)


def test_replacement_prevent_pattern_present(
    replacement_processor, initial_input_ids, initial_scores
):
    """Tests replacement is prevented when prevent pattern is present."""
    replacement_processor.current_step = 1
    replacement_processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    scores = initial_scores.clone()
    input_ids = torch.cat(
        [initial_input_ids, torch.tensor([[8], [5], [14]], dtype=torch.long)],
        dim=1,
    )
    input_ids = torch.cat(
        [input_ids, torch.tensor([[7], [5], [14]], dtype=torch.long)], dim=1
    )
    processed_scores = replacement_processor(input_ids, scores)
    torch.testing.assert_close(processed_scores[0], scores[0])
    assert replacement_processor.replacement_counts[0] == 0
    assert torch.all(torch.isneginf(processed_scores[1:, 11]))
    assert torch.all(replacement_processor.replacement_counts[1:] == 1)


def test_replacement_correctness_pass(
    replacement_processor, initial_input_ids, initial_scores
):
    """Tests no replacement occurs when correctness callback passes."""
    replacement_processor.correctness_callback = correctness_always_pass
    replacement_processor.current_step = 1
    replacement_processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    scores = initial_scores.clone()
    input_ids = torch.cat(
        [initial_input_ids, torch.tensor([[3], [5], [14]], dtype=torch.long)],
        dim=1,
    )
    processed_scores = replacement_processor(input_ids, scores)
    torch.testing.assert_close(processed_scores, scores)
    assert torch.all(replacement_processor.replacement_counts == 0)


def test_replacement_correctness_conditional(
    mock_tokenizer, initial_input_ids, initial_scores
):
    """Tests conditional replacement based on correctness callback."""
    processor = ExploreLogitsProcessor(
        initial_seq_len=5,
        tokenizer=mock_tokenizer,
        temperature=1.0,
        explore_steps=1,
        explore_skip=0,
        replace_source_tokens=[11],
        replace_target_tokens=[12, 13],
        replace_prob=1.0,
        replace_max_per_seq=1,
        correctness_callback=correctness_if_contains_fail,
    )
    processor.current_step = 1
    processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    scores = initial_scores.clone()
    scores[:, [11, 12, 13]] = torch.tensor([5.0, 4.0, 3.0], dtype=torch.float)
    input_ids = torch.cat(
        [initial_input_ids, torch.tensor([[22], [9], [14]], dtype=torch.long)],
        dim=1,
    )
    processed_scores = processor(input_ids, scores)
    assert torch.isneginf(processed_scores[0, 11])
    assert processor.replacement_counts[0] == 1
    torch.testing.assert_close(processed_scores[1:], scores[1:])
    assert torch.all(processor.replacement_counts[1:] == 0)


def test_replacement_step_condition(
    replacement_processor, initial_input_ids, initial_scores
):
    """Tests replacement only occurs after exploration window."""
    replacement_processor.current_step = 0
    replacement_processor._initialize_state(
        initial_input_ids.shape[0], initial_input_ids.device
    )
    scores = initial_scores.clone()
    processed_scores = replacement_processor(initial_input_ids, scores)
    k = min(replacement_processor.explore_top_k, scores.shape[-1])
    _, top_k_indices = torch.topk(scores, k=k, dim=-1)
    for i in range(3):
        expected_non_inf = torch.zeros_like(
            processed_scores[i], dtype=torch.bool
        )
        expected_non_inf[top_k_indices[i]] = True
        assert torch.equal(
            ~torch.isneginf(processed_scores[i]), expected_non_inf
        )
        assert torch.all(processed_scores[i][expected_non_inf] == 0.0)
    assert torch.all(replacement_processor.replacement_counts == 0)
