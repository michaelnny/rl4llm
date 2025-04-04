from typing import Callable, List, Optional, Tuple, Union
from unittest.mock import (
    MagicMock,
    Mock,
    call,
    patch,
)

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor

from rl4llm.generation.explore_processor import ExploreLogitsProcessor


# --- Mock Tokenizer ---
class MockTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_map, **kwargs):
        # *** Assign vocab map AND essential special token IDs BEFORE super().__init__ ***
        self._vocab_map = vocab_map
        self._inv_vocab_map = {v: k for k, v in vocab_map.items()}

        # Set essential IDs directly to prevent __getattr__ recursion
        # Ensure these keys exist in your vocab_map
        self.pad_token_id = self._vocab_map.get('<pad>', 0)
        self.eos_token_id = self._vocab_map.get('<eos>', 1)
        self.unk_token_id = self._vocab_map.get('<unk>', 2)

        # Initialize base class - it might call get_vocab() or other methods internally
        # It will now find pad_token_id etc., directly if needed.
        super().__init__(
            pad_token='<pad>',  # Pass token strings to super
            eos_token='<eos>',
            unk_token='<unk>',
            vocab_file=None,  # Indicate we are not loading from file
            **kwargs,  # Pass other kwargs like model_max_length if needed
        )

        # We already set the IDs, super() might have set them too based on the
        # token strings passed, but our direct assignment takes precedence if done before.
        # No need to set them again here unless super() overwrites them unexpectedly.

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_map)

    def _convert_token_to_id(self, token):
        # This should now work without recursion because self.unk_token_id exists directly
        return self._vocab_map.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        return self._inv_vocab_map.get(
            index, '<unk>'
        )  # Use the string "<unk>" here

    def get_vocab(self):
        return self._vocab_map.copy()

    def _tokenize(self, text, **kwargs):
        return text.split()

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        decoded = []
        special_tokens_to_skip = set()
        if skip_special_tokens:
            # Use the actual IDs stored in the instance
            if self.pad_token_id is not None:
                special_tokens_to_skip.add(self.pad_token_id)
            if self.eos_token_id is not None:
                special_tokens_to_skip.add(self.eos_token_id)
            if self.unk_token_id is not None:
                special_tokens_to_skip.add(self.unk_token_id)

        for seq in sequences:
            current_seq_tokens = []
            for idx in seq:
                item_id = idx.item() if isinstance(idx, torch.Tensor) else idx
                if skip_special_tokens and item_id in special_tokens_to_skip:
                    continue
                # Use _convert_id_to_token which handles the <unk> string correctly
                current_seq_tokens.append(self._convert_id_to_token(item_id))
            decoded.append(' '.join(current_seq_tokens))
        return decoded

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        vocab_path = f"{save_directory}/{filename_prefix or ''}vocab.txt"
        return (vocab_path,)

    def _save_special_tokens_map(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        path = (
            f"{save_directory}/{filename_prefix or ''}special_tokens_map.json"
        )
        return (path,)


@pytest.fixture(scope='session')
def mock_tokenizer():
    # Define a simple vocabulary for testing
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
    # Batch size 3, sequence length 5
    return torch.tensor(
        [
            [3, 4, 0, 0, 0],  # "hello world <pad> <pad> <pad>"
            [6, 5, 1, 0, 0],  # "a test <eos> <pad> <pad>"
            [14, 15, 15, 0, 0],  # "another token token <pad> <pad>"
        ],
        dtype=torch.long,
    )


@pytest.fixture
def initial_scores(mock_tokenizer):
    # Batch size 3, vocab size from mock tokenizer
    batch_size = 3
    vocab_size = mock_tokenizer.vocab_size
    # Make scores somewhat predictable for testing
    scores = torch.randn(batch_size, vocab_size) * 2
    # Ensure some specific tokens have higher scores for testing top-k etc.
    scores[0, 3] = 10  # hello
    scores[0, 4] = 9  # world
    scores[0, 11] = 8  # source
    scores[0, 23] = 7  # AAA
    scores[0, 24] = 6  # BBB
    scores[1, 5] = 10  # test
    scores[1, 6] = 9  # a
    scores[1, 11] = 8  # source
    scores[1, 25] = 7  # CCC
    scores[1, 26] = 6  # DDD
    scores[2, 15] = 10  # token (repeated in input)
    scores[2, 14] = 9  # another
    scores[2, 11] = 8  # source
    scores[2, 27] = 7  # EEE
    scores[2, 28] = 6  # FFF
    return scores.float()


# --- Correctness Callback Examples ---
def correctness_always_fail(texts: List[str]) -> List[float]:
    """Marks all sequences as incorrect."""
    return [0.0] * len(texts)


def correctness_always_pass(texts: List[str]) -> List[float]:
    """Marks all sequences as correct."""
    return [1.0] * len(texts)


def correctness_if_contains_fail(texts: List[str]) -> List[float]:
    """Marks sequences containing 'fail' as incorrect."""
    return [0.0 if 'fail' in text else 1.0 for text in texts]


# Helper to simulate multiple steps
def simulate_steps(processor, initial_ids, initial_scores, num_steps):
    """Simulates N generation steps, returning scores at each step."""
    all_scores = []
    current_ids = initial_ids.clone()
    # *** Use predictable scores for subsequent steps' INPUT ***
    step_input_scores = initial_scores.clone()

    for step in range(num_steps):
        # Pass the predictable input scores for this step
        processed_scores = processor(current_ids, step_input_scores.clone())
        all_scores.append(processed_scores.clone())

        # Simulate adding a dummy token (e.g., argmax of processed scores)
        # Note: This doesn't perfectly mimic generate(), but tests processor state
        next_tokens = torch.argmax(processed_scores, dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_tokens], dim=1)
        # Keep using the same initial_scores for the *next* step's input for predictability
        # If your processor modifies scores in-place unexpectedly, clone here too.
        step_input_scores = initial_scores.clone()

    return all_scores


class TestExploreLogitsProcessorInitialization:

    def test_init_float_temp(self, mock_tokenizer):
        processor = ExploreLogitsProcessor(
            initial_seq_len=5, tokenizer=mock_tokenizer, temperature=0.7
        )
        assert processor._initial_temperature_val == 0.7
        assert not processor._is_batch_temp
        assert processor.temperature is None  # Initialized in __call__

    def test_init_list_temp(self, mock_tokenizer):
        processor = ExploreLogitsProcessor(
            initial_seq_len=5, tokenizer=mock_tokenizer, temperature=[0.7, 0.8]
        )
        assert torch.equal(
            processor._initial_temperature_val,
            torch.tensor([0.7, 0.8], dtype=torch.float32),
        )
        assert processor._is_batch_temp
        assert processor.temperature is None

    def test_init_tensor_temp(self, mock_tokenizer):
        temp_tensor = torch.tensor([0.7, 0.8], dtype=torch.float32)
        processor = ExploreLogitsProcessor(
            initial_seq_len=5, tokenizer=mock_tokenizer, temperature=temp_tensor
        )
        assert torch.equal(processor._initial_temperature_val, temp_tensor)
        assert processor._is_batch_temp
        assert processor.temperature is None

    def test_init_invalid_temp_type(self, mock_tokenizer):
        with pytest.raises(TypeError):
            ExploreLogitsProcessor(
                initial_seq_len=5,
                tokenizer=mock_tokenizer,
                temperature='invalid',
            )

    def test_init_state_variables(self, mock_tokenizer):
        processor = ExploreLogitsProcessor(
            initial_seq_len=5, tokenizer=mock_tokenizer, temperature=1.0
        )
        assert processor.temperature is None
        assert processor.source_tokens_tensor is None
        assert processor.target_tokens_tensor is None
        assert processor.current_step == 0
        assert processor.replacement_counts is None


class TestExploreLogitsProcessorStateManagement:

    def test_state_initialization_first_call(
        self, mock_tokenizer, initial_input_ids, initial_scores
    ):
        batch_size = initial_input_ids.shape[0]
        initial_len = initial_input_ids.shape[1]
        temp = [0.5, 1.0, 1.5]
        source = [mock_tokenizer._convert_token_to_id('source')]
        target = [mock_tokenizer._convert_token_to_id('target1')]

        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=temp,
            replace_source_tokens=source,
            replace_target_tokens=target,
        )

        # First call should initialize state
        processed_scores = processor(initial_input_ids, initial_scores)

        assert processor.current_step == 0
        assert processor.temperature is not None
        assert processor.temperature.shape == (batch_size,)
        assert torch.equal(
            processor.temperature, torch.tensor(temp, dtype=torch.float32)
        )
        assert processor.source_tokens_tensor is not None
        assert torch.equal(
            processor.source_tokens_tensor,
            torch.tensor(source, dtype=torch.long),
        )
        assert processor.target_tokens_tensor is not None
        assert torch.equal(
            processor.target_tokens_tensor,
            torch.tensor(target, dtype=torch.long),
        )
        assert processor.replacement_counts is not None
        assert torch.equal(
            processor.replacement_counts,
            torch.zeros(batch_size, dtype=torch.long),
        )

    def test_state_increment_step(
        self, mock_tokenizer, initial_input_ids, initial_scores
    ):
        initial_len = initial_input_ids.shape[1]
        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=1.0,
        )

        # First call
        processor(initial_input_ids, initial_scores.clone())
        assert processor.current_step == 0

        # Second call (simulate adding a token)
        longer_input_ids = torch.cat(
            [
                initial_input_ids,
                torch.zeros((initial_input_ids.shape[0], 1), dtype=torch.long),
            ],
            dim=1,
        )
        processor(longer_input_ids, initial_scores.clone())
        assert processor.current_step == 1

        # Third call
        even_longer_input_ids = torch.cat(
            [
                longer_input_ids,
                torch.zeros((initial_input_ids.shape[0], 1), dtype=torch.long),
            ],
            dim=1,
        )
        processor(even_longer_input_ids, initial_scores.clone())
        assert processor.current_step == 2

    def test_state_reset_on_initial_len(
        self, mock_tokenizer, initial_input_ids, initial_scores
    ):
        initial_len = initial_input_ids.shape[1]
        batch_size = initial_input_ids.shape[0]
        device = initial_input_ids.device  # Get device

        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=1.0,
            replace_source_tokens=[11],
            replace_target_tokens=[12],
            replace_max_per_seq=1,
            replace_prob=1.0,
            correctness_callback=correctness_always_fail,
            # *** Explicitly set explore params for this test ***
            explore_steps=1,  # Or any value > 0
            explore_skip=0,  # Or any value >= 0
        )

        # First call
        processor(
            initial_input_ids.clone().to(device),
            initial_scores.clone().to(device),
        )
        assert processor.current_step == 0
        assert torch.equal(
            processor.replacement_counts,
            torch.zeros(batch_size, dtype=torch.long, device=device),
        )

        # Second call (simulate adding token and replacement)
        longer_input_ids = torch.cat(
            [
                initial_input_ids,
                torch.tensor([[10], [10], [10]], device=device),
            ],
            dim=1,
        )
        processor(longer_input_ids, initial_scores.clone().to(device))
        assert processor.current_step == 1

        # Manually set replacement counts to simulate state change
        processor.replacement_counts = torch.ones(
            batch_size, dtype=torch.long, device=device
        )
        assert torch.equal(
            processor.replacement_counts,
            torch.ones(batch_size, dtype=torch.long, device=device),
        )

        # Third call, but with original input_ids length -> should reset state
        processor(
            initial_input_ids.clone().to(device),
            initial_scores.clone().to(device),
        )
        assert processor.current_step == 0  # Verify step reset

        # Check the counts AFTER the third call
        expected_counts = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        actual_counts = processor.replacement_counts

        # Optional debug prints (can remove after confirming fix)
        # print(f"\nDebug Info for test_state_reset_on_initial_len:")
        # print(f"  Device: {device}")
        # print(f"  Expected Counts: {expected_counts} (Device: {expected_counts.device})")
        # print(f"  Actual Counts after reset call: {actual_counts} (Device: {actual_counts.device})")

        assert torch.equal(
            actual_counts, expected_counts
        ), f"Replacement counts did not reset. Expected {expected_counts}, got {actual_counts}"


class TestExploreLogitsProcessorTemperature:

    @pytest.mark.parametrize('temp_val', [1.0, 0.7, 1.5])
    def test_scalar_temperature(
        self, mock_tokenizer, initial_input_ids, initial_scores, temp_val
    ):
        initial_len = initial_input_ids.shape[1]
        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=temp_val,
        )
        processed_scores = processor(initial_input_ids, initial_scores.clone())

        if temp_val == 1.0:
            # Should be close, allow for float precision
            torch.testing.assert_close(
                processed_scores, initial_scores, rtol=1e-6, atol=1e-6
            )
        else:
            expected_scores = initial_scores / temp_val
            torch.testing.assert_close(processed_scores, expected_scores)
            # Check relative order is maintained
            assert torch.argmax(processed_scores[0]) == torch.argmax(
                initial_scores[0]
            )
            assert torch.argmax(processed_scores[1]) == torch.argmax(
                initial_scores[1]
            )
            assert torch.argmax(processed_scores[2]) == torch.argmax(
                initial_scores[2]
            )

    def test_batch_temperature(
        self, mock_tokenizer, initial_input_ids, initial_scores
    ):
        initial_len = initial_input_ids.shape[1]
        temps = [0.5, 1.0, 2.0]
        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=temps,
        )
        processed_scores = processor(initial_input_ids, initial_scores.clone())

        expected_scores = initial_scores.clone()
        expected_scores[0] /= temps[0]
        expected_scores[1] /= temps[1]  # Stays same
        expected_scores[2] /= temps[2]

        torch.testing.assert_close(processed_scores, expected_scores)

    def test_zero_temperature(
        self, mock_tokenizer, initial_input_ids, initial_scores
    ):
        initial_len = initial_input_ids.shape[1]
        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=0.0,
        )
        processed_scores = processor(initial_input_ids, initial_scores.clone())

        # Find expected argmax for each row
        expected_argmax = torch.argmax(initial_scores, dim=-1)

        for i in range(initial_input_ids.shape[0]):
            # Check that only the argmax position has a high score
            assert torch.isneginf(processed_scores[i]).sum() == (
                initial_scores.shape[1] - 1
            )
            assert not torch.isneginf(processed_scores[i, expected_argmax[i]])
            # The value should be the boosted value (100.0 in the code)
            assert processed_scores[i, expected_argmax[i]] == 100.0

    def test_mixed_zero_batch_temperature(
        self, mock_tokenizer, initial_input_ids, initial_scores
    ):
        initial_len = initial_input_ids.shape[1]
        temps = [0.5, 0.0, 1.5]
        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=temps,
        )
        processed_scores = processor(initial_input_ids, initial_scores.clone())

        # Row 0: Scaled by 0.5
        torch.testing.assert_close(processed_scores[0], initial_scores[0] / 0.5)

        # Row 1: Zero temperature (greedy)
        expected_argmax_1 = torch.argmax(initial_scores[1])
        assert torch.isneginf(processed_scores[1]).sum() == (
            initial_scores.shape[1] - 1
        )
        assert not torch.isneginf(processed_scores[1, expected_argmax_1])
        assert processed_scores[1, expected_argmax_1] == 100.0

        # Row 2: Scaled by 1.5
        torch.testing.assert_close(processed_scores[2], initial_scores[2] / 1.5)


class TestExploreLogitsProcessorExploration:

    @pytest.mark.parametrize(
        'step, explore_skip, explore_steps, should_explore',
        [
            (0, 2, 5, False),  # Before skip
            (1, 2, 5, False),  # Before skip
            (2, 2, 5, True),  # Start explore
            (3, 2, 5, True),
            (6, 2, 5, True),  # Last explore step
            (7, 2, 5, False),  # After explore window
            (0, 0, 3, True),  # Explore immediately
            (2, 0, 3, True),  # Last explore step (immediate)
            (3, 0, 3, False),  # After explore window (immediate)
        ],
    )
    def test_exploration_activation(
        self,
        mock_tokenizer,
        initial_input_ids,
        initial_scores,
        step,
        explore_skip,
        explore_steps,
        should_explore,
    ):
        initial_len = initial_input_ids.shape[1]
        explore_k = 5  # Example K
        processor = ExploreLogitsProcessor(
            initial_seq_len=initial_len,
            tokenizer=mock_tokenizer,
            temperature=1.0,  # Keep temp simple
            explore_steps=explore_steps,
            explore_skip=explore_skip,
            explore_top_k=explore_k,
        )

        # Simulate steps
        all_processed_scores = simulate_steps(
            processor, initial_input_ids, initial_scores, step + 1
        )
        processed_scores_at_step = all_processed_scores[step]

        # Determine expected top-k if exploration happened
        if should_explore:
            # Calculate the effective K for this step (simplified decay for test)
            effective_steps = max(0, step - explore_skip)
            current_explore_top_k = max(
                2, int(explore_k * (0.9**effective_steps))
            )
            k = min(current_explore_top_k, initial_scores.shape[-1])

            # Find the actual top k indices from the *original* scores (before modification)
            # Note: In reality, temp/penalty are applied first, but we use temp=1, penalty=1 here
            _, top_k_indices = torch.topk(
                initial_scores, k=k, dim=-1
            )  # Use original scores for reference

            for i in range(initial_input_ids.shape[0]):
                # Check that only the top k indices have score 0.0
                expected_non_inf = torch.zeros_like(
                    processed_scores_at_step[i], dtype=torch.bool
                )
                expected_non_inf[top_k_indices[i]] = True

                actual_non_inf = ~torch.isneginf(processed_scores_at_step[i])
                assert torch.equal(actual_non_inf, expected_non_inf)
                # Check the non-inf values are indeed 0.0
                assert torch.all(
                    processed_scores_at_step[i][actual_non_inf] == 0.0
                )
        else:
            # No exploration, scores should be same as initial (temp=1)
            torch.testing.assert_close(
                processed_scores_at_step, initial_scores, rtol=1e-6, atol=1e-6
            )


class TestExploreLogitsProcessorReplacement:

    # --- Fixture for a processor configured for replacement testing ---
    @pytest.fixture
    def replacement_processor(self, mock_tokenizer):
        source_id = mock_tokenizer._convert_token_to_id('source')  # 11
        target1_id = mock_tokenizer._convert_token_to_id('target1')  # 12
        target2_id = mock_tokenizer._convert_token_to_id('target2')  # 13
        prevent_pattern = [
            mock_tokenizer._convert_token_to_id('prevent'),  # 8
            mock_tokenizer._convert_token_to_id('pattern'),  # 7
        ]
        return ExploreLogitsProcessor(
            initial_seq_len=5,  # Assume prompt length 5
            tokenizer=mock_tokenizer,
            temperature=1.0,  # Keep temp simple
            explore_steps=1,  # Explore step 0
            explore_skip=0,
            replace_source_tokens=[source_id],
            replace_target_tokens=[target1_id, target2_id],
            replace_prevent_patterns=[prevent_pattern],
            replace_prob=1.0,  # Ensure replacement happens if eligible
            replace_max_per_seq=2,  # Allow 2 replacements
            correctness_callback=correctness_always_fail,  # Assume incorrect by default
        )

    # --- Test Cases ---

    def test_replacement_eligible_basic(
        self, replacement_processor, initial_input_ids, initial_scores
    ):
        # Step 1: After exploration window, eligible for replacement
        processor = replacement_processor
        processor.current_step = 1  # Simulate being past explore step 0
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )  # Init state manually

        # Make scores predictable: ensure source token isn't highest, target tokens aren't highest initially
        scores = initial_scores.clone()
        scores[:, 11] = 5.0  # source
        scores[:, 12] = 4.0  # target1
        scores[:, 13] = 3.0  # target2

        # Input IDs don't contain prevent pattern, count=0, callback=fail -> eligible
        input_ids = torch.cat(
            [initial_input_ids, torch.tensor([[3], [5], [14]])], dim=1
        )  # Add non-prevent tokens

        processed_scores = processor(input_ids, scores)

        # Check source token penalized
        assert torch.all(torch.isneginf(processed_scores[:, 11]))

        # Check one target token boosted (score = original + 100.0)
        boosted1 = torch.isclose(processed_scores[:, 12], scores[:, 12] + 100.0)
        boosted2 = torch.isclose(processed_scores[:, 13], scores[:, 13] + 100.0)
        assert torch.all(
            boosted1 ^ boosted2
        )  # Exactly one should be boosted per row (XOR)

        # Check replacement count incremented
        assert torch.all(processor.replacement_counts == 1)

    def test_replacement_probability_zero(
        self, replacement_processor, initial_input_ids, initial_scores
    ):
        processor = replacement_processor
        processor.replace_prob = 0.0  # Set prob to 0
        processor.current_step = 1
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )

        scores = initial_scores.clone()
        input_ids = torch.cat(
            [initial_input_ids, torch.tensor([[3], [5], [14]])], dim=1
        )

        processed_scores = processor(input_ids, scores)

        # Scores should be unchanged (temp=1)
        torch.testing.assert_close(processed_scores, scores)
        # Count should be zero
        assert torch.all(processor.replacement_counts == 0)

    @patch('torch.rand')  # Mock torch.rand
    def test_replacement_probability_half(
        self,
        mock_rand,
        replacement_processor,
        initial_input_ids,
        initial_scores,
    ):
        processor = replacement_processor
        processor.replace_prob = 0.5
        processor.current_step = 1
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )

        # Mock rand to return values < 0.5 for first row, >= 0.5 for others
        mock_rand.return_value = torch.tensor(
            [0.2, 0.6, 0.7], device=initial_input_ids.device
        )

        scores = initial_scores.clone()
        scores[:, 11] = 5.0
        scores[:, 12] = 4.0
        scores[:, 13] = 3.0
        input_ids = torch.cat(
            [initial_input_ids, torch.tensor([[3], [5], [14]])], dim=1
        )

        processed_scores = processor(input_ids, scores)

        # Row 0: Should have replacement
        assert torch.isneginf(processed_scores[0, 11])
        boosted1 = torch.isclose(processed_scores[0, 12], scores[0, 12] + 100.0)
        boosted2 = torch.isclose(processed_scores[0, 13], scores[0, 13] + 100.0)
        assert boosted1 ^ boosted2
        assert processor.replacement_counts[0] == 1

        # Row 1, 2: Should NOT have replacement
        torch.testing.assert_close(processed_scores[1:], scores[1:])
        assert torch.all(processor.replacement_counts[1:] == 0)

    def test_replacement_max_count(
        self, replacement_processor, initial_input_ids, initial_scores
    ):
        processor = replacement_processor
        processor.replace_max_per_seq = 1  # Set max to 1
        processor.current_step = 1
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )

        scores = initial_scores.clone()
        scores[:, 11] = 5.0
        scores[:, 12] = 4.0
        scores[:, 13] = 3.0
        input_ids_step1 = torch.cat(
            [initial_input_ids, torch.tensor([[3], [5], [14]])], dim=1
        )

        # First call (step 1) - should replace
        processed_scores_1 = processor(input_ids_step1, scores.clone())
        assert torch.all(processor.replacement_counts == 1)
        assert torch.all(
            torch.isneginf(processed_scores_1[:, 11])
        )  # Source penalized

        # Second call (step 2) - should NOT replace (max reached)
        processor.current_step = 2
        input_ids_step2 = torch.cat(
            [input_ids_step1, torch.tensor([[3], [5], [14]])], dim=1
        )
        processed_scores_2 = processor(input_ids_step2, scores.clone())

        # Scores should be same as input scores (temp=1)
        torch.testing.assert_close(processed_scores_2, scores)
        # Count should remain 1
        assert torch.all(processor.replacement_counts == 1)

    def test_replacement_prevent_pattern_present(
        self, replacement_processor, initial_input_ids, initial_scores
    ):
        processor = replacement_processor
        processor.current_step = 1
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )

        scores = initial_scores.clone()
        # Add the prevent pattern [8, 7] to the generated part of row 0
        input_ids = initial_input_ids.clone()
        input_ids = torch.cat(
            [input_ids, torch.tensor([[8], [5], [14]])], dim=1
        )  # Add first part
        input_ids = torch.cat(
            [input_ids, torch.tensor([[7], [5], [14]])], dim=1
        )  # Add second part

        processed_scores = processor(input_ids, scores.clone())

        # Row 0 should NOT be replaced due to pattern
        torch.testing.assert_close(processed_scores[0], scores[0])
        assert processor.replacement_counts[0] == 0

        # Row 1, 2 should be replaced (no pattern)
        assert torch.all(torch.isneginf(processed_scores[1:, 11]))
        assert torch.all(processor.replacement_counts[1:] == 1)

    def test_replacement_correctness_pass(
        self, replacement_processor, initial_input_ids, initial_scores
    ):
        processor = replacement_processor
        processor.correctness_callback = (
            correctness_always_pass  # Set callback to always pass
        )
        processor.current_step = 1
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )

        scores = initial_scores.clone()
        input_ids = torch.cat(
            [initial_input_ids, torch.tensor([[3], [5], [14]])], dim=1
        )

        processed_scores = processor(input_ids, scores.clone())

        # No replacement should happen because callback returns >= 1.0
        torch.testing.assert_close(processed_scores, scores)
        assert torch.all(processor.replacement_counts == 0)

    def test_replacement_correctness_conditional(
        self, mock_tokenizer, initial_input_ids, initial_scores
    ):
        # Use the conditional callback
        source_id = mock_tokenizer._convert_token_to_id('source')  # 11
        target1_id = mock_tokenizer._convert_token_to_id('target1')  # 12
        target2_id = mock_tokenizer._convert_token_to_id('target2')  # 13
        fail_id = mock_tokenizer._convert_token_to_id('fail')  # 22
        correct_id = mock_tokenizer._convert_token_to_id('correct')  # 9

        processor = ExploreLogitsProcessor(
            initial_seq_len=5,
            tokenizer=mock_tokenizer,
            temperature=1.0,
            explore_steps=1,
            explore_skip=0,
            replace_source_tokens=[source_id],
            replace_target_tokens=[target1_id, target2_id],
            replace_prob=1.0,
            replace_max_per_seq=1,
            correctness_callback=correctness_if_contains_fail,  # Use conditional callback
        )
        processor.current_step = 1
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )

        scores = initial_scores.clone()
        scores[:, 11] = 5.0
        scores[:, 12] = 4.0
        scores[:, 13] = 3.0

        # Row 0: Add "fail" token -> should replace
        # Row 1: Add "correct" token -> should NOT replace
        # Row 2: Add "another" token -> should replace (contains no "fail")
        input_ids = torch.cat(
            [initial_input_ids, torch.tensor([[fail_id], [correct_id], [14]])],
            dim=1,
        )

        processed_scores = processor(input_ids, scores.clone())

        # Row 0: Replaced
        assert torch.isneginf(processed_scores[0, 11])
        assert processor.replacement_counts[0] == 1

        # Row 1: Not replaced
        torch.testing.assert_close(processed_scores[1], scores[1])
        assert processor.replacement_counts[1] == 0

        # Row 2: Replaced (callback returns 1.0 only if "fail" is present, otherwise < 1.0 -> incorrect)
        # Correction: The callback returns 0.0 IF "fail" is present, 1.0 otherwise.
        # So Row 0 should replace, Row 1 should NOT, Row 2 should NOT.
        # Let's adjust the expectation based on the callback logic:
        # Row 0: Contains "fail" -> callback returns 0.0 -> incorrect -> replace
        # Row 1: Contains "correct" -> callback returns 1.0 -> correct -> NO replace
        # Row 2: Contains "another" -> callback returns 1.0 -> correct -> NO replace
        torch.testing.assert_close(processed_scores[2], scores[2])
        assert processor.replacement_counts[2] == 0

    def test_replacement_step_condition(
        self, replacement_processor, initial_input_ids, initial_scores
    ):
        # Replacement should only happen AFTER explore window (step >= explore_skip + explore_steps)
        # In replacement_processor, explore_skip=0, explore_steps=1. So replace starts at step 1.
        processor = replacement_processor
        processor.current_step = 0  # Still in exploration step
        processor._initialize_state(
            initial_input_ids.shape[0], initial_input_ids.device
        )

        scores = initial_scores.clone()
        input_ids = (
            initial_input_ids.clone()
        )  # Use initial length, generated part is empty

        processed_scores = processor(input_ids, scores.clone())

        # Exploration should happen, but NO replacement yet
        # Check exploration effect (top-k = 0.0, others -inf)
        k = min(processor.explore_top_k, scores.shape[-1])
        _, top_k_indices = torch.topk(scores, k=k, dim=-1)
        for i in range(initial_input_ids.shape[0]):
            expected_non_inf = torch.zeros_like(
                processed_scores[i], dtype=torch.bool
            )
            expected_non_inf[top_k_indices[i]] = True
            actual_non_inf = ~torch.isneginf(processed_scores[i])
            assert torch.equal(actual_non_inf, expected_non_inf)
            assert torch.all(processed_scores[i][actual_non_inf] == 0.0)

        # Check replacement count is still zero
        assert torch.all(processor.replacement_counts == 0)
