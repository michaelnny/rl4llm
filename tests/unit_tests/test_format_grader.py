from typing import List, Tuple

import pytest

from rl4llm.graders.format_grader import format_structure_grader


def test_format_structure_grader():
    """Test that completion starting with code format (```) results in a score of -0.5"""
    completion = "```print('Hello, world!')```"
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = r'\\This is some code'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = 'This is some text ```'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = r'\boxed{This is the answer}'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    completion = '123 This is a number'
    assert format_structure_grader(completion, seq_length=100) == -0.5

    # too short
    completion = 'Short completion'
    assert format_structure_grader(completion, seq_length=10) == -0.5

    # check repetitions
    completion = (
        'This is a very long completion that is definitely more than 100 words long. It should pass the length check.' * 10
    )
    assert format_structure_grader(completion, seq_length=100) == -0.5
