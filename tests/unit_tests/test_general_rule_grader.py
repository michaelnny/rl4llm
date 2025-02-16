import pytest

from rl4llm.graders.general_grader import general_rule_grader


def test_code_format_start():
    """Test that completion starting with code format (```) results in a score of -0.5"""
    completion = "```print('Hello, world!')```"
    assert general_rule_grader(completion) == -0.5


def test_code_format_start_with_backslash():
    """Test that completion starting with '\\' results in a score of -0.5"""
    completion = r'\\This is some code'
    assert general_rule_grader(completion) == -0.5


def test_code_format_end():
    """Test that completion ending with code format (```) results in a score of -0.5"""
    completion = 'This is some text ```'
    assert general_rule_grader(completion) == -0.5


def test_boxed_format_start():
    """Test that completion starting with '\\boxed' results in a score of -0.5"""
    completion = r'\boxed{This is the answer}'
    assert general_rule_grader(completion) == -0.5


def test_numeric_start():
    """Test that completion starting with a number results in a score of -0.5"""
    completion = '123 This is a number'
    assert general_rule_grader(completion) == -0.5


def test_short_completion():
    """Test that completion shorter than 100 characters results in a score of -0.5"""
    completion = 'Short completion'
    assert general_rule_grader(completion) == -0.5


def test_long_completion():
    """Test that a sufficiently long completion (>= 100 characters) results in a score of 0.0"""
    completion = (
        'This is a very long completion that is definitely more than 100 words long. It should pass the length check.' * 10
    )
    assert general_rule_grader(completion) == 0.0
