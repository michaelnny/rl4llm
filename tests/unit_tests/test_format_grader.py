import pytest

from rl4llm.graders.format_grader import format_structure_grader, validate_xml_structure


# Test cases for validate_xml_structure
def test_valid_structure():
    xml_input = '<think>What is the capital of France?</think><answer>Paris</answer>'
    assert validate_xml_structure(xml_input) is True


def test_invalid_structure_missing_think():
    xml_input = '<answer>Paris</answer>'
    assert validate_xml_structure(xml_input) is False


def test_invalid_structure_missing_answer():
    xml_input = '<think>What is the capital of France?</think>'
    assert validate_xml_structure(xml_input) is False


def test_invalid_structure_extra_think_tags():
    xml_input = '<think>What is the capital of France?</think><think>Where is Eiffel Tower?</think><answer>Paris</answer>'
    assert validate_xml_structure(xml_input) is False


def test_invalid_structure_extra_answer_tags():
    xml_input = '<think>What is the capital of France?</think><answer>Paris</answer><answer>London</answer>'
    assert validate_xml_structure(xml_input) is False


def test_empty_think_content():
    xml_input = '<think></think><answer>Paris</answer>'
    assert validate_xml_structure(xml_input) is False


def test_empty_answer_content():
    xml_input = '<think>What is the capital of France?</think><answer></answer>'
    assert validate_xml_structure(xml_input) is False


def test_repetition_in_answer():
    xml_input = '<think>What is the capital of France?</think><answer>The answer for the capital of France is Paris. The answer for the capital of France is Paris. The answer for the capital of France is Paris. The answer for the capital of France is Paris.</answer>'
    assert validate_xml_structure(xml_input) is False


def test_invalid_with_spaces_in_tags():
    xml_input = '   <think>What is the capital of France?</think>   <answer>Paris</answer>  '
    assert validate_xml_structure(xml_input) is False


def test_multiple_spaces_in_answer():
    xml_input = '<think>What is the capital of France?</think><answer>    Paris   </answer>'
    assert validate_xml_structure(xml_input) is True


# Test cases for format_structure_grader
def test_valid_score():
    xml_input = '<think>What is the capital of France?</think><answer>Paris</answer>'
    assert format_structure_grader(xml_input) == 1.0


def test_invalid_score_missing_think():
    xml_input = '<answer>Paris</answer>'
    assert format_structure_grader(xml_input) == 0.0


def test_invalid_score_missing_answer():
    xml_input = '<think>What is the capital of France?</think>'
    assert format_structure_grader(xml_input) == 0.0


def test_invalid_score_multiple_answer_tags():
    xml_input = '<think>What is the capital of France?</think><answer>Paris</answer><answer>London</answer>'
    assert format_structure_grader(xml_input) == 0.0


def test_invalid_score_empty_think():
    xml_input = '<think></think><answer>What is the capital of France? London</answer>'
    assert format_structure_grader(xml_input) == 0.0


def test_invalid_score_empty_answer():
    xml_input = '<think>What is the capital of France? London</think><answer></answer>'
    assert format_structure_grader(xml_input) == 0.0
