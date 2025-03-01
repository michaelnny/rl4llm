import re

from .text_utils import has_irregular_words, has_repetitions

xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def validate_xml_structure(completion_text: str) -> bool:
    # Strip any code block markers
    text = re.sub(r'```.*?\n|```', '', completion_text, flags=re.DOTALL)

    # Count the number of tag pairs
    think_tags = re.findall(r'<think>', text)
    answer_tags = re.findall(r'<answer>', text)

    # If there's more than one pair, reject
    if len(think_tags) > 1 or len(answer_tags) > 1:
        return False  # Multiple tag pairs found

    if not completion_text.startswith('<think>') or not completion_text.endswith('</answer>'):
        return False

    # Check if a single valid pair exists with non-empty content
    xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
    match = re.match(xml_pattern, text, re.DOTALL | re.MULTILINE)

    if not match:
        return False  # No valid pair or incorrect structure

    # Check that content inside tags is not empty (after stripping whitespace)
    think_content = match.group(1).strip()
    answer_content = match.group(2).strip()

    if not think_content or not answer_content:
        return False  # Empty content in one or both tags

    # if has_repetitions(think_content) or has_repetitions(answer_content):
    if has_repetitions(answer_content):
        return False

    return True


def format_structure_grader(completion: str) -> float:
    """Checks for general rules like format, length etc"""
    score = 1.0

    if not validate_xml_structure(completion.strip()):
        return 0.0

    return score
