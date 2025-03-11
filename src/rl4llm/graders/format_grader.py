import re

from .text_utils import has_repetitions

xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def validate_xml_structure(completion_text: str) -> bool:
    """Checking for `<think></think><answer></answer>` format with non-empty content."""
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

    # if has_repetitions(think_content, 12, 5):
    #     return False

    return True


def format_structure_grader(completion: str, xml_format: bool = True) -> float:
    """Checks for general rules like XML format, repetition  etc"""
    completion_text = completion.strip()

    if has_repetitions(completion_text, 12, 5):
        return -1.0

    if xml_format:
        if validate_xml_structure(completion_text):
            return 1.0
        else:
            return 0.0
    else:
        return 0.0
