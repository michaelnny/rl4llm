import re

from .text_utils import has_repetitions

xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def validate_xml_structure(completion_text: str) -> bool:
    """Checking for `<think></think><answer></answer>` format with non-empty content."""
    # Strip any code block markers
    text = re.sub(r'```.*?\n|```', '', completion_text, flags=re.DOTALL)

    # Count tag occurrences
    think_open = text.count('<think>')
    think_close = text.count('</think>')
    answer_open = text.count('<answer>')
    answer_close = text.count('</answer>')
    if think_open != 1 or think_close != 1 or answer_open != 1 or answer_close != 1:
        return False  # Wrong number of tags

    # Check basic structure
    if not text.startswith('<think>') or not text.endswith('</answer>'):
        return False

    # Validate structure with regex
    xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
    match = re.match(xml_pattern, text, re.DOTALL | re.MULTILINE)

    if not match:
        return False  # No valid pair or incorrect structure

    think_content = match.group(1).strip()
    answer_content = match.group(2).strip()

    if not think_content or not answer_content:
        return False  # Empty content

    # Check for forbidden tags in content
    forbidden_tags = r'<think>|</think>|<answer>|</answer>'
    if re.search(forbidden_tags, think_content) or re.search(forbidden_tags, answer_content):
        return False

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
