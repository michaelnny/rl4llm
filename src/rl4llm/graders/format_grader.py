import re

from .text_utils import has_repetitions, has_incoherent_content

xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def format_structure_grader(completion: str, xml_format: bool = True) -> float:
    """Checks for general rules like XML format, repetition  etc"""
    text = completion.strip()

    if xml_format:
        # Strip any code block markers
        text = re.sub(r'```.*?\n|```', '', text, flags=re.DOTALL)

        # Validate structure with regex
        xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
        match = re.match(xml_pattern, text, re.DOTALL | re.MULTILINE)

        if not match:
            if has_repetitions(text, 10, 4):
                return -1.0
            return 0.0

        think_content = match.group(1).strip()
        answer_content = match.group(2).strip()

        if has_repetitions(think_content, 15, 5) or has_repetitions(answer_content, 8, 3):
            return -1.0

        # TODO: this rule-based solution is not reliable as it flags lots of false positives/negatives
        # probably need to train a small classifier model to detect incoherent content
        # if has_incoherent_content(think_content) or has_incoherent_content(answer_content):
        #     return -1.0

        if not think_content or not answer_content:
            return 0.0

        # Count tag occurrences
        think_open = text.count('<think>')
        think_close = text.count('</think>')
        answer_open = text.count('<answer>')
        answer_close = text.count('</answer>')
        if think_open != 1 or think_close != 1 or answer_open != 1 or answer_close != 1:
            return 0.0

        # Check basic structure
        if not text.startswith('<think>') or not text.endswith('</answer>'):
            return 0.0

        # Check for forbidden tags in content
        forbidden_tags = r'<think>|</think>|<answer>|</answer>'
        if re.search(forbidden_tags, think_content) or re.search(forbidden_tags, answer_content):
            return 0.0

        return 1.0
    else:
        if has_repetitions(text, 12, 5):
            return -1.0

        return 0.0
