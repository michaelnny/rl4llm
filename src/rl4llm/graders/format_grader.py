import re

from .text_utils import has_irregular_words, has_repetitions

xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def check_invalid_format(text: str) -> bool:
    # Avoid processing if text is empty after stripping
    stripped_text = text.strip()
    if not stripped_text:
        return True

    # Check for invalid formatting
    invalid_conditions = [
        stripped_text.startswith(('```', '`')),  # start with code block
        stripped_text.endswith(('```', '`')),
        stripped_text.startswith(('\\\\', '\\boxed', 'The answer is', 'The correct answer is')),  # start with direct answer
        stripped_text[0].isdigit(),  # start with numerical answer, or bullet point
        has_repetitions(stripped_text),  # check for n-gram repetitions
        # has_irregular_words(stripped_text, 25),  # check for very long words, not working so well for latex and Chinese
    ]

    return any(invalid_conditions)


def format_structure_grader(completion: str, seq_length: int, min_length: int = 100, xml_format: bool = False) -> float:
    """Checks for general rules like format, length etc"""
    score = 0.0
    completion_text = completion.strip()

    if xml_format:
        match = re.match(xml_pattern, completion_text, re.DOTALL | re.MULTILINE)
        if not match:
            return -0.5  # if XML doesn't match

        think_content = match.group(1).strip() if match.group(1) else ''
        answer_content = match.group(2).strip() if match.group(2) else ''

        if (
            not think_content
            or not answer_content
            or check_invalid_format(think_content)
            or check_invalid_format(answer_content)
        ):
            return -0.5

    else:
        if seq_length < min_length or check_invalid_format(completion_text):
            return -0.5

    # no conditions are violated
    return score
