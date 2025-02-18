import re

xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def check_has_invalid_format(text: str) -> bool:
    try:
        is_invalid = False
        if not text:  # empty text
            is_invalid = True
        elif text.strip().startswith('```') or text.strip().endswith('```'):  # start with code
            is_invalid = True
        elif (
            text.strip().startswith(r'\\')
            or text.strip().startswith(r'\boxed')
            or text.strip().startswith('The answer is')
            or text.strip().startswith('The correct answer is')
        ):  # start with answer block
            is_invalid = True
        elif text.strip()[0].isdigit():  # start with numerical answer or bullet point
            is_invalid = True

        return is_invalid
    except Exception as _e:
        return True


def format_structure_grader(completion: str, seq_length: int, min_length: int = 100, xml_format: bool = False) -> float:
    """Checks for general rules like format, length etc"""
    score = 0.0
    completion_text = completion.strip()

    if xml_format:
        # DeepSeek R1 style XML format
        match = re.match(xml_pattern, completion_text, re.DOTALL | re.MULTILINE)
        if not match:  # If the XML format doesn't match
            score = -0.5
        else:
            # Extract content inside <think> and <answer> tags
            think_content = match.group(1).strip() if match.group(1) else ''
            answer_content = match.group(2).strip() if match.group(2) else ''

            # Check that the content within <think> and <answer> is not empty
            if not think_content:
                score = -0.5
            if not answer_content:
                score = -0.5

            if check_has_invalid_format(think_content) or check_has_invalid_format(answer_content):
                score = -0.5
    else:
        if seq_length < min_length:
            score = -0.5
        elif check_has_invalid_format(completion_text):
            score = -0.5

    return score
