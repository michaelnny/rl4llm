import re
from collections import Counter
from typing import List, Tuple

xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'


def get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from text with better normalization and tokenization."""
    # Normalize text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation using regex

    # Tokenize the text, accounting for multiple spaces or tab characters
    words = re.split(r'\s+', text.strip())

    # Generate n-grams
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return ngrams


def has_repetitions(text: str, ngram_size: int = 10, repetition_threshold: int = 3) -> bool:
    """
    Detect if text has obvious repetitions of either complete sentences
    or n-grams above a specified threshold.

    Args:
        text (str): Input text to analyze
        ngram_size (int): Size of n-grams to check
        repetition_threshold (int): Minimum number of repetitions to consider

    Returns:
        bool: True if repetitions are found, False otherwise
    """

    if len(text.split()) < ngram_size:
        return False

    # Extract n-grams
    ngrams = get_ngrams(text, ngram_size)

    if not ngrams:
        return False

    # Count occurrences of each n-gram
    ngram_counts = Counter(ngrams)
    repetitions = {' '.join(ngram): count for ngram, count in ngram_counts.items() if count > repetition_threshold}

    return bool(repetitions)


def has_invalid_format(text: str) -> bool:
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

            if has_invalid_format(think_content) or has_invalid_format(answer_content):
                score = -0.5

            if has_repetitions(think_content) or has_repetitions(answer_content):
                score = -0.5
    else:
        if has_repetitions(completion_text):
            score = -0.5
        elif seq_length < min_length:
            score = -0.5
        elif has_invalid_format(completion_text):
            score = -0.5

    return score
