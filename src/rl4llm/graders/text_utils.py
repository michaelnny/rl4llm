import re
import string
from collections import Counter


def has_incoherent_content(text):
    """
    Returns True if the text appears incoherent and False if it looks like a properly structured explanation.

    This version first removes any HTML-like tags (e.g. , <answer>) by replacing them with newlines.
    Then it splits the text into candidate lines (using newlines if available, else falls back on sentence splitting).

    For each candidate that isn’t very short (fewer than 3 words for non–math lines), it “cleans” the line
    by stripping out any non–word characters at its start. If the first (nonempty) character is:
      – a digit, then it is acceptable;
      – an ASCII letter, then it must be uppercase (non–ASCII letters like Chinese are accepted);
      – otherwise, the line is rejected.

    Next, for non–math lines the ratio of digits to letters is checked: if digits overwhelm letters (ratio 0.4), the line is rejected.

    Finally, if fewer than 60% of candidate lines are acceptable the text is labeled as incoherent.
    """
    # Remove or break on any HTML-style tags (like  or <answer>):
    text = re.sub(r'<[^>]+>', '\n', text)

    def is_math_mode(line):
        # If the line starts with a math delimiter (commonly used in LaTeX) or contains a LaTeX command, mark it as math mode.
        if line.startswith(r'\[') or line.startswith(r'\(') or line.startswith(r'\]') or line.startswith(r'\)'):
            return True
        if r'\text' in line:
            return True
        # If the entire line consists solely of digits and simple math symbols, treat it as math mode.
        if re.fullmatch(r'[\d\.\+\-\=\/\*\(\)\s]+', line):
            return True
        return False

    # First try splitting the text according to newlines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # If that produces only a single candidate, fall back on splitting by sentence punctuation.
    if len(lines) < 2:
        lines = re.split(r'(?<=[.!?])\s+', text)
        lines = [line.strip() for line in lines if line.strip()]

    total = 0
    valid = 0
    for line in lines:
        # For non–math lines, ignore very short lines (fewer than 3 words)
        words = line.split()
        if not is_math_mode(line) and len(words) < 3:
            continue
        total += 1

        # Remove any leading punctuation (except for backslashes needed for LaTeX)
        cleaned = re.sub(r'^[^\w\u4e00-\u9fff\\]+', '', line)
        if not cleaned:
            continue

        # If in math mode, count the line as valid and move to the next.
        if is_math_mode(cleaned):
            valid += 1
            continue

        # Check the first-character rule.
        first = cleaned[0]
        if first.isdigit():
            pass
        elif first.isalpha():
            if first.isascii() and not first.isupper():
                continue  # Reject if ASCII letter is not uppercase.
        else:
            continue  # Reject if the first character isn’t alphanumeric.

        # Count letters and digits.
        letter_count = sum(1 for c in cleaned if c.isalpha())
        digit_count = sum(1 for c in cleaned if c.isdigit())
        # For non-math text having letters, if digits overwhelm letters, skip this line.
        if letter_count > 0 and (digit_count / letter_count) > 0.4:
            continue
        if letter_count == 0 and not is_math_mode(cleaned):
            continue

        valid += 1

    # When no candidate lines are found, assume there is no incoherent content.
    if total == 0:
        return False
    fraction = valid / total
    # Label the text as incoherent if fewer than 60% of candidate lines pass.
    return fraction < 0.6


def has_repetitions(text: str, ngram_size: int = 10, threshold: int = 3) -> bool:
    """
    Checks if there are N-gram repetition.

    Args:
        text: the raw text to check
        ngram_size: size of the n-grams
        threshold: threshold for minimum number of repetitions to consider as true

    Returns:
        bool indicate if there are repetitions detected in the text.
    """

    assert ngram_size > 3
    assert threshold > 2

    def zipngram(text: str, n_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(n_size)])

    ngram_counts = Counter()
    for ng in zipngram(text, ngram_size):
        ngram_counts[ng] += 1

    for count in ngram_counts.values():
        if count >= threshold:
            return True

    return False
