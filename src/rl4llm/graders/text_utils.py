import re
from collections import Counter


def has_irregular_words(text: str, min_length: int = 20) -> bool:
    """Checks for very long irregular words while ignoring LaTeX and equations.

    Args:
        text (str): Input text to check for irregular words
        min_length (int): Threshold for minimum word length

    Returns:
        bool: True if long words are found (excluding LaTeX/equations), False otherwise
    """
    # Define patterns to detect LaTeX and mathematical expressions
    latex_patterns = [
        r'\\\w+',  # LaTeX commands (e.g., \textbf, \frac)
        r'\$.*?\$',  # Inline math mode $...$
        r'\\\[.*?\\\]',  # Display math mode \[...\]
        r'\{.*?\}',  # Curly brace content often used in LaTeX
    ]

    # First, mask LaTeX/equation content to prevent false positives
    masked_text = text
    for pattern in latex_patterns:
        masked_text = re.sub(pattern, ' MASKED ', masked_text, flags=re.DOTALL)

    # Split the text into words by whitespace
    words = masked_text.split()

    for word in words:
        # Skip if it's our mask or obviously not a real word
        if word == 'MASKED' or not any(c.isalpha() for c in word):
            continue

        # Remove common punctuation from consideration in length
        cleaned_word = re.sub(r'[.,!?;:()\'"]', '', word)

        # Check if the cleaned word is longer than the specified length
        if len(cleaned_word) >= min_length:
            return True

    return False


def has_repetitions(text: str, ngram_size: int = 8, threshold: int = 3) -> bool:
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
