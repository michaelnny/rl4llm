import re
from collections import Counter


def has_irregular_words(text: str, min_length: int = 20) -> bool:
    """Checks for very long irregular words.

    Args:
        text (str): Input text to check for repetitions
        min_length (int): Threshold to minimum word length

    Returns:
        bool: True if long words are found, False otherwise
    """
    # Split the text into words (ignoring punctuation)
    words = re.findall(r'\b\w+\b', text)

    for word in words:
        # Check if the word is longer than the specified length
        if len(word) >= min_length:
            return True

    return False


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
        if count > threshold:
            return True

    return False
