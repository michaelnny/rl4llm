import re
import string
from collections import Counter
from typing import Dict, List, Tuple


class RepetitionDetector:
    """
    Detects if text has n-gram repetition.

    Args:
        ngram_size: Size of the n-grams.
        ngram_threshold: Threshold for detecting repetition of n-grams.
    """

    def __init__(self, ngram_size: int = 10, ngram_threshold: int = 3):
        assert ngram_size >= 3
        assert ngram_threshold >= 2

        self.ngram_size = ngram_size
        self.ngram_threshold = ngram_threshold

    @staticmethod
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

    def detect_ngram_repetition(self, text: str) -> Tuple[bool, Dict[str, int]]:
        """
        Detect repeated n-grams in the text, with improved normalization and n-gram extraction.
        """
        if len(text.split()) < self.ngram_size:
            return False, {}  # Not enough words for n-gram extraction

        # Extract n-grams
        ngrams = self.get_ngrams(text, self.ngram_size)

        if not ngrams:
            return False, {}

        # Count occurrences of each n-gram
        ngram_counts = Counter(ngrams)
        repetitions = {' '.join(ngram): count for ngram, count in ngram_counts.items() if count > self.ngram_threshold}

        return bool(repetitions), repetitions

    def check_repetition(self, text: str) -> bool:
        """Check if the input text contains n-gram repetition."""
        repetition_detected, _ = self.detect_ngram_repetition(text)
        return repetition_detected
