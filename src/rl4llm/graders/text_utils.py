from collections import Counter


def check_repetition(
    text: str,
    min_ngram: int = 8,
    max_ngram: int = 16,
    ngram_threshold: int = 6,
    min_sentence_words: int = 8,
    sentence_threshold: int = 4,
) -> bool:
    """
    Checks for repetition patterns in the given text.

    Returns True if any n-gram (of lengths between min_ngram and max_ngram)
    repeats more than ngram_threshold times, or if any sentence (with more than
    min_sentence_words words) repeats more than sentence_threshold times.
    """

    if not (min_ngram > 5 and min_ngram < max_ngram):
        raise ValueError(
            'min_ngram must be greater than 5 and less than max_ngram.'
        )
    if not (
        ngram_threshold > 5
        and min_sentence_words > 5
        and sentence_threshold > 3
    ):
        raise ValueError(
            'ngram_threshold and min_sentence_words must be > 5, and sentence_threshold must be > 3.'
        )

    text_lower = text.lower()

    # Check repeated n-grams
    words = text_lower.split()
    for n in range(min_ngram, max_ngram + 1):
        # Create n-grams and count occurrences
        ngram_counts = Counter(
            tuple(words[i : i + n]) for i in range(len(words) - n + 1)
        )
        if any(count >= ngram_threshold for count in ngram_counts.values()):
            return True

    # Check repeated sentences
    sentences = [
        line.strip()
        for line in text_lower.splitlines()
        if len(line.split()) > min_sentence_words
    ]
    sentence_counts = Counter(sentences)
    if any(count >= sentence_threshold for count in sentence_counts.values()):
        return True

    return False
