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


def has_repetitions(text: str, repetition_threshold: int = 3) -> bool:
    """
    Detect meaningful content repetitions while ignoring mathematical and
    problem-solving structures.

    Args:
        text (str): Input text to check for repetitions
        repetition_threshold (int): Threshold to the repetition

    Returns:
        bool: True if meaningful repetitions are found, False otherwise
    """

    assert repetition_threshold > 2

    # First, remove all mathematical expressions
    text = re.sub(r'\\\[.*?\\\]', '', text)  # Remove display math
    text = re.sub(r'\\\(.*?\\\)', '', text)  # Remove inline math
    text = re.sub(r'\\text{.*?}', '', text)  # Remove \text commands
    text = re.sub(r'\\boxed{.*?}', '', text)  # Remove \boxed commands

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]\s*', text) if s.strip()]

    def is_math_context(sentence):
        """Check if sentence is in mathematical/explanatory context"""
        math_indicators = [
            'calculate',
            'determine',
            'find',
            'equals',
            'total',
            'first',
            'next',
            'then',
            'therefore',
            'thus',
            'sum',
            'difference',
            'product',
            'quotient',
            'subtract',
            'add',
            'multiply',
            'divide',
        ]
        return any(indicator in sentence.lower() for indicator in math_indicators)

    def get_substantial_phrases(sentence, min_words=6):
        words = sentence.split()
        if len(words) < min_words:
            return []

        phrases = []
        for i in range(len(words) - min_words + 1):
            # For mathematical contexts, only consider longer phrases
            min_length = 10 if is_math_context(sentence) else 6

            for length in range(min_length, min(len(words) - i + 1, 15)):
                phrase = ' '.join(words[i : i + length])
                # Add length requirement for math contexts
                if (len(phrase) >= 40 and is_math_context(sentence)) or (len(phrase) >= 20 and not is_math_context(sentence)):
                    phrases.append(phrase)
        return phrases

    # Get all substantial phrases
    all_phrases = []
    for sentence in sentences:
        all_phrases.extend(get_substantial_phrases(sentence))

    # Count phrases
    phrase_counter = Counter(all_phrases)

    # Different thresholds based on context
    for phrase, count in phrase_counter.items():
        if count > repetition_threshold:  # Need at least 3 occurrences
            if is_math_context(phrase):
                # For math context, need more repetitions and longer phrases
                if count > repetition_threshold + 1 and len(phrase.split()) >= 10:
                    return True
            else:
                # For normal context, standard threshold
                if len(phrase.split()) >= 6:
                    return True

    # Check for exact sentence repetitions
    sentence_counter = Counter(sentences)
    repetitions = [
        count > repetition_threshold
        for sentence, count in sentence_counter.items()
        if len(sentence.split()) >= 6 and not is_math_context(sentence)
    ]
    exact_repetitions = any(repetitions)
    return exact_repetitions
