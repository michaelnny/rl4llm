import re
from collections import Counter


def has_incoherent_content(text):

    # Define English stop words
    stop_words = {
        'the',
        'a',
        'an',
        'and',
        'or',
        'but',
        'in',
        'on',
        'at',
        'to',
        'for',
        'of',
        'with',
        'by',
        'from',
        'up',
        'about',
        'into',
        'over',
        'after',
        'beneath',
        'under',
        'above',
        'is',
        'are',
        'was',
        'were',
        'be',
        'been',
        'am',
        'has',
        'have',
        'had',
        'do',
        'does',
        'did',
        'will',
        'would',
        'shall',
        'should',
        'can',
        'could',
        'may',
        'might',
        'must',
    }

    # Strip HTML-like tags
    stripped_text = re.sub(r'<[^>]*>', '', text)
    total_length = len(text) + 0.001

    # Statistical measures
    punctuation_count = len(re.findall(r'[,./\\;:!?+\-*=&|<>{}[\]()]', text))
    digit_count = len(re.findall(r'\d', text))
    space_count = len(re.findall(r'\s', text))
    letter_count = len(re.findall(r'[a-zA-Z\u4e00-\u9fff\u3040-\u30ff\u0400-\u04FF]', text))

    punctuation_ratio = punctuation_count / total_length
    digit_ratio = digit_count / total_length
    space_ratio = space_count / total_length

    # Existing pattern checks
    excessive_numeric_pattern = r'(?:\d+\s*[,./\\\+\-\*]\s*){3,}'
    numeric_pattern_matches = len(re.findall(excessive_numeric_pattern, text))
    symbol_cluster = r'[\\\/\|\*\+]{3,}'
    symbol_cluster_matches = len(re.findall(symbol_cluster, text))
    short_clusters = r'(?:\b\w{1,2}\b\s+){5,}'
    short_cluster_matches = len(re.findall(short_clusters, text))

    lines = text.split('\n')
    unusual_line_count = sum(1 for line in lines if re.search(r'^\s*[\d\s\+\-\*\/\\]{5,}', line))
    unusual_line_ratio = unusual_line_count / max(len(lines), 1)

    # Word-based analysis
    words = re.findall(r'\b\w+\b', stripped_text.lower())
    if words:
        unique_words = set(words)
        unique_word_ratio = len(unique_words) / len(words)
        word_freq = Counter(words)
        stop_word_freq = sum(word_freq[word] for word in stop_words if word in word_freq) / len(words)
        # Long unique words (likely nonsense)
        long_unique_words = sum(1 for word in unique_words if len(word) > 6)
        long_unique_ratio = long_unique_words / len(words)
    else:
        unique_word_ratio = stop_word_freq = long_unique_ratio = 0

    # Adjusted conditions
    if numeric_pattern_matches > 0:
        return True
    if digit_ratio > 0.08 and punctuation_ratio > 0.2:
        return True
    if symbol_cluster_matches > 0:
        return True
    if short_cluster_matches > 0 and digit_ratio > 0.1:
        return True
    if unusual_line_ratio > 0.3 and len(lines) > 3:
        return True
    if letter_count < total_length * 0.4 and digit_count > total_length * 0.1 and punctuation_count > total_length * 0.1:
        return True

    # Improved broken tags detection
    tag_count = len(re.findall(r'</?\w+[^>]*>', text))  # Count any tag-like structures
    malformed_tag_count = len(re.findall(r'</\w+[^>]*>|<\w+[^>]*[^/]>(?!.*</\w+>)|/>', text))
    if tag_count > 2 and malformed_tag_count > 0:
        return True

    # Linguistic incoherence checks
    if len(words) > 20:
        # if stop_word_freq < 0.05:
        #     return True
        if unique_word_ratio > 0.85:
            return True
        if long_unique_ratio > 0.5:  # High proportion of long unique words
            return True

    # Existing sentence-based check
    sentences = re.split(r'[.!?。]+', text)
    real_sentences = [s for s in sentences if len(s.strip()) > 10]
    if (
        punctuation_ratio > 0.1
        and digit_ratio > 0.1
        and len(real_sentences) < 2
        and len(text) > 100
        and not re.search(r'\\[\(\[\{].*?\\[\)\]\}]', text)
    ):
        return True

    return False


#     """
#     Checks if text contains incoherent content.
#     Returns True if content is incoherent, False if coherent.

#     The function considers text coherent if it:
#     - Forms proper sentences with logical flow
#     - Contains meaningful language (even if multilingual)
#     - Has proper structure and punctuation patterns

#     Text is considered incoherent if it:
#     - Contains random numeric patterns with excessive separators
#     - Has excessive special characters without context
#     - Contains code-like fragments mixed with natural language
#     - Shows nonsensical character combinations
#     """
#     import re

#     # Check for excessive numeric patterns with separators (a sign of incoherence)
#     numeric_pattern = r'(?:\d+\s*[,./\\\+\-\*]\s*){4,}'
#     if re.search(numeric_pattern, text):
#         return True

#     # Check for excessive amounts of special characters interspersed randomly
#     special_char_ratio = len(re.findall(r'[^\w\s]', text)) / (len(text) + 0.001)
#     if special_char_ratio > 0.12:  # Threshold determined from examples
#         return True

#     # Check for code-like fragments or HTML tags (common in incoherent examples)
#     code_fragments = r'</[a-z]+>|</?[a-z]+>|\{|\}|\[solution\]|</think>|<answer>|</answer>'
#     if re.search(code_fragments, text):
#         return True

#     # Check for excessive isolated symbols together
#     symbol_cluster = r'[\\\/\|\*\+]{3,}'
#     if re.search(symbol_cluster, text):
#         return True

#     # Check for repetitive number patterns with separators
#     number_separator_pattern = r'(?:\d+\s*[,./\\]\s*\d+\s*){3,}'
#     if re.search(number_separator_pattern, text):
#         return True

#     # Look for large clusters of short words and numbers separated by spaces
#     # (often present in incoherent text)
#     short_clusters = r'(?:\b\w{1,2}\b\s+){5,}'
#     if re.search(short_clusters, text):
#         return True

#     # Check for excessive word/number/symbol combinations
#     gibberish_pattern = r'(?:\w{1,2}[\+\-\*\/\\\|]){3,}'
#     if re.search(gibberish_pattern, text):
#         return True

#     # If none of the above patterns match, consider the text coherent
#     return False


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
