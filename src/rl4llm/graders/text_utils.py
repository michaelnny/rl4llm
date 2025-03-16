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
        if line.startswith(r"\[") or line.startswith(r"\(") or line.startswith(r"\]") or line.startswith(r"\)"):
            return True
        if r"\text" in line:
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




# def has_incoherent_content(text):
#     """
#     Returns True if the text appears incoherent, False if it looks like a properly structured explanation.
    
#     The function first breaks the text into candidate lines (using newlines if available, otherwise using common
#     end‐of–sentence punctuation). Then for each line it does two checks:
#       1. It “cleans” the start (allowing for a leading backslash that might begin a LaTeX math block) and then 
#          requires that the first “real” character is either a digit or (if an ASCII letter) an uppercase letter.
#          (Non–ASCII letters -- as in Chinese – are accepted as well.)
#       2. For non–math lines it computes the ratio of digits to letters and “disqualifies” any that have an unusually high
#          ratio (here above 0.4).
         
#     A helper function is used to mark math–mode lines (for example those starting with “\[” or containing “\text”)
#     so that they do not get falsely flagged.
    
#     Finally the fraction of “good” lines is computed (ignoring any very short lines that are not math). If fewer than 60%
#     pass the checks the text is flagged as incoherent (True).
#     """
#     def is_math_mode(line):
#         # If the line starts with a math delimiter or contains LaTeX commands.
#         if line.startswith(r"\[") or line.startswith(r"\(") or line.startswith(r"\]") or line.startswith(r"\)"):
#             return True
#         if r"\text" in line:
#             return True
#         # Also, if the entire line is made up only of digits, dot, plus, minus, equals, slash, asterisk,
#         # parentheses and whitespace then treat it as a math expression.
#         if re.fullmatch(r'[\d\.\+\-\=\/\*\(\)\s]+', line):
#             return True
#         return False

#     # If there are several (nonempty) lines then use those;
#     # otherwise, fall back on a basic sentence split.
#     lines = [line.strip() for line in text.splitlines() if line.strip()]
#     if len(lines) < 2:
#         lines = re.split(r'(?<=[.!?])\s+', text)
#         lines = [line.strip() for line in lines if line.strip()]
    
#     total = 0
#     valid = 0
#     for line in lines:
#         # For non–math lines, ignore very short lines (fewer than 3 words)
#         words = line.split()
#         if not is_math_mode(line) and len(words) < 3:
#             continue
#         total += 1

#         # Remove any leading characters that are not word characters (or Chinese letters) or a backslash.
#         cleaned = re.sub(r'^[^\w\u4e00-\u9fff\\]+', '', line)
#         if not cleaned:
#             continue

#         # If this is math mode, count it as valid and continue.
#         if is_math_mode(cleaned):
#             valid += 1
#             continue

#         # Check the “first‐character rule”: if it starts with a digit it is fine; if it’s an ASCII letter, require uppercase;
#         # non–ASCII letters (for example Chinese) are accepted.
#         first = cleaned[0]
#         if first.isdigit():
#             pass
#         elif first.isalpha():
#             if first.isascii() and not first.isupper():
#                 continue  # reject this line
#         else:
#             continue  # if the first character is not alphanumeric, reject

#         # Next check the ratio of digits to letters.
#         letter_count = sum(1 for c in cleaned if c.isalpha())
#         digit_count = sum(1 for c in cleaned if c.isdigit())
#         # For non–math lines that have letters, if digits overwhelm letters (here > 40%), reject.
#         if letter_count > 0 and (digit_count / letter_count) > 0.4:
#             continue
#         if letter_count == 0 and not is_math_mode(cleaned):
#             continue

#         valid += 1

#     # If no candidate lines were found, assume the text does not have incoherent content.
#     if total == 0:
#         return False
#     fraction = valid / total
#     # Return True (incoherent) if fewer than 60% of lines look acceptable.
#     return fraction < 0.6




# def has_incoherent_content(text):
#     # First check if this is clearly a well-structured math solution
#     # Well-structured solutions often contain these patterns
#     math_solution_patterns = [
#         r"Therefore,.*is \*\*\d+\*\*\.",
#         r"<answer>.*</answer>",
#         r"Therefore, each person got \d+ seashells\.",
#         r"Therefore, the total.*is \d+ square inches\."
#     ]

#     for pattern in math_solution_patterns:
#         if re.search(pattern, text.strip()):
#             return False

#     # Check for sentences that have math formulas in a coherent way
#     math_formulas = re.findall(r'\\[[(].*?\\[])]', text)
#     equations = re.findall(r'=[^=]*?=', text)
#     if len(math_formulas) >= 3 or len(equations) >= 3:
#         if re.search(r'Therefore|Thus|So|Hence', text):
#             return False

#     # Look for incoherence signals

#     # 1. Disconnected number sequences with weird punctuation
#     disconnected_numbers = re.findall(r'(\d+\s*[,./\\;:*+\-]\s*\d+\s*[,./\\;:*+\-]\s*\d+)', text)
#     if disconnected_numbers:
#         return True

#     # 2. Random number sequences mixed with operators
#     random_number_sequences = re.findall(r'(\d+\s*[+\-*/\\]\s*\d+\s*[+\-*/\\]\s*\d+)', text)
#     if random_number_sequences and not re.search(r'total|sum|product|difference|calculate', text, re.IGNORECASE):
#         return True

#     # 3. Check for unusually many isolated numbers
#     isolated_numbers = re.findall(r'\b\d+\b', text)
#     if len(isolated_numbers) > 10 and not re.search(r'calculate|equation|solve|step|problem', text, re.IGNORECASE):
#         return True

#     # 4. Check for strings with unusual symbols
#     unusual_symbols = re.findall(r'[^\w\s,.;:?!()\[\]{}"\'-]{2,}', text)
#     if unusual_symbols:
#         return True

#     # 5. Gibberish words (look for very long made-up words)
#     words = re.findall(r'\b[a-zA-Z]{12,}\b', text)
#     for word in words:
#         if not word.lower() in ['consequently', 'approximately', 'multiplication', 'calculation', 'mathematics', 'relationship', 'intersection', 'contribution', 'determining', 'contributions']:
#             return True

#     # 6. Check for weird run-on words without spaces
#     run_on_words = re.findall(r'\b[a-zA-Z]{20,}\b', text)
#     if run_on_words:
#         return True

#     # 8. Excessive punctuation
#     excessive_punctuation = re.findall(r'[,.;:!?]{3,}', text)
#     if excessive_punctuation:
#         return True

#     # 9. Check for excessively spaced text
#     if re.search(r'[ ]{4,}', text):
#         return True

#     # 10. Random mix of symbols and letters
#     if re.search(r'[a-zA-Z][^\w\s]{2,}[a-zA-Z]', text):
#         return True

#     # 11. Check for Chinese characters mixed with Latin characters
#     chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
#     latin_chars = re.findall(r'[a-zA-Z]', text)
#     if chinese_chars and latin_chars and len(chinese_chars) < 20:
#         return True

#     # 12. Check for random punctuation inside words
#     if re.search(r'\w[^\w\s]{2,}\w', text):
#         return True

#     # 13. Catch-all for sentences with nonsensical structure - many commas with numbers
#     nonsensical_structure = re.findall(r'(\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)', text)
#     if nonsensical_structure:
#         return True

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
