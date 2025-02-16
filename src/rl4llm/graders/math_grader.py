import logging
import math
import re
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# Pattern for list markers at the start of lines
LIST_MARKER_PATTERN = re.compile(
    r"""
    ^                           # Start of line
    \s*                         # Optional leading whitespace
    (?:
        (?:\d+\.?)|            # Numbered list with optional period (e.g., "1.", "2" or "1)", "2)")
        (?:[a-zA-Z]\.?)|       # Letter list with optional period (e.g., "a.", "b" or "a)", "b)")
        (?:[-*•●○◆▪])|        # Bullet points
        (?:\(?(?:\d+|[a-zA-Z])\)?) # Parenthesized numbers or letters
    )
    \s+                         # Required whitespace after marker
    """,
    re.VERBOSE | re.MULTILINE,
)

# Pattern for arithmetic expressions and standalone numbers
NUMBER_PATTERN = re.compile(
    r"""
    (?<!\S)                     # Ensure the match is not preceded by a non-whitespace character
    (?:[$€£¥₹])?               # Optional leading currency symbol
    [+-]?                       # Optional sign (+ or -)
    (?:                         # Start of non-capturing group for the integer part
        \d{1,3}(?:,\d{3})+     # Numbers with commas, e.g., 1,234 or 12,345,678
        |                       # OR
        \d+                     # Numbers without commas, e.g., 1234 or 12345678
    )
    (?:\.\d+)?                  # Optional decimal part, e.g., .56
    (?:[eE][+-]?\d+)?          # Optional scientific notation, e.g., e+10 or E-5
    (?:                         # Optional trailing units/symbols
        \s*                     # Optional whitespace
        (?:%|°[CF]|kg|km/h|mph|USD|EUR|GBP|JPY|INR)?  # Common units and currencies
    )?
    (?=\s|[.,!?;:]|$)          # Ensure the match is followed by whitespace, punctuation, or end of string
    """,
    re.VERBOSE,
)


def extract_math_answer_from_last_boxed(answer_text: str) -> Optional[str]:
    """
    Extracts the content from the last boxed expression in a string.
    """
    last_boxed = _last_boxed_only_string(answer_text)
    if last_boxed:
        return _remove_boxed(last_boxed)
    return None


def extract_math_answer_from_patterned_text(answer_text: str) -> Optional[str]:
    """
    Extracts the answer from a string, handling various cases, including
    text after the answer and removing trailing commas.
    """
    match = re.search(r'the\s+(final\s+)?answer\s+is:?\s*(.*?)(?:\.|\s+|$)', answer_text, re.IGNORECASE)
    if match:
        answer = match.group(2).strip()
        if answer.endswith(',') or answer.endswith('.'):
            answer = answer[:-1].strip()
        return answer if answer else None
    return None


def extract_last_n_numerical_values(answer_text: str, size: int = 2) -> Optional[List[str]]:
    """
    Extract a list of numerical values from the last N positions of a text string.
    """
    if not answer_text:
        return None

    # Split text into lines and process each line
    lines = answer_text.split('\n')
    valid_numbers = []

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Remove LaTeX math indicators
        line = re.sub(r'\\\\[()\[\]]|\\[()\[\]]', '', line)

        # Check if the line starts with a list marker
        list_marker_match = LIST_MARKER_PATTERN.match(line)
        if list_marker_match:
            # If it's a list marker, only look for numbers after the marker
            remainder = line[list_marker_match.end() :]
            numbers_in_line = NUMBER_PATTERN.findall(remainder)
        else:
            # If no list marker, look for numbers in the whole line
            numbers_in_line = NUMBER_PATTERN.findall(line)

        valid_numbers.extend(numbers_in_line)

    if not valid_numbers:
        return None

    # Return last 'size' numbers as a list
    return [normalize_number(d) for d in valid_numbers[-size:]]


def normalize_number(s: str) -> str:
    """
    Cleans the extracted number by removing commas and unnecessary characters.
    """
    if isinstance(s, (int, float)):
        return s

    # remove latex math indicators
    normed_s = str(s)
    normed_s = re.sub(r'\\', '', normed_s)

    # Strip any leading/trailing whitespace
    normed_s = normed_s.strip()

    # Remove currency signs and other special characters
    symbols = {
        '$',
        '€',
        '£',
        '¥',
        '₹',
        ',',
        '%',
        'kg',
        'km/h',
        'mph',
        'USD',
        'EUR',
        'GBP',
        'JPY',
        'INR',
    }

    for symbol in symbols:
        normed_s = normed_s.replace(symbol, '')
    normed_s = normed_s.strip()
    return normed_s


def _remove_boxed(s: str) -> str:
    """
    Removes the boxing commands (`\boxed` or `\fbox`) from a string.
    """
    if s.startswith('\\boxed '):
        return s[len('\\boxed ') :]
    elif s.startswith('\\boxed{') and s.endswith('}'):
        return s[len('\\boxed{') : -1]
    elif s.startswith('\\fbox{') and s.endswith('}'):
        return s[len('\\fbox{') : -1]
    else:
        raise ValueError('String does not start with a recognized boxing command.')


def _last_boxed_only_string(string: str) -> Optional[str]:
    """
    Retrieves the last boxed expression from a string.
    """
    # Search for \boxed with a space
    if '\\boxed ' in string:
        parts = string.split('\\boxed ')
        if len(parts) > 1:
            # Take the last part and extract up to the first '$' if present
            last_part = parts[-1].split('$')[0].strip()
            return f"\\boxed {last_part}"

    # Search for \boxed{...} or \fbox{...}
    for boxing_command in ['\\boxed{', '\\fbox{']:
        idx = string.rfind(boxing_command)
        if idx != -1:
            start_idx = idx + len(boxing_command)
            brace_count = 1
            i = start_idx
            while i < len(string):
                if string[i] == '{':
                    brace_count += 1
                elif string[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return string[idx : i + 1]
                i += 1
    return None


def _fix_fractions(input_string: str) -> str:
    """
    Corrects the formatting of fraction expressions in a string.
    """
    substrings = input_string.split('\\frac')
    new_string = substrings[0]
    if len(substrings) > 1:
        substrings = substrings[1:]
        for substring in substrings:
            new_string += '\\frac'
            if substring[0] == '{':
                new_string += substring
            else:
                try:
                    assert len(substring) >= 2
                except AssertionError:
                    return input_string
                numerator = substring[0]
                denominator_candidate = substring[1]
                if denominator_candidate != '{':
                    if len(substring) > 2:
                        remaining_substring = substring[2:]
                        new_string += '{' + numerator + '}{' + denominator_candidate + '}' + remaining_substring
                    else:
                        new_string += '{' + numerator + '}{' + denominator_candidate + '}'
                else:
                    if len(substring) > 2:
                        remaining_substring = substring[2:]
                        new_string += '{' + numerator + '}' + denominator_candidate + remaining_substring
                    else:
                        new_string += '{' + numerator + '}' + denominator_candidate
    return new_string


def _fix_a_slash_b_notation(input_string: str) -> str:
    """
    Converts simple division notation (e.g., "a/b") to LaTeX fraction format (e.g., "\\frac{a}{b}").
    """
    if len(input_string.split('/')) != 2:
        return input_string
    numerator_str = input_string.split('/')[0]
    denominator_str = input_string.split('/')[1]
    try:
        numerator = int(numerator_str)
        denominator = int(denominator_str)
        assert input_string == '{}/{}'.format(numerator, denominator)
        new_string = '\\frac{' + str(numerator) + '}{' + str(denominator) + '}'
        return new_string
    except ValueError:
        return input_string
    except AssertionError:
        return input_string


def _remove_right_side_units(input_string: str) -> str:
    """
    Removes unit descriptions from the right side of a string.
    """
    if '\\text{ ' in input_string:
        splits = input_string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return input_string


def _fix_sqrt_notation(input_string: str) -> str:
    """
    Corrects the formatting of square root expressions in a string.
    """
    if '\\sqrt' not in input_string:
        return input_string
    splits = input_string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            char_after_sqrt = split[0]
            new_substring = '\\sqrt{' + char_after_sqrt + '}' + split[1:]
        else:
            new_substring = '\\sqrt' + split
        new_string += new_substring
    return new_string


def _fix_tan_notation(input_string: str) -> str:
    out_string = re.sub(r'\\tan(-?[0-9.a-zA-Z]+)', r'\\tan{\1}', input_string)
    out_string = re.sub(r'\\tan\s+(\w+)$', r'\\tan{\1}', out_string)
    return out_string


def _has_numbers(text: str) -> bool:
    """Checks if the text contains any digits."""
    return bool(re.search(r'\d', text))


def _normalize_latex_string(input_string: str) -> str:
    """
    Performs a series of cleaning and formatting operations on a string.
    """
    original_string = input_string

    # linebreaks
    string = input_string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = string.replace('cfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # Remove \text{} wrappers
    if _has_numbers(original_string):  # Use the original latex to check for numbers
        # Remove \text{} commands and their content
        string = re.sub(r'\\text{.*?}', '', string)
        string = re.sub(r'\text{.*?}', '', string)
    else:
        # Remove \text{} commands
        string = re.sub(r'\\text{([^}]*)}', r'\1', string)
        string = re.sub(r'\text{([^}]*)}', r'\1', string)

    # Remove empty {} block
    string = re.sub(r'\{\}', '', string)

    # remove units
    string = re.sub(r'\{(c|m)?m\}(\^(2|3))?', '', string).strip()
    string = re.sub(r'p\.m\.$', '', string).strip()
    string = re.sub(r'(\d)\s*t$', r'\1', string).strip()

    # remove dollar signs
    string = string.replace('\\$', '')
    string = string.replace('$', '')

    # remove units (on the right)
    string = _remove_right_side_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace(r'\%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')

    # remove \cdot
    string = string.replace('\\cdot', '')

    # normalize infinity
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')

    string = string.replace('\\mathbf', '')
    string = string.replace('\\mathrm', '')

    # remove \mbox{...}
    string = re.sub(r'\\mbox{.*?}', '', string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt_notation(string)
    string = _fix_tan_notation(string)

    # Remove unnecessary whitespace
    string = re.sub(r'\s+', '', string)

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fractions(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b_notation(string)

    # Remove ",!" from numbers, e.g., "1,!000" --> "1000"
    string = re.sub(r'(\d),!(\d)', r'\1\2', string)

    # Regex pattern to match valid numbers with commas as thousand separators
    pattern = r'^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?$'
    if re.fullmatch(pattern, string):
        string = string.replace(',', '')

    return string


def try_compare_fractions_equal(input_str1: str, input_str2: str) -> bool:
    """Try to compare two fractions."""

    def fraction_to_float(frac_str):
        try:
            # Normalize fractions (both \frac, \dfrac, \tfrac) including negatives and decimals
            frac_str = re.sub(r'\\(?:frac|dfrac|tfrac){([+-]?\d*\.?\d+)}{([+-]?\d*\.?\d+)}', r'\1/\2', frac_str)
            return float(frac_str)
        except ValueError:
            pass

        try:
            num, denom = frac_str.split('/')
            denom = float(denom)
            if denom == 0:
                raise ValueError('Denominator cannot be zero.')
            return float(num) / denom
        except Exception:
            pass
        return None

    try:
        frac_1 = fraction_to_float(input_str1)
        frac_2 = fraction_to_float(input_str2)
        if frac_1 is not None and frac_2 is not None:
            return math.isclose(frac_1, frac_2, rel_tol=1e-6)
    except Exception:
        pass
    return False


def check_expressions_equivalent(expression1: Optional[str], expression2: Optional[str], verbose: bool = False) -> bool:
    """
    Checks if two mathematical expressions are equivalent after applying a series of normalization steps.

    Args:
        expression1: The first mathematical expression string.
        expression2: The second mathematical expression string.
        verbose: If True, prints the normalized forms of the expressions before comparison.

    Returns:
        True if the normalized forms of the two expressions are identical, False otherwise.
    """
    if expression1 is None and expression2 is None:
        logger.warning('Both values are None')
        return True
    if expression1 is None or expression2 is None:
        return False

    try:
        normalized_expression1 = _normalize_latex_string(expression1)
        normalized_expression2 = _normalize_latex_string(expression2)
        logger.debug(normalized_expression1, normalized_expression2)
        return normalized_expression1 == normalized_expression2 or float(normalized_expression1) == float(
            normalized_expression2
        )
    except Exception:
        pass

    try:
        # Try to evaluate the expressions as mathematical expressions
        # example: '-\frac{1}{4}' vs '-0.25'
        is_frac_equal = try_compare_fractions_equal(expression1, expression2)
        if is_frac_equal:
            return True
    except Exception:
        pass

    # Fail back to string comparison
    return expression1 == expression2


def math_problem_grader(
    full_answer: str,
    ground_truth: str,
    last_n: int = 3,
) -> float:
    """
    Enhanced grader that handles multiple answer formats and extraction methods.
    """
    if full_answer is None or ground_truth is None:
        return 0.0

    logger.debug(f"Processing answer: {full_answer}")
    logger.debug(f"Ground truth: {ground_truth}")

    # 1. Try boxed answers
    boxed_answer = extract_math_answer_from_last_boxed(full_answer)
    if boxed_answer is not None:
        logger.debug(f"Found boxed answer: {boxed_answer}")
        if check_expressions_equivalent(boxed_answer, ground_truth):
            return 1.0
        else:
            return 0.0

    # 2. Try patterned answers
    pattern_answer = extract_math_answer_from_patterned_text(full_answer)
    if pattern_answer is not None:
        logger.debug(f"Found patterned answer: {pattern_answer}")
        if check_expressions_equivalent(pattern_answer, ground_truth):
            return 1.0
        else:
            return 0.0

    # 3. Fallback to last N numerical values
    number_list = extract_last_n_numerical_values(full_answer, size=last_n)
    if number_list:
        for num in number_list:
            if check_expressions_equivalent(num, ground_truth):
                return 1.0

    return 0.0
