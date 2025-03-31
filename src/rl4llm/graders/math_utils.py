"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence).
"""

import re
from typing import List, Optional, Tuple, Union

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
    match = re.search(
        r'the\s+(final\s+)?answer\s+is:?\s*([\d,\.]+)(?:\.|\s+|$)',
        answer_text,
        re.IGNORECASE,
    )
    if match:
        answer = match.group(2).strip()
        if answer.endswith(',') or answer.endswith('.'):
            answer = answer[:-1].strip()
        return answer if answer else None
    return None


def extract_last_n_numerical_values(
    answer_text: str, size: int = 2
) -> Optional[List[str]]:
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
    return [_normalize_answer_string(d) for d in valid_numbers[-size:]]


def normalize_math_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        return _normalize_latex_string(answer)
    except:
        return answer


def _normalize_answer_string(s: str) -> str:
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
        raise ValueError(
            'String does not start with a recognized boxing command.'
        )


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

    # Unified pattern to match all fraction cases
    pattern = r'(?:\\)?frac(?:{(\d+)}([^}{])|([^}{]){(\d+)}|(\d+){(\d+)}|([^}{])([^}{]))'

    def replace_frac(match):
        # Extract all possible groups (some will be None depending on the pattern matched)
        g1, g2, g3, g4, g5, g6, g7, g8 = match.groups()

        # Case 1: frac{num}single (e.g., frac{16}3)
        if g1 and g2:
            return f'\\frac{{{g1}}}{{{g2}}}'
        # Case 2: fracsingle{num} (e.g., frac3{16})
        elif g3 and g4:
            return f'\\frac{{{g3}}}{{{g4}}}'
        # Case 3: fracnum{num} (e.g., frac247{33})
        elif g5 and g6:
            return f'\\frac{{{g5}}}{{{g6}}}'
        # Case 4: fracsinglesingle (e.g., frac35 or \frac35)
        elif g7 and g8:
            return f'\\frac{{{g7}}}{{{g8}}}'
        return match.group(0)  # Fallback for no match

    # Process all fraction patterns in one go
    result = re.sub(pattern, replace_frac, input_string)

    # Handle any remaining correctly formatted fractions with existing braces
    if '\\frac{' in result and '}' in result:
        return result

    return input_string  # Return original if no valid fractions found


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
    if _has_numbers(
        original_string
    ):  # Use the original latex to check for numbers
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
