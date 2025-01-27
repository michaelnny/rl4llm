"""Math data augmenter class"""

import random
import re
from typing import Tuple, List, Dict


class MathAugmenter:
    """Math data augmenter by randomly replacing some numerical values in the text with similar values."""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def augment_text(self, text: str, max_replacements: int = 3) -> Tuple[str, List[Dict]]:
        """Augment text by replacing some numerical values with similar values.

        Args:
            text (str): The text to augment.
            max_replacements (int): Maximum number of replacements to make.

        Returns:
            Tuple[str, List[Dict]]: The augmented text and a list of replacements stats.
                Each replacement is a dictionary with the following keys:
                - 'original_value': The original numerical value replaced.
                - 'new_value': The new numerical value generated.
                - 'start_line': The line number where the replacement started.
                - 'lines_changed': A list of line numbers where the replacement was made.
        """
        lines = text.splitlines(keepends=True)
        if not lines:
            return text, []

        start_line_index = random.randint(0, len(lines) - 1)
        start_line_number = start_line_index + 1

        replacements_made = []
        modified_lines = list(lines)
        replaced_numbers_map = {}  # Keep track of already replaced numbers to avoid re-replacement
        replacements_count = 0

        number_pattern = r'[-+]?\d*\.?\d+'

        for line_index in range(start_line_index, len(modified_lines)):
            if replacements_count >= max_replacements:
                break  # Stop if max replacements reached

            line = modified_lines[line_index]
            found_numbers_str = re.findall(number_pattern, line)
            found_numbers_unique_str = list(dict.fromkeys(found_numbers_str))  # Remove duplicates but keep order

            numbers_in_line = []
            for number_str in found_numbers_unique_str:
                if number_str not in replaced_numbers_map:  # Only consider numbers not already replaced
                    try:
                        num = int(number_str)
                        numbers_in_line.append((number_str, int))
                    except ValueError:
                        try:
                            num = float(number_str)
                            numbers_in_line.append((number_str, float))
                        except ValueError:
                            pass

            if numbers_in_line:
                number_str_to_replace, number_type = random.choice(numbers_in_line)  # Choose one number per line
                original_number_val = number_type(number_str_to_replace)
                new_number_val = self._generate_similar_number(original_number_val, number_type)
                new_number_str = str(new_number_val)

                replaced_numbers_map[number_str_to_replace] = new_number_str  # Mark as replaced globally
                replacement_record = {
                    'original_value': number_str_to_replace,
                    'new_value': new_number_str,
                    'start_line': start_line_number,
                    'lines_changed': [],
                }
                replacements_made.append(replacement_record)
                replacements_count += 1  # Increment replacement count

                for subsequent_line_index in range(start_line_index, len(modified_lines)):
                    subsequent_line = modified_lines[subsequent_line_index]
                    original_subsequent_line = subsequent_line
                    modified_lines[subsequent_line_index] = subsequent_line.replace(number_str_to_replace, new_number_str)
                    if modified_lines[subsequent_line_index] != original_subsequent_line:
                        replacement_record['lines_changed'].append(subsequent_line_index + 1)

        modified_text = ''.join(modified_lines)
        return modified_text, replacements_made

    def _generate_similar_number(self, original_number, number_type):
        if number_type is int:
            change = random.randint(-2, 2)
            return original_number + change
        elif number_type is float:
            magnitude = abs(original_number)
            if magnitude < 1:
                change_factor = random.uniform(-0.2, 0.2)
            else:
                change_factor = random.uniform(-0.1, 0.1)
            change = original_number * change_factor
            return original_number + change
        return original_number
