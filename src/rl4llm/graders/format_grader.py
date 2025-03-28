import re
from typing import Dict

from .base_grader import BaseGrader


class FormatGrader(BaseGrader):
    """Checks for output format like XML structure, coherent"""

    def __call__(self, answer, ground_truth=None, **kwargs):
        """Returns graded scores (1.0 or -1.0) for one or more answers."""
        if isinstance(answer, str):

            xml_format = kwargs.get('xml_format', True)
            if xml_format:
                is_xml_valid = self.__check_xml_format(answer)
                return 1.0 if is_xml_valid else 0.0
            return 0.0
        elif isinstance(answer, list):
            # Batched answer case
            scores = []
            for ans in answer:
                xml_format = kwargs.get('xml_format', True)
                if xml_format:
                    is_xml_valid = self.__check_xml_format(ans)
                    scores.append(1.0 if is_xml_valid else 0.0)
                else:
                    scores.append(0.0)
            return scores
        else:
            raise ValueError('answer must be a string or a list of strings')

    def __check_xml_format(self, text) -> bool:
        """Checks R1 style XML format"""

        xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'

        text = text.strip()

        # Strip any code block markers
        text = re.sub(r'```.*?\n|```', '', text, flags=re.DOTALL)

        # Validate structure with regex
        xml_pattern = r'^<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
        match = re.match(xml_pattern, text, re.DOTALL | re.MULTILINE)

        if not match:
            return False

        think_content = match.group(1).strip()
        answer_content = match.group(2).strip()
        if not think_content or not answer_content:
            return False

        # Count tag occurrences
        think_open = text.count('<think>')
        think_close = text.count('</think>')
        answer_open = text.count('<answer>')
        answer_close = text.count('</answer>')
        if think_open != 1 or think_close != 1 or answer_open != 1 or answer_close != 1:
            return False

        # Check basic structure
        if not text.startswith('<think>') or not text.endswith('</answer>'):
            return False

        # Check for forbidden tags in content
        forbidden_tags = r'<think>|</think>|<answer>|</answer>'
        if re.search(forbidden_tags, think_content) or re.search(forbidden_tags, answer_content):
            return False

        return True
