import re
from typing import Dict

import torch

from rl4llm.utils import build_longformer_classification_model_and_tokenizer

from .base_grader import BaseGrader


class FormatGrader(BaseGrader):
    """Checks for output format like XML structure, coherent"""

    def __init__(self, model_args: Dict, torch_dtype: torch.dtype, device: torch.device, **kwargs) -> float:
        self.device = device

        model, tokenizer = build_longformer_classification_model_and_tokenizer(model_args, torch_dtype)
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.threshold = kwargs.get('threshold', 0.9)

    def __call__(self, answer, ground_truth=None, **kwargs):
        """Returns graded scores (1.0 or -1.0) for one or more answers."""
        if isinstance(answer, str):
            # Single answer case
            is_coherent_list = self.__check_coherent(answer)
            is_coherent = is_coherent_list[0]  # Extract the single result
            if not is_coherent:
                return -1.0
            xml_format = kwargs.get('xml_format', True)
            if xml_format:
                is_xml_valid = self.__check_xml_format(answer)
                return 1.0 if is_xml_valid else -1.0
            return 1.0
        elif isinstance(answer, list):
            # Batched answer case
            is_coherent_list = self.__check_coherent(answer)
            scores = []
            for ans, is_coherent in zip(answer, is_coherent_list):
                if not is_coherent:
                    scores.append(-1.0)
                else:
                    xml_format = kwargs.get('xml_format', True)
                    if xml_format:
                        is_xml_valid = self.__check_xml_format(ans)
                        scores.append(1.0 if is_xml_valid else -1.0)
                    else:
                        scores.append(1.0)
            return scores
        else:
            raise ValueError('answer must be a string or a list of strings')

    @torch.inference_mode()
    def __check_coherent(self, texts) -> list[bool]:
        """Checks coherent content for one or more texts using a pretrained Longformer sequence classification model."""
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize all texts in a batch
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=True,  # Pad to the longest in the batch, not max_length
        ).to(self.device)

        # Get model output for the batch
        output = self.model(**inputs)

        # Convert logits to probabilities
        probs = torch.softmax(output.logits, dim=1)
        prob_coherent = probs[:, 1].tolist()  # Probabilities for class 1 (coherent)

        # Return list of coherence decisions
        return [p > self.threshold for p in prob_coherent]

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
