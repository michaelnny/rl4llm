import re
from abc import ABC, abstractmethod
from collections import Counter

import torch

from rl4llm.utils import build_longformer_classification_model_and_tokenizer


class BaseGrader(ABC):
    """
    Base grader.
    """

    def __init__(
        self,
        model_args: dict = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device('cuda'),
        **kwargs,
    ) -> float:
        model = None
        tokenizer = None
        if model_args is not None:
            model, tokenizer = build_longformer_classification_model_and_tokenizer(model_args, torch_dtype)
            model = model.eval().to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def _check_coherent(self, texts: list[str]) -> list[bool]:
        """Checks coherent content for one or more texts using a pretrained Longformer sequence classification model."""
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

        # not enabled
        if self.model is None:
            return [True] * len(texts)

        # Tokenize all texts in a batch
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=True,  # 'max_length'
        ).to(self.device)

        # Get model output for the batch
        output = self.model(**inputs)

        # Convert logits to probabilities
        probs = torch.softmax(output.logits, dim=1)

        # Convert predictions to list of booleans (True for class 1, False for class 0)
        predictions = torch.argmax(probs, dim=1).cpu().tolist()
        return [pred == 1 for pred in predictions]

    def _check_repetition(
        self,
        text,
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
            raise ValueError('min_ngram must be greater than 5 and less than max_ngram.')
        if not (ngram_threshold > 5 and min_sentence_words > 5 and sentence_threshold > 3):
            raise ValueError('ngram_threshold and min_sentence_words must be > 5, and sentence_threshold must be > 3.')

        text_lower = text.lower()

        # Check repeated n-grams
        words = text_lower.split()
        for n in range(min_ngram, max_ngram + 1):
            # Create n-grams and count occurrences
            ngram_counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
            if any(count >= ngram_threshold for count in ngram_counts.values()):
                return True

        # Check repeated sentences
        sentences = [line.strip() for line in text_lower.splitlines() if len(line.split()) > min_sentence_words]
        sentence_counts = Counter(sentences)
        if any(count >= sentence_threshold for count in sentence_counts.values()):
            return True

        return False

    @abstractmethod
    def __call__(self, answer, ground_truth, **kwargs) -> float:
        """Returns graded scores"""
