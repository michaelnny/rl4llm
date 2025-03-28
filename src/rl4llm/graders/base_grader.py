from abc import ABC, abstractmethod

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

    @abstractmethod
    def __call__(self, answer, ground_truth, **kwargs) -> float:
        """Returns graded scores"""
