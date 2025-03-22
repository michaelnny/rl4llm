from abc import ABC, abstractmethod


class BaseGrader(ABC):
    """
    Base grader.
    """

    @abstractmethod
    def __call__(self, answer, ground_truth, **kwargs) -> float:
        """Returns graded scores"""
