import abc
import logging
from typing import Optional

# --- Base Handler Definition ---


class BaseHandler(abc.ABC):
    """Abstract base class for logging handlers."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes the base handler.

        Args:
            logger: Logger instance for internal handler logging.
        """
        self._logger = (
            logger if logger is not None else logging.getLogger(self.__class__.__name__)
        )
        self._logger.debug(f"Initialized {self.__class__.__name__}")

    @abc.abstractmethod
    def close(self) -> None:
        """
        Abstract method for closing the handler and releasing resources.
        Subclasses must implement this.
        """
        pass

    # Optional: Add a flush method if universally applicable,
    # but currently only SampleHandler needs an explicit one.
    # def flush(self) -> None:
    #     """Optional method to flush buffered data."""
    #     pass
