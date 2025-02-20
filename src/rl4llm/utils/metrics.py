import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np


class MetricsCollector:
    """Metrics collector for RL training"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics for a new iteration"""
        self._metrics = defaultdict(list)

    @contextmanager
    def timer(self, phase: str):
        """Context manager for timing operations during a given phase"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self._log_time(phase, elapsed)

    def _log_time(self, phase: str, elapsed: float):
        """Log time for a given phase"""
        self._metrics[f"elapsed/{phase}_time"].append(elapsed)

    def add_metric(self, name: str, value: float):
        """Add a generation-related metric"""
        assert isinstance(value, (int, float))
        self._metrics[name].append(value)

    def add_metrics_batch(self, name: str, values: List[float]):
        """Add a generation-related metric"""
        assert isinstance(values, list)
        self._metrics[name].extend(values)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {name: values for name, values in self._metrics.items()}

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        summary = {}

        # Summarize generation metrics
        for name, values in self._metrics.items():
            summary[name] = np.mean(values).item()
            # Add std dev and variance for multiple values
            if len(values) > 1 and 'loss' not in name and 'kl' not in name:
                summary[f"{name}_std"] = np.std(values).item()
                summary[f"{name}_var"] = np.var(values).item()

        return summary
