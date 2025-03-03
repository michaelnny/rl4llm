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

    def get_summary(self, skip_list: List[str] = ['loss', 'grad_norm', 'prompt_length', 'total_reward']) -> Dict[str, float]:
        """Get summary of all metrics"""
        summary = {}

        for name, values in self._metrics.items():
            # Basic mean calculation
            summary[name] = np.mean(values).item()

            # Skip if only one value or in skip_list
            if len(values) <= 1 or any(k in name for k in skip_list):
                continue

            # Add standard deviation
            summary[f"{name}_std"] = np.std(values).item()

            # Handle length-specific metrics
            if 'completion_length' in name and 'training' in name:
                summary[f"{name}_max"] = np.max(values).item()
                summary[f"{name}_min"] = np.min(values).item()

                max_value = np.max(values).item()
                max_count = np.sum(values == max_value).item()
                summary[f"{name}_max_ratio"] = (max_count / len(values)) if max_count > 1 else 0.0
                summary[f"{name}_p99"] = np.percentile(values, 99).item()

        return summary
