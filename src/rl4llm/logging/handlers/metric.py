import logging
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union
import torch
import numpy as np

from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers.base import BaseHandler


def calculate_percentile(data: List[float], p: int) -> float:
    """Compute percentile for the given data"""
    if not data:
        return np.nan
    return np.percentile(data, p)


def is_valid_array(x) -> bool:
    """Check if x is a non-None, non-empty NumPy array."""
    return x is not None and isinstance(x, np.ndarray) and x.size > 0


class MetricHandler(BaseHandler):
    """Handles buffering, aggregation, and distributed gathering of scalar metrics."""

    AGGREGATORS: Dict[str, Callable] = {
        "mean": lambda x: np.mean(x) if is_valid_array(x) else np.nan,
        "std": lambda x: np.std(x) if is_valid_array(x) else np.nan,
        "min": lambda x: np.min(x) if is_valid_array(x) else np.nan,
        "max": lambda x: np.max(x) if is_valid_array(x) else np.nan,
        "sum": lambda x: np.sum(x) if is_valid_array(x) else 0.0,
        "last": lambda x: x[-1]
        if is_valid_array(x)
        else np.nan,  # Return NaN if empty for consistency
        "p50": lambda x: calculate_percentile(x.tolist(), 50)
        if is_valid_array(x)
        else np.nan,
        "p90": lambda x: calculate_percentile(x.tolist(), 90)
        if is_valid_array(x)
        else np.nan,
        "p95": lambda x: calculate_percentile(x.tolist(), 95)
        if is_valid_array(x)
        else np.nan,
        "p99": lambda x: calculate_percentile(x.tolist(), 99)
        if is_valid_array(x)
        else np.nan,
        "count": lambda x: len(x) if x is not None and isinstance(x, np.ndarray) else 0,
    }
    BASE_DEFAULT_METRICS_AGGREGATION_CONFIG = {
        "reward": ["mean", "std", "p50", "p90", "min", "max"],
        "loss": ["mean"],
        "learning_rate": ["last"],
        "lr": ["last"],
        "grad_norm": ["mean", "max"],
        "gradient_norm": ["mean", "max"],
        "entropy": ["mean"],
        "kl": ["mean", "std"],
        "kl_divergence": ["mean", "std"],
        "accuracy": ["mean"],
        "perplexity": ["mean"],
        "policy_update": ["last"],
        "global_step": ["last"],
        r".*_loss$": ["mean"],
        r".*_reward$": ["mean", "std"],
        r".*_count$": ["sum"],
        r".*_total$": ["sum"],
        r".*_update$": ["sum"],
        r".*_episodes$": ["sum"],
        r"^time/.*_sec$": ["mean", "sum"],
        "default": ["mean"],
    }

    def __init__(
        self,
        dist_manager: DistributedManager,
        user_aggregation_config: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)  # Initialize base class
        self.dist_manager = dist_manager
        self.is_master = dist_manager.is_master
        self.world_size = dist_manager.world_size

        self.effective_metrics_config = (
            self.BASE_DEFAULT_METRICS_AGGREGATION_CONFIG.copy()
        )
        if user_aggregation_config:
            self.effective_metrics_config.update(user_aggregation_config)
            self._logger.info("User metrics aggregation config provided. Merged.")
        else:
            self._logger.info("Using base default metrics aggregation config.")
        self._logger.debug(f"Effective metrics config: {self.effective_metrics_config}")

        self._metric_buffer: Dict[str, List[Union[float, int]]] = defaultdict(list)
        self._logged_regex_errors: set = set()

    def log_scalar(self, key: str, value: Union[float, int, torch.Tensor]):
        """Buffers a scalar metric locally."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if not isinstance(value, (float, int)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                self._logger.warning(
                    f"Could not convert metric '{key}' value '{value}' to float. Skipping."
                )
                return
        if not np.isfinite(value):
            self._logger.warning(
                f"Non-finite value ({value}) received for metric '{key}'. Skipping."
            )
            return
        self._metric_buffer[key].append(value)
        self._logger.debug(f"Buffered scalar [{key}]: {value}")

    def _get_aggregation_methods(self, metric_name: str) -> List[str]:
        """Determines aggregation methods using the effective_metrics_config."""
        if metric_name in self.effective_metrics_config:
            return self.effective_metrics_config[metric_name]
        for pattern, methods in self.effective_metrics_config.items():
            is_regex = any(c in pattern for c in r"^$*+?{}[]\|()")
            if is_regex:
                try:
                    if re.match(pattern, metric_name):
                        return methods
                except re.error as e:
                    if pattern not in self._logged_regex_errors:
                        self._logger.warning(
                            f"Invalid regex pattern '{pattern}': {e}. Skipping."
                        )
                        self._logged_regex_errors.add(pattern)
        return self.effective_metrics_config.get("default", ["mean"])

    def aggregate(self) -> Dict[str, float]:
        """Gathers metrics from all ranks and computes aggregates on rank 0."""
        final_aggregated_metrics = {}
        local_metric_buffer = self._metric_buffer
        gathered_buffers: Optional[List[Dict[str, List[Union[float, int]]]]] = None

        if self.world_size > 1:
            self.dist_manager.barrier()
            gathered_buffers = self.dist_manager.gather_object(
                local_metric_buffer, dst=0
            )
        elif self.is_master:
            gathered_buffers = [local_metric_buffer]

        if self.is_master and gathered_buffers:
            combined_metrics: Dict[str, List[Union[float, int]]] = defaultdict(list)
            all_metric_keys = set()
            for rank_buffer in gathered_buffers:
                if rank_buffer:
                    all_metric_keys.update(rank_buffer.keys())
                    for key, values in rank_buffer.items():
                        combined_metrics[key].extend(
                            v for v in values if np.isfinite(v)
                        )
            self._logger.debug(
                f"Master combined metrics for keys: {sorted(list(all_metric_keys))}"
            )

            for key, all_values in combined_metrics.items():
                if not all_values:
                    self._logger.debug(f"Skipping empty/non-finite metric: {key}")
                    continue
                aggregation_methods = self._get_aggregation_methods(key)
                self._logger.debug(
                    f"Aggregating '{key}' using methods: {aggregation_methods}"
                )
                for method_name in aggregation_methods:
                    aggregator_func = self.AGGREGATORS.get(method_name)
                    if aggregator_func:
                        try:
                            computed_value = aggregator_func(np.array(all_values))
                            log_key = key
                            if method_name != "mean" or len(aggregation_methods) > 1:
                                log_key = f"{key}_{method_name}"
                            final_aggregated_metrics[log_key] = float(computed_value)
                        except Exception as e:
                            self._logger.error(
                                f"Error computing '{method_name}' for '{key}': {e}"
                            )
                    else:
                        self._logger.warning(
                            f"Aggregator '{method_name}' not found for '{key}'."
                        )

        if self.world_size > 1:
            self.dist_manager.barrier()
        return final_aggregated_metrics

    def clear_buffer(self) -> None:
        """Clears the local metric buffer."""
        self._metric_buffer.clear()
        self._logger.debug("Cleared local metric buffer.")

    def close(self) -> None:
        """Closes the MetricHandler (currently no specific resources to release)."""
        self._logger.debug("Closing MetricHandler.")
        # No network or file resources owned directly by this handler
        pass
