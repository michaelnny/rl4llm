import logging
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

import numpy as np
import torch

from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers.base import BaseHandler


def is_valid_array(x) -> bool:
    """Check if x is a non-None, non-empty NumPy array."""
    return x is not None and isinstance(x, np.ndarray) and x.size > 0


class MetricHandler(BaseHandler):
    """
    Handles buffering, aggregation, and distributed gathering of scalar metrics.
    Includes caching for aggregation method lookups and pre-compiled regexes.
    """

    # Refined AGGREGATORS with inline checks for percentiles
    AGGREGATORS: Dict[str, Callable[[np.ndarray], Union[float, int]]] = {
        'mean': lambda x: np.mean(x) if is_valid_array(x) else np.nan,
        'std': lambda x: np.std(x) if is_valid_array(x) else np.nan,
        'min': lambda x: np.min(x) if is_valid_array(x) else np.nan,
        'max': lambda x: np.max(x) if is_valid_array(x) else np.nan,
        'sum': lambda x: np.sum(x) if is_valid_array(x) else 0.0,
        'last': lambda x: x[-1] if is_valid_array(x) else np.nan,
        # Use inline check + np.percentile directly
        'p50': lambda x: (
            np.percentile(x.astype(float), 50) if is_valid_array(x) else np.nan
        ),
        'p90': lambda x: (
            np.percentile(x.astype(float), 90) if is_valid_array(x) else np.nan
        ),
        'p95': lambda x: (
            np.percentile(x.astype(float), 95) if is_valid_array(x) else np.nan
        ),
        'p99': lambda x: (
            np.percentile(x.astype(float), 99) if is_valid_array(x) else np.nan
        ),
        'count': lambda x: len(x) if isinstance(x, np.ndarray) else 0,
    }

    # BASE_DEFAULT_METRICS_AGGREGATION_CONFIG remains the same
    BASE_DEFAULT_METRICS_AGGREGATION_CONFIG = {
        'completion_length': ['mean', 'std', 'p50', 'p90', 'min', 'max'],
        'reward': ['mean', 'std', 'p90', 'min', 'max'],
        'loss': ['mean'],
        'learning_rate': ['last'],
        'lr': ['last'],
        'grad_norm': ['mean', 'max'],
        'gradient_norm': ['mean', 'max'],
        'entropy': ['mean'],
        'kl': ['mean', 'std'],
        'kl_divergence': ['mean', 'std'],
        'return': ['mean', 'std'],
        'advantage': ['mean', 'std'],
        'accuracy': ['mean'],
        'perplexity': ['mean'],
        'policy_update': ['last'],
        'global_step': ['last'],
        r'.*_loss$': ['mean'],
        r'.*_reward$': ['mean', 'std'],
        r'.*_count$': ['sum'],
        r'.*_total$': ['sum'],
        r'.*_update$': ['sum'],
        r'.*_episodes$': ['sum'],
        r'^time/.*$': ['sum'],
        'default': ['mean'],
    }

    def __init__(
        self,
        dist_manager: DistributedManager,
        user_aggregation_config: Optional[Dict[str, List[str]]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)
        self.dist_manager = dist_manager
        self.is_master = dist_manager.is_master
        self.world_size = dist_manager.world_size

        effective_config = self.BASE_DEFAULT_METRICS_AGGREGATION_CONFIG.copy()
        if user_aggregation_config:
            effective_config.update(user_aggregation_config)
            self._logger.info(
                'User metrics aggregation config provided. Merged.'
            )
        else:
            self._logger.info('Using base default metrics aggregation config.')
        self._logger.debug(f"Raw effective metrics config: {effective_config}")

        self._metric_buffer: Dict[str, List[Union[float, int]]] = defaultdict(
            list
        )
        self._aggregation_methods_cache: Dict[str, List[str]] = {}
        self._non_regex_config: Dict[str, List[str]] = {}
        self._regex_config: List[Tuple[Pattern[str], List[str]]] = []
        self._default_methods: List[str] = ['mean']

        regex_chars = r'^$*+?{}[]\|()'
        logged_regex_errors: Set[str] = set()

        for pattern, methods in effective_config.items():
            if pattern == 'default':
                self._default_methods = methods
                continue

            is_regex = any(c in pattern for c in regex_chars)
            if is_regex:
                try:
                    compiled_regex = re.compile(pattern)
                    self._regex_config.append((compiled_regex, methods))
                    self._logger.debug(f"Compiled regex: '{pattern}'")
                except re.error as e:
                    if pattern not in logged_regex_errors:
                        self._logger.warning(
                            f"Invalid regex pattern '{pattern}' during init: {e}. Skipping."
                        )
                        logged_regex_errors.add(pattern)
            else:
                self._non_regex_config[pattern] = methods
                self._logger.debug(f"Stored non-regex config for: '{pattern}'")

        self._logger.info(
            f"Metric config processing complete. "
            f"{len(self._non_regex_config)} non-regex rules, "
            f"{len(self._regex_config)} regex rules."
        )
        self._logger.debug(f"Default methods set to: {self._default_methods}")

    def log_scalar(self, key: str, value: Union[float, int, torch.Tensor]):
        """Buffers a scalar metric locally."""
        original_type = type(value).__name__
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                self._logger.warning(
                    f"Metric '{key}' received a non-scalar tensor "
                    f"(shape: {value.shape}). Attempting to use item()."
                )
                if value.requires_grad:
                    value = value.detach()
                try:
                    value = value.item()
                except ValueError:
                    self._logger.error(
                        f"Could not convert non-scalar tensor for metric '{key}' "
                        f"to scalar. Skipping."
                    )
                    return
            else:
                if value.requires_grad:
                    value = value.detach()
                value = value.item()

        if not isinstance(value, (float, int)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                self._logger.warning(
                    f"Could not convert metric '{key}' value '{value}' "
                    f"(original type: {original_type}) to float. Skipping."
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
        """
        Determines aggregation methods using pre-processed configs and cache.
        Order of matching: Cache -> Exact Match -> Prefix Match -> Regex -> Default.
        """
        if metric_name in self._aggregation_methods_cache:
            return self._aggregation_methods_cache[metric_name]

        if metric_name in self._non_regex_config:
            methods = self._non_regex_config[metric_name]
            self._aggregation_methods_cache[metric_name] = methods
            return methods

        if '/' in metric_name:
            prefix = metric_name.split('/')[0]
            if prefix in self._non_regex_config:
                methods = self._non_regex_config[prefix]
                self._aggregation_methods_cache[metric_name] = methods
                return methods

        for compiled_regex, methods in self._regex_config:
            if compiled_regex.match(metric_name):
                self._aggregation_methods_cache[metric_name] = methods
                return methods

        self._aggregation_methods_cache[metric_name] = self._default_methods
        return self._default_methods

    def aggregate(self) -> Dict[str, float]:
        """Gathers metrics from all ranks and computes aggregates on rank 0."""
        final_aggregated_metrics = {}
        local_metric_buffer = self._metric_buffer.copy()
        gathered_buffers: Optional[List[Dict[str, List[Union[float, int]]]]] = (
            None
        )

        if self.world_size > 1:
            self.dist_manager.barrier()
            gathered_buffers = self.dist_manager.gather_object(
                local_metric_buffer, dst=0
            )
        elif self.is_master:
            gathered_buffers = [local_metric_buffer]

        if self.is_master and gathered_buffers:
            combined_metrics: Dict[str, List[Union[float, int]]] = defaultdict(
                list
            )
            all_metric_keys: Set[str] = set()

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
                    self._logger.debug(
                        f"Skipping aggregation for '{key}': No valid values after gathering."
                    )
                    continue

                aggregation_methods = self._get_aggregation_methods(key)
                self._logger.debug(
                    f"Aggregating '{key}' using methods: {aggregation_methods}"
                )

                values_array = np.array(all_values)  # Create array once

                for method_name in aggregation_methods:
                    aggregator_func = self.AGGREGATORS.get(method_name)
                    if aggregator_func:
                        try:
                            # *** Pass the numpy array directly ***
                            computed_value = aggregator_func(values_array)

                            log_key = key
                            if (
                                method_name != 'mean'
                                or len(aggregation_methods) > 1
                            ):
                                log_key = f"{key}_{method_name}"

                            # *** Store only if finite ***
                            if np.isfinite(computed_value):
                                final_aggregated_metrics[log_key] = float(
                                    computed_value
                                )
                            else:
                                self._logger.debug(
                                    f"Aggregation method '{method_name}' for key '{key}' "
                                    f"resulted in non-finite value ({computed_value}). Skipping storage."
                                )

                        except Exception as e:
                            self._logger.error(
                                f"Error computing '{method_name}' for metric '{key}' "
                                f"with {len(all_values)} values ({values_array.dtype}): {e}",  # Added dtype
                                exc_info=True,
                            )
                    else:
                        self._logger.warning(
                            f"Aggregator function '{method_name}' not found for metric '{key}'. Skipping method."
                        )

        if self.world_size > 1:
            self.dist_manager.barrier()

        self.clear_buffer()  # Clear after processing

        return final_aggregated_metrics

    def clear_buffer(self) -> None:
        """Clears the local metric buffer."""
        self._metric_buffer.clear()
        self._logger.debug('Cleared local metric buffer.')

    def close(self) -> None:
        """Closes the MetricHandler."""
        self._logger.debug('Closing MetricHandler.')
        self.clear_buffer()
        self._aggregation_methods_cache.clear()
        pass
