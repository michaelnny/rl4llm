"""Handler for aggregate metrics"""

import logging
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

import numpy as np
import torch

from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers.base_handler import BaseHandler


def is_valid_array(x) -> bool:
    """Check if x is a non-None, non-empty NumPy array."""
    return x is not None and isinstance(x, np.ndarray) and x.size > 0


class MetricHandler(BaseHandler):
    """
    Handles buffering, aggregation, and distributed gathering of scalar metrics.
    Includes caching for aggregation method lookups and pre-compiled regexes.
    Supports exact, fuzzy (keyword), and regex matching for aggregation rules
    using a single sorted list for non-regex patterns.
    """

    AGGREGATORS: Dict[str, Callable[[np.ndarray], Union[float, int]]] = {
        'mean': lambda x: np.mean(x) if is_valid_array(x) else np.nan,
        'std': lambda x: np.std(x) if is_valid_array(x) else np.nan,
        'min': lambda x: np.min(x) if is_valid_array(x) else np.nan,
        'max': lambda x: np.max(x) if is_valid_array(x) else np.nan,
        'sum': lambda x: np.sum(x) if is_valid_array(x) else 0.0,
        'last': lambda x: x[-1] if is_valid_array(x) else np.nan,
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

    BASE_DEFAULT_METRICS_AGGREGATION_CONFIG = {
        # Non-regex keywords
        'prompt_length': ['mean', 'std'],
        'completion_length': ['mean', 'std', 'p50', 'p90'],
        'reward': ['mean', 'std', 'p50', 'p90'],
        'accuracy_reward': ['mean', 'std', 'p50', 'p90'],
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
        # Regex patterns
        r'.*_reward$': ['mean', 'std'],
        r'.*_count$': ['sum'],
        r'.*_total$': ['sum'],
        r'.*_update$': ['sum'],
        r'^time/.*$': ['sum'],
        r'^resource/.*$': ['mean', 'min', 'max'],
        # Default
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
                'User metrics aggregation config provided. Merged (user keys override base).'
            )
        else:
            self._logger.info('Using base default metrics aggregation config.')
        self._logger.debug(f"Raw effective metrics config: {effective_config}")

        self._metric_buffer: Dict[str, List[Union[float, int]]] = defaultdict(
            list
        )
        self._aggregation_methods_cache: Dict[str, List[str]] = {}

        # Store non-regex keys sorted by length descending for prioritized matching (exact & fuzzy)
        self._non_regex_config_sorted: List[Tuple[str, List[str]]] = []
        # Store compiled regex patterns
        self._regex_config: List[Tuple[Pattern[str], List[str]]] = []
        # Store default methods
        self._default_methods: List[str] = ['mean']

        regex_chars = r'^$*+?{}[]\|().'
        logged_regex_errors: Set[str] = set()
        non_regex_items = {}  # Temporary dict to build the sorted list

        for pattern, methods in effective_config.items():
            if pattern == 'default':
                self._default_methods = methods
                continue

            is_regex = any(c in pattern for c in regex_chars)
            if pattern.startswith('^') or pattern.endswith('$'):
                is_regex = True

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
                # Collect non-regex items to be sorted later
                non_regex_items[pattern] = methods
                self._logger.debug(
                    f"Identified non-regex config for: '{pattern}'"
                )

        # Create the single sorted list for non-regex matching (longest first)
        self._non_regex_config_sorted = sorted(
            non_regex_items.items(), key=lambda item: len(item[0]), reverse=True
        )
        self._logger.debug(
            f"Non-regex keys (sorted by length for matching): {[k for k, v in self._non_regex_config_sorted]}"
        )

        self._logger.info(
            f"Metric config processing complete. "
            f"{len(self._non_regex_config_sorted)} non-regex rules (sorted list), "
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
                    value = value.cpu().item()
                except ValueError:
                    self._logger.error(
                        f"Could not convert non-scalar tensor for metric '{key}' "
                        f"to scalar. Skipping."
                    )
                    return
                except RuntimeError as e:
                    self._logger.error(
                        f"Runtime error converting tensor for metric '{key}' "
                        f"to scalar (is it on the correct device?): {e}. Skipping."
                    )
                    return
            else:
                if value.requires_grad:
                    value = value.detach()
                value = value.cpu().item()

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

        # Buffer non-finite values, handle during aggregation
        self._metric_buffer[key].append(value)
        self._logger.debug(f"Buffered scalar [{key}]: {value}")

    def _get_aggregation_methods(self, metric_name: str) -> List[str]:
        """
        Determines aggregation methods using pre-processed configs and cache.
        Order of matching: Cache -> Exact/Fuzzy (longest keyword first) -> Regex -> Default.
        Exact matches are prioritized within the non-regex list iteration.
        """
        # 1. Check Cache
        if metric_name in self._aggregation_methods_cache:
            return self._aggregation_methods_cache[metric_name]

        # 2. Check Non-Regex List (Exact Match first, then Fuzzy)
        for keyword, methods in self._non_regex_config_sorted:
            # Prioritize exact match
            if keyword == metric_name:
                self._logger.debug(
                    f"Metric '{metric_name}': Found exact match with keyword '{keyword}'."
                )
                self._aggregation_methods_cache[metric_name] = methods
                return methods
            # If not exact, check for fuzzy match (substring)
            elif keyword in metric_name:
                self._logger.debug(
                    f"Metric '{metric_name}': Found fuzzy match with keyword '{keyword}'."
                )
                self._aggregation_methods_cache[metric_name] = methods
                return methods

        # 3. Check Regex Match (only if no non-regex match found)
        for compiled_regex, methods in self._regex_config:
            if compiled_regex.match(metric_name):
                self._logger.debug(
                    f"Metric '{metric_name}': Found regex match with pattern '{compiled_regex.pattern}'."
                )
                self._aggregation_methods_cache[metric_name] = methods
                return methods

        # 4. Use Default
        self._logger.debug(
            f"Metric '{metric_name}': No specific match found, using default methods."
        )
        self._aggregation_methods_cache[metric_name] = self._default_methods
        return self._default_methods

    def aggregate(self) -> Dict[str, float]:
        """Gathers metrics from all ranks and computes aggregates on rank 0."""
        final_aggregated_metrics = {}
        local_metric_buffer = {
            k: list(v) for k, v in self._metric_buffer.items()
        }
        self.clear_buffer()

        gathered_buffers: Optional[List[Dict[str, List[Union[float, int]]]]] = (
            None
        )

        if self.world_size > 1:
            self.dist_manager.barrier()
            gathered_data = self.dist_manager.gather_object(
                local_metric_buffer, dst=0
            )
            if self.is_master:
                gathered_buffers = gathered_data
        elif self.is_master:
            gathered_buffers = [local_metric_buffer]

        if self.is_master and gathered_buffers is not None:
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
                        f"Skipping aggregation for '{key}': No finite values after gathering."
                    )
                    continue

                aggregation_methods = self._get_aggregation_methods(key)
                self._logger.debug(
                    f"Aggregating '{key}' using methods: {aggregation_methods}"
                )

                values_array = np.array(all_values, dtype=float)

                for method_name in aggregation_methods:
                    aggregator_func = self.AGGREGATORS.get(method_name)
                    if aggregator_func:
                        try:
                            computed_value = aggregator_func(values_array)
                            log_key = key
                            if (
                                method_name != 'last'
                                or not key.startswith('time/')
                                or not key.endswith('count')
                                or not key.endswith('total')
                            ) and len(aggregation_methods) > 1:
                                log_key = f"{key}_{method_name}"

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
                                f"with {len(all_values)} finite values ({values_array.dtype}): {e}",
                                exc_info=True,
                            )
                    else:
                        self._logger.warning(
                            f"Aggregator function '{method_name}' not found for metric '{key}'. Skipping method."
                        )

        self.dist_manager.barrier()

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
