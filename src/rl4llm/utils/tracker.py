"""For tracking training progress and logging data"""
import gzip
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import yaml
from torch.utils.tensorboard import SummaryWriter

from rl4llm.types import Episode


class FileHandler:
    """Simple file handler that manages a single file with buffered writing"""

    def __init__(self, save_path: str, file_type: str = 'csv', compress: bool = True):
        if file_type not in ['csv', 'jsonl']:
            raise ValueError("file_type must be either 'csv' or 'jsonl'")

        self.compress = compress
        # Add .gz extension if compression is enabled
        self.out_file = f"{save_path}.gz" if compress else save_path
        self.file_type = file_type
        self.buffer = []
        self._is_header_written_csv = False if file_type == 'csv' else True  # track csv header

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)

    def log_entry(self, data: Dict):
        """Add an entry to the buffer"""
        self.buffer.append(data)

    def flush(self):
        """Write buffered data to file"""
        if not self.buffer:
            return

        if self.file_type == 'csv':
            df = pd.DataFrame(self.buffer)
            write_header = not self._is_header_written_csv

            if self.compress:
                df.to_csv(
                    self.out_file, mode='a' if not write_header else 'w', header=write_header, index=False, compression='gzip'
                )
            else:
                df.to_csv(
                    self.out_file,
                    mode='a' if not write_header else 'w',
                    header=write_header,
                    index=False,
                )
            if write_header:
                self._is_header_written_csv = True

        else:  # jsonl
            mode = 'at' if self.compress else 'a'
            opener = gzip.open if self.compress else open

            with opener(self.out_file, mode, encoding='utf-8') as f:
                for item in self.buffer:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

        self.buffer.clear()

    def close(self):
        """Flush remaining data and close"""
        self.flush()


class TensorBoardLogger:
    """Handles logging to TensorBoard (no interval management now)"""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, name: str, value: float, step: int, tag: str):
        self.writer.add_scalar(f"{tag}/{name}", value, step)

    def log_dict(self, data: Dict[str, Any], step: int, tag: str):
        for name, value in data.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{tag}/{name}", value, step)

    def log_text(self, name: str, text: str, step: int, tag: str):
        self.writer.add_text(f"{tag}/{name}", text, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class ActorTracker:
    """Handles actor-specific logging including episode data"""

    def __init__(self, tb_logger: TensorBoardLogger, output_path: str, role: str, episode_log_interval: int = 10):
        self.tb_logger = tb_logger
        self.role_name = role.lower()
        self.step_tag = f"{self.role_name}/episode"
        self.iter_tag = f"{self.role_name}/iteration"
        self.episode_log_interval = episode_log_interval  # Interval for episode logs
        self._step_count = 0
        self._iter_count = 0

        # Initialize specific files upfront
        self.episode_file = FileHandler(
            os.path.join(output_path, f'{self.role_name}_episodes.jsonl'), file_type='jsonl', compress=True
        )
        self.stats_file = FileHandler(
            os.path.join(output_path, f'iteration_{self.role_name}_stats.csv'), file_type='csv', compress=False
        )

    def log_step(self, data: Episode):
        self._step_count += 1

        # Log episode to jsonl file
        episode_data = data.model_dump()
        # remove 'token_ids' from each transition's action, as it's not helpful
        for t in episode_data.get('transitions', []):
            if 'token_ids' in t['action']:
                del t['action']['token_ids']
            if 'token_logits' in t['action']:
                del t['action']['token_logits']
        self.episode_file.log_entry(episode_data)

        if self._step_count % self.episode_log_interval == 0:
            # Log episode statistics
            stats = self._collect_episode_stats(data)
            self.tb_logger.log_dict(stats, self._step_count, self.step_tag)
            # Log sample text
            self._log_episode_sample(data, self._step_count)

    def log_iteration(self, data: Dict[str, Any]):
        self._iter_count += 1
        self.tb_logger.log_dict(data, self._iter_count, self.iter_tag)
        record = {'iteration': self._iter_count, **data}
        self.stats_file.log_entry(record)

    def _collect_episode_stats(self, episode: Episode) -> Dict[str, Any]:
        stats = {}
        for t in episode.transitions:
            prefix = t.state.state_id
            stats.update(
                {
                    f"{prefix}_tokens": t.action.usage.completion_tokens,
                    f"{prefix}_reward": t.reward,
                }
            )
        if len(episode.transitions) > 1:
            stats['total_tokens'] = episode.count_total_tokens()
            stats['prompt_tokens'] = episode.count_prompt_tokens()
            stats['completion_tokens'] = episode.count_completion_tokens()
            stats['total_reward'] = episode.count_total_rewards()
        return stats

    def _log_episode_sample(self, episode: Episode, step: int):
        messages = episode.get_chat_messages_for_logging()
        full_chat = '\n\n'.join([f"[{t['role']}]\n\n{t['content']}" for t in messages])

        formatted_text = (
            f"**Question [{episode.task_type}]**: {episode.question}\n\n"
            f"**Ground Truth**: {episode.ground_truth}\n\n"
            f"**Short Answer**: {episode.short_answer}\n\n"
            f"**Graded Reward**: {episode.graded_reward:.2f}\n\n"
            f"**Full Chat History**:\n```json\n{full_chat}\n```"
        )
        self.tb_logger.log_text('sample', formatted_text, step, self.step_tag)

    def flush(self):
        self.episode_file.flush()
        self.stats_file.flush()

    def close(self):
        self.episode_file.close()
        self.stats_file.close()


class LearnerTracker:
    """Handles learner-specific logging"""

    def __init__(self, tb_logger: TensorBoardLogger, output_path: str, step_log_interval: int = 10):
        self.tb_logger = tb_logger
        self.role_name = 'learner'.lower()  # Hardcoded role name as it's always learner
        self.step_tag = f"{self.role_name}/step"
        self.iter_tag = f"{self.role_name}/iteration"
        self.step_log_interval = step_log_interval
        self._step_count = 0
        self._iter_count = 0

        # Initialize specific files upfront
        self.step_file = FileHandler(os.path.join(output_path, 'learner_steps.csv'), file_type='csv', compress=False)
        self.stats_file = FileHandler(os.path.join(output_path, 'iteration_learner_stats.csv'), file_type='csv', compress=False)

    def log_step_stats(self, stats: Dict[str, Any]):
        self._step_count += 1

        # always log to file
        record = {'step': self._step_count, **stats}
        self.step_file.log_entry(record)

        if self._step_count % self.step_log_interval == 0:
            # Log to tensorboard
            self.tb_logger.log_dict(stats, self._step_count, self.step_tag)

    def log_iteration(self, stats: Dict[str, Any]):
        self._iter_count += 1
        self.tb_logger.log_dict(stats, self._iter_count, self.iter_tag)
        record = {'iteration': self._iter_count, **stats}
        self.stats_file.log_entry(record)

    def flush(self):
        self.step_file.flush()
        self.stats_file.flush()

    def close(self):
        self.step_file.close()
        self.stats_file.close()


class TrainingTracker:
    """Main orchestrator for all tracking components"""

    def __init__(
        self,
        output_paths: Dict[str, str],
        log_intervals: Optional[Dict[str, int]] = None,  # Now used for default intervals
    ):
        assert 'tensorboard' in output_paths, "output_paths must contain 'tensorboard' key"
        assert 'samples' in output_paths, "output_paths must contain 'samples' key"
        assert 'checkpoints' in output_paths, "output_paths must contain 'checkpoints' key"

        self.output_paths = output_paths
        if log_intervals is None:
            log_intervals = {'actor': 5, 'evaluator': 5, 'learner': 1}

        # Default intervals (can be overridden by log_intervals)
        default_actor_interval = log_intervals.get('actor', 5)
        default_evaluator_interval = log_intervals.get('evaluator', 5)
        default_learner_interval = log_intervals.get('learner', 5)

        # Initialize loggers
        self.tb_logger = TensorBoardLogger(output_paths['tensorboard'])

        # Initialize role-specific trackers, passing in intervals
        self.actor = ActorTracker(self.tb_logger, output_paths['samples'], 'actor', episode_log_interval=default_actor_interval)
        self.learner = LearnerTracker(self.tb_logger, output_paths['samples'], step_log_interval=default_learner_interval)
        self.evaluator = ActorTracker(
            self.tb_logger, output_paths['samples'], 'evaluator', episode_log_interval=default_evaluator_interval
        )

    def log_params(self, config: Dict[str, Any], step: int = 0):
        """Log configuration parameters"""
        config_str = yaml.dump(config, sort_keys=False, indent=4)
        self.tb_logger.log_text('config', f"```yaml\n{config_str}\n```", step, 'parameters')

    def log_actor_step(
        self,
        data: Episode,
        for_evaluator: bool = False,
    ):
        """Log actor episode data and statistics"""
        if for_evaluator:
            self.evaluator.log_step(data)
        else:
            self.actor.log_step(data)

    def log_actor_iteration_stats(self, stats: Dict[str, Any], for_evaluator: bool = False):
        """Log actor statistics at iteration level"""
        if for_evaluator:
            self.evaluator.log_iteration(stats)
            self.evaluator.flush()
        else:
            self.actor.log_iteration(stats)
            self.actor.flush()

    def log_learner_step_stats(self, stats: Dict[str, Any]):
        """Log learner statistics at step level"""
        self.learner.log_step_stats(stats)

    def log_learner_iteration_stats(self, stats: Dict[str, Any]):
        """Log learner statistics at iteration level"""
        self.learner.log_iteration(stats)
        self.learner.flush()

    def flush(self):
        """Flush all trackers"""
        for tracker in [self.actor, self.learner, self.evaluator]:
            tracker.flush()

    def close(self):
        """Close all trackers and cleanup"""
        for tracker in [self.actor, self.learner, self.evaluator]:
            tracker.close()
