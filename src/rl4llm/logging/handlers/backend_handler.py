"""Handler for logging metrics to Tensorboard"""

import logging
import os
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

from rl4llm.core.distributed import DistributedManager
from rl4llm.logging.handlers.base_handler import BaseHandler


class BackendHandler(BaseHandler):
    """
    Handles logging metrics, samples, and hyperparameters to WandB/TensorBoard.
    Manages the creation and lifecycle of the backend writer. Operates only on master.
    """

    def __init__(
        self,
        log_dir: str,
        enable_wandb: bool,
        enable_tensorboard: bool,
        is_master: bool,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the BackendHandler and sets up the writer if on master rank.

        Args:
            log_dir: Base directory for logs (used for WandB/TB paths).
            enable_wandb: Flag to enable WandB.
            enable_tensorboard: Flag to enable TensorBoard.
            is_master: Boolean indicating if the current process is the master.
            logger: Logger instance.
        """
        super().__init__(logger)
        self.is_master = is_master
        self.log_dir = log_dir
        self._writer: Optional[Any] = None
        self._can_log_flag: bool = False

        if self.is_master:
            self._writer = self._setup_backend_writer(
                enable_wandb, enable_tensorboard
            )
            self._can_log_flag = self._writer is not None
        else:
            self._logger.debug(
                'Not master rank, skipping backend writer setup.'
            )

    def _setup_backend_writer(
        self, enable_wandb: bool, enable_tensorboard: bool
    ) -> Optional[Any]:
        """Initializes WandB or TensorBoard writer. Called only on master."""
        backend_writer = None
        backend_name = 'None'

        # WandB Initialization
        if enable_wandb:
            try:
                import wandb

                wandb_dir = os.path.join(self.log_dir, 'wandb')
                os.makedirs(wandb_dir, exist_ok=True)
                run_name = f"rl4llm_run_{time.strftime('%Y%m%d_%H%M%S')}"

                run_id = wandb.util.generate_id()

                wandb.init(
                    project='rl4llm_project',
                    dir=self.log_dir,
                    sync_tensorboard=False,
                    name=run_name,
                    id=run_id,
                    resume='allow',
                    save_code=False,
                    settings=wandb.Settings(
                        _stats_sample_rate_seconds=300,
                        _stats_disk_paths=[self.log_dir],
                        log_internal=os.path.join(
                            wandb_dir, 'wandb-internal.log'
                        ),
                        sync_file=os.path.join(wandb_dir, 'wandb-sync.json'),
                        files_dir=wandb_dir,
                    ),
                )
                backend_writer = wandb
                backend_name = 'WandB'
                self._logger.info(
                    f"Initialized WandB. Run name: {run_name}, Run ID: {run_id}"
                )
                if wandb.run:
                    self._logger.info(f"WandB dashboard: {wandb.run.get_url()}")

            except ImportError:
                self._logger.warning(
                    'WandB requested but `wandb` package not installed. Skipping.'
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize WandB: {e}. Disabling WandB logging."
                )

        # TensorBoard Initialization
        if enable_tensorboard and not backend_writer:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_log_dir = os.path.join(self.log_dir, 'tensorboard')
                os.makedirs(tb_log_dir, exist_ok=True)
                backend_writer = SummaryWriter(log_dir=tb_log_dir)
                backend_name = 'TensorBoard'
                self._logger.info(
                    f"Initialized TensorBoard. Logs in: {tb_log_dir}"
                )
                try:
                    abs_log_dir = os.path.abspath(tb_log_dir)
                    self._logger.info(
                        f"To view TensorBoard, run: tensorboard --logdir '{abs_log_dir}'"
                    )
                except Exception:
                    pass

            except ImportError:
                self._logger.warning(
                    'TensorBoard requested but `tensorboard` package not installed. Skipping.'
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize TensorBoard: {e}. Disabling TensorBoard logging."
                )

        if not backend_writer:
            self._logger.warning(
                'No backend logger (WandB/TensorBoard) was initialized.'
            )
        else:
            self._logger.info(f"Using {backend_name} for backend logging.")

        return backend_writer

    def _is_wandb_writer(self) -> bool:
        return self._writer is not None and hasattr(self._writer, 'log')

    def _is_tensorboard_writer(self) -> bool:
        return self._writer is not None and hasattr(self._writer, 'add_scalar')

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: int
    ) -> None:
        """Logs multiple scalar metrics to the backend."""
        if not self._can_log_flag or not metrics:
            return
        try:
            valid_metrics = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            }
            if not valid_metrics:
                self._logger.debug(
                    f"No valid finite metrics to log at step {step}"
                )
                return

            if self._is_wandb_writer():
                self._writer.log(valid_metrics, step=step)
            elif self._is_tensorboard_writer():
                for name, value in valid_metrics.items():
                    self._writer.add_scalar(name, value, step)
            self._logger.debug(
                f"Logged {len(valid_metrics)} metrics to backend at step {step}"
            )
        except Exception as e:
            self._logger.warning(
                f"Failed to log metrics to backend at step {step}: {e}"
            )

    def log_sample_text(self, tag: str, formatted_text: str, step: int) -> None:
        """Logs formatted text (representing a sample) to the backend."""
        if not self._can_log_flag:
            return
        try:
            log_tag = f"samples/{tag}"
            if self._is_wandb_writer():
                import wandb

                html_text = formatted_text.replace('\n\n', '<br><br>').replace(
                    '\n', '<br>'
                )
                self._writer.log({log_tag: wandb.Html(html_text)}, step=step)
            elif hasattr(self._writer, 'add_text'):  # TensorBoard
                self._writer.add_text(log_tag, formatted_text, step)
            self._logger.debug(
                f"Logged sample text '{log_tag}' to backend at step {step}"
            )
        except Exception as e:
            self._logger.warning(
                f"Failed to log sample text (tag: {tag}) to backend at step {step}: {e}"
            )

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters to the backend."""
        if not self._can_log_flag:
            return
        try:
            if self._is_wandb_writer():
                self._writer.config.update(params, allow_val_change=True)
            elif hasattr(self._writer, 'add_text'):
                params_str = yaml.dump(params, sort_keys=False, indent=2)
                self._writer.add_text(
                    'configuration/hyperparameters',
                    f"```yaml\n{params_str}\n```",
                    0,
                )
                self._logger.info(
                    'Logged hyperparameters using add_text (TensorBoard).'
                )
            self._logger.debug('Logged hyperparameters to backend.')
        except Exception as e:
            self._logger.warning(
                f"Failed to log hyperparameters to backend: {e}"
            )

    def close(self) -> None:
        """Closes the backend writer if it exists."""
        if not self._can_log_flag or not self._writer:
            return
        try:
            if hasattr(self._writer, 'finish'):  # WandB
                self._writer.finish()
                self._logger.info('Closed WandB writer.')
            elif hasattr(self._writer, 'close'):  # TensorBoard
                self._writer.close()
                self._logger.info('Closed TensorBoard writer.')
            self._writer = None  # Release writer reference
            self._can_log_flag = False
        except Exception as e:
            self._logger.error(f"Failed to close backend writer: {e}")
