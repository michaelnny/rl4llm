"""Base trainer to optimize policy model"""

import glob
import logging
import math
import os
import shutil
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed import DeepSpeedEngine
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from typing_extensions import Self

from rl4llm.models import CustomQwen2Model
from rl4llm.types import PPOConfig, SFTConfig
from rl4llm.utils import (
    TrainingTracker,
    load_yaml_config_file,
    save_to_json_file,
    save_yaml_config_file,
    set_seed,
    setup_logging,
)


def get_checkpoint_folders(ckpt_path: str) -> list[str]:
    """Get all checkpoint folders sorted by modification time (newest first)."""
    if not os.path.exists(ckpt_path):
        return []

    # Get all subdirectories in the checkpoint path
    folders = glob.glob(os.path.join(ckpt_path, 'checkpoint_*'))
    # Sort by modification time, newest first
    folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return folders


def cleanup_old_checkpoints(ckpt_path: str, keep_n: int):
    """Remove all but the N most recent checkpoint folders."""
    folders = get_checkpoint_folders(ckpt_path)

    # Keep 'final' checkpoint and N most recent checkpoints
    for folder in folders[keep_n:]:
        try:
            shutil.rmtree(folder)
        except OSError as e:
            print(f"Error removing checkpoint {folder}: {e}")


def get_trainable_params_groups(policy_model: torch.nn.Module, optimizer_config: Dict[str, Any]) -> List[Dict]:

    opt_params = optimizer_config['params']

    # Check required optimizer parameters
    required_keys = ['betas', 'eps', 'weight_decay']
    missing_keys = [key for key in required_keys if key not in opt_params]
    if missing_keys:
        raise KeyError(f"Missing required keys in optimizer_config['params']: {missing_keys}")

    weight_decay = float(opt_params['weight_decay'])

    # Separate parameters by weight decay and model type
    lr = float(opt_params['lr'])
    base_params = []
    value_params = []
    nodecay_params = []
    for name, param in policy_model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ['norm', 'tok_embeddings']):
                nodecay_params.append(param)
            elif 'value_output' in name:
                value_params.append(param)
            else:
                base_params.append(param)

    # order is important here
    param_groups = [
        {'params': nodecay_params, 'lr': lr, 'weight_decay': 0.0, 'name': 'policy_nodecay'},
        {'params': base_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'policy'},
        {'params': value_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'value'},
    ]

    return param_groups


class BaseTrainer(ABC):
    """Base class for model training"""

    def __init__(
        self,
    ) -> None:
        self.local_rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.seed: Optional[int] = None
        self.device: Optional[torch.device] = None
        self.compute_dtype: Optional[torch.dtype] = None
        self.model_kwargs: Optional[Dict] = None
        self.policy_model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.policy_engine: Optional[DeepSpeedEngine] = None

        self.batch_size: Optional[int] = None
        self.batch_size_per_gpu: Optional[int] = None
        self.config: Optional[Dict] = None
        self.train_cfg: Union[SFTConfig, PPOConfig] = None
        self.tracker: Optional[TrainingTracker] = None
        self.pad_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.stop_tokens: Optional[List[str]] = None
        self.output_paths: Dict = None
        self.job_name: str = None
        self.ckpt_dir = None
        self.logger = None

        self.update_count = 0
        self.iteration_count = 0

    @classmethod
    def from_config(cls, config_path: str) -> Self:
        """
        Create a trainer instance from a config file.

        Args:
            config_path: Path to the YAML config file

        Returns:
            An initialized trainer instance
        """
        # Load config
        config = load_yaml_config_file(config_path)

        # Create trainer instance
        trainer = cls()
        trainer.config = config

        # Initialize all components
        trainer._setup_environment()
        trainer._initialize_paths()
        trainer._setup_logging()
        trainer._initialize_model()
        trainer._init_tokenizer()

        return trainer

    def _setup_environment(self) -> None:
        """Set up basic training environment."""
        deepspeed.init_distributed()
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.seed = int(self.config.get('job', {}).get('seed', 42)) + self.local_rank
        set_seed(self.seed)

        self.device = torch.device(f"cuda:{self.local_rank}")
        # torch.cuda.set_device(self.local_rank)

        ds_config = self.config['deepspeed']

        if 'fp16' in ds_config and ds_config['fp16']['enabled']:
            self.compute_dtype = torch.float16
        elif 'bf16' in ds_config and ds_config['bf16']['enabled']:
            self.compute_dtype = torch.bfloat16
        else:
            self.compute_dtype = torch.float32  # Or your default if no mixed precision

    def _initialize_paths(self) -> None:
        """Initialize output directories and paths."""

        if self.local_rank != 0:
            return

        job_name = self.config.get('job', {}).get('name', 'default_run')
        artifacts_path = self.config.get('job', {}).get('artifacts_path', './runs')
        self.job_name = job_name

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        workdir = f"{job_name}_{timestamp}"
        base_dir = os.path.join(artifacts_path, workdir)
        self.output_paths = {
            'base_dir': base_dir,
            'checkpoints': os.path.join(base_dir, 'checkpoints'),
            'samples': os.path.join(base_dir, 'samples'),
            'tensorboard': os.path.join(base_dir, 'tb_logs'),
            'log_file': os.path.join(base_dir, 'run.log'),
            'config_file': os.path.join(base_dir, 'config.yaml'),
        }

        self.ckpt_dir = self.output_paths['checkpoints']

        # Create directories
        for path in self.output_paths.values():
            if isinstance(path, str) and not path.endswith(('.log', '.yaml')):
                os.makedirs(path, exist_ok=True)

        # Save config
        save_yaml_config_file(self.config, self.output_paths['config_file'])

    def _setup_logging(self) -> None:
        """Initialize logging."""
        log_config = self.config.get('logging', {})
        self.logger = setup_logging(
            log_config.get('level', 'INFO'),
            log_file=self.output_paths['log_file'] if self.local_rank == 0 else None,
            rank=self.local_rank,
        )

        if self.local_rank == 0:
            self.logger.info(f"Starting job {self.job_name!r}")
            self.logger.info(f"Artifacts will be saved at {self.output_paths['base_dir']!r}")
            self.tracker = TrainingTracker(
                tb_log_dir=self.output_paths['tensorboard'],
                samples_dir=self.output_paths['samples'],
                log_intervals=self.config.get('logging', {}).get('intervals', None),
            )

            self.tracker.log_params(self.config)

    def _initialize_model(self) -> None:
        """Initialize the model with common post-processing steps."""
        model_config = self.config['model']
        self.model_name = model_config['pretrained_model']

        if model_config['load_in_4bit']:
            assert self.compute_dtype == torch.bfloat16, "only support bnb_4bit_compute_dtype = bf16"
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            nf4_config = None

        self.model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": self.compute_dtype,
            "use_cache": False,
            "attn_implementation": model_config.get('attn_implementation', 'flash_attention_2'),
            "quantization_config": nf4_config,
        }

        model = CustomQwen2Model.from_pretrained(**self.model_kwargs)
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)
            if nf4_config is not None:
                if "lm_head" in name or "embed_tokens" in name or "value_output" in name:
                    if hasattr(module, "weight"):
                        module = module.to(self.compute_dtype)

        # Setup activation checkpointing
        if model_config.get('activation_checkpoint', False):
            self.logger.info('Setup activation checkpoint...')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # checkpoint_fn = deepspeed.checkpointing.checkpoint
            # checkpoint_fn = partial(checkpoint, use_reentrant=False)
            # model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=checkpoint_fn)

        self.disable_dropout(model)
        self.policy_model = model
        # Initialize value head if needed
        if model_config.get('initialize_value_weights', False):
            self.logger.info('Initialize value head weights...')
            self._init_value_head()

        ds_config = self.config['deepspeed']

        self.batch_size_per_gpu = ds_config['train_micro_batch_size_per_gpu']
        self.batch_size = (
            ds_config['train_batch_size']
            if 'train_batch_size' in ds_config
            else self.batch_size_per_gpu * ds_config['gradient_accumulation_steps']
        )

        param_groups = get_trainable_params_groups(self.policy_model, ds_config['optimizer'])
        self.policy_engine, self.ds_optimizer, _, self.ds_scheduler = deepspeed.initialize(
            model=self.policy_model,
            model_parameters=param_groups,
            config=ds_config,
            args={"local_rank": self.local_rank},
            dist_init_required=True,
        )

        self.policy_model = self.policy_engine.module

    @staticmethod
    def disable_dropout(model: torch.nn.Module) -> None:
        """
        Disable dropout layers during training by setting their p-value to 0.
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                logging.info(f"Disabling dropout in layer: {module}")
                module.p = 0.0  # Set dropout probability to 0 to disable it

    @staticmethod
    def freeze_model(model: torch.nn.Module) -> None:
        """Mark all parameters non-trainable by set requires_grad to False."""
        logging.info('Freeze all parameters in model')
        for param in model.parameters():
            param.requires_grad = False

    def _init_tokenizer(self) -> None:
        """Initialize tokenizer and related attributes."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if 'llama' in self.model_name.lower():
            tokenizer.pad_token = '<|reserved_special_token_0|>'
            tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
        else:
            # Ensure the tokenizer has a pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = [self.tokenizer.eos_token]

    def _init_value_head(self) -> None:
        """Initialize value head weights with custom initialization."""

        if hasattr(self.policy_model, "value_output") and self.policy_model.value_output:
            num_layers = self.policy_model.config.num_hidden_layers
            for module in self.policy_model.value_output.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    def get_update_count(self) -> int:
        return self.update_count

    def get_iteration_count(self) -> int:
        return self.iteration_count

    def get_dynamic_discount(
        self,
        episode_length: int,
    ):
        """
        Dynamically calculates the discount rate with nonlinear scaling based on episode length.

        Args:
            episode_length (int): The length of the current episode.

        Returns:
            float: Adjusted discount rate.
        """
        cfg = self.train_cfg
        normalized_length = min(episode_length / cfg.max_expected_length, 1.0)
        # Apply nonlinear scaling to emphasize longer episodes
        scaled_length = normalized_length**cfg.nonlinear_scaling_factor
        gamma = cfg.min_gamma + (cfg.max_gamma - cfg.min_gamma) * scaled_length
        return gamma

    def on_exit(self):
        if self.tracker is not None:
            self.tracker.flush()
            self.tracker.close()

        self.save_checkpoint(is_final=True)

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save checkpoint and maintain only N most recent checkpoints."""
        if not (
            self.train_cfg.ckpt_enabled and self.update_count >= 1 and self.update_count % self.train_cfg.ckpt_interval == 0
        ):
            return

        dist.barrier()

        if self.local_rank != 0:
            return

        if is_best:
            # Save final checkpoint in a special folder
            save_path = os.path.join(self.ckpt_dir, 'best')
        elif is_final:
            # Save final checkpoint in a special folder
            save_path = os.path.join(self.ckpt_dir, 'final')
        else:
            # Create a new checkpoint folder with step
            checkpoint_name = f"checkpoint_{self.update_count}"
            save_path = os.path.join(self.ckpt_dir, checkpoint_name)

        os.makedirs(save_path, exist_ok=True)
        self.logger.info(f"Saving checkpoint to {save_path!r}")

        # Use DeepSpeed's save checkpoint if enabled
        self.policy_engine.save_checkpoint(save_path)

        # For native llama models
        # param_file = os.path.join(save_path, 'params.json')
        # if not os.path.exists(param_file):
        #     params = self.policy_model.params.as_dict()
        #     save_to_json_file(params, param_file)

        if not is_final:
            # Cleanup old checkpoints, keeping only N most recent
            cleanup_old_checkpoints(self.ckpt_dir, self.train_cfg.ckpt_keep_last_n)

    @abstractmethod
    def train(self, **kwargs) -> None:
        """
        Abstract method for training loop implementation.
        Should be implemented by specific trainer classes.
        """
        pass

    # --- Private Helper Methods ---

    def _prepare_model_inputs(self, input_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute casual attention mask based on the input tokens"""
        device = self.device
        attention_mask = (input_tokens != self.pad_token_id).bool()
        # batch_size, seq_len = input_tokens.size()
        # attn_mask = self.policy_model.create_causal_attention_mask(
        #     attention_mask=attention_mask,
        #     sequence_length=seq_len,
        #     target_length=seq_len,
        #     batch_size=batch_size,
        #     device=device,
        # )
        return input_tokens.to(device), attention_mask.to(device)

    @staticmethod
    def _clear_gpu_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_lr_by_group_name(self, name: str) -> float:
        """Get learning rate for a specific parameter group"""
        for group in self.policy_engine.optimizer.param_groups:
            if group['name'] == name:
                return group['lr']
        # raise ValueError(f"No parameter group found with name: {name}")
        # fallback to get lr from first group
        return self.policy_engine.optimizer.param_groups[0]['lr']

    def _get_common_stats(self) -> Dict:
        return {
            'policy/learning_rate': self._get_lr_by_group_name('policy'),
            'value/learning_rate': self._get_lr_by_group_name('value'),
        }

    def _log_batch_stats(self, batch_stats: Dict[str, Any]):
        """Log full batch stats"""
        batch_stats.update(self._get_common_stats())
        if self.tracker:
            self.tracker.log_learner_step_stats(batch_stats)

    def _aggregate_stats(self, accumulated_stats: Dict[str, List[torch.Tensor]], for_ppo: bool = False) -> Dict[str, float]:
        agg_stats = {}

        for key, values in accumulated_stats.items():
            if not values:  # Handle cases where no values were collected
                continue

            # Convert list of tensors to a single tensor
            if isinstance(values, list):
                values_tensor = torch.stack(values)
            else:
                values_tensor = values

            # Gather tensors from all ranks
            all_values = self._gather_tensor(values_tensor)

            # Compute mean across all ranks
            agg_stats[key] = torch.mean(all_values).item()

            # Compute variance if needed
            var_keys = ['objective/kl_score', 'objective/returns'] if for_ppo else []
            if key in var_keys:
                agg_stats[f"{key}_var"] = torch.var(all_values).item()

        if for_ppo:
            if 'value/error' in agg_stats and 'objective/returns_var' in agg_stats:
                agg_stats['value/var_explained'] = 1 - agg_stats['value/error'] / agg_stats['objective/returns_var']
            if 'objective/kl_score' in agg_stats and 'objective/rewards' in agg_stats:
                agg_stats['objective/total_reward'] = agg_stats['objective/kl_score'] + agg_stats['objective/rewards']
        return agg_stats

    def _gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gathers a tensor from all ranks."""
        if not dist.is_initialized():
            return tensor  # Return local tensor if not using distributed training

        output_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors)
