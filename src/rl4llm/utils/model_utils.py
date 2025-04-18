import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LongformerForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from rl4llm.models import AutoModelWithValueHead

logger = logging.getLogger(__name__)


def build_value_model_and_tokenizer(
    model_config: Dict, torch_dtype: torch.dtype
) -> Tuple[AutoModelWithValueHead, PreTrainedTokenizer]:
    """Creates the value model and tokenizer from the given configuration."""

    model_name = model_config['pretrained_model']
    checkpoint_path = model_config.get('checkpoint_path', None)
    load_in_4bit = model_config.get('load_in_4bit', False)
    gradient_checkpointing = model_config.get('gradient_checkpointing', False)
    flash_attention = model_config.get('flash_attention', None)
    model_max_length = model_config.get('model_max_length', None)

    logger.info(f"Loading value model and tokenizer for {model_name!r}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True if checkpoint_path else False,
    )

    if model_max_length:
        tokenizer.model_max_length = model_max_length

    assert tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= 1
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= 1

    model_args = {
        'pretrained_model_name_or_path': (
            checkpoint_path if checkpoint_path else model_name
        ),
        'local_files_only': True if checkpoint_path else False,
        'torch_dtype': torch_dtype,
        'use_cache': False,
        'attn_implementation': (
            'flash_attention_2' if flash_attention else 'eager'
        ),
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    if load_in_4bit:
        model_args['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model: AutoModelWithValueHead = AutoModelWithValueHead.from_pretrained(
        **model_args,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )

    return model, tokenizer


def build_policy_model_and_tokenizer(
    model_config: Dict, torch_dtype: torch.dtype
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Creates the policy model and tokenizer from the given configuration."""

    model_name = model_config['pretrained_model']
    checkpoint_path = model_config.get('checkpoint_path', None)
    load_in_4bit = model_config.get('load_in_4bit', False)
    gradient_checkpointing = model_config.get('gradient_checkpointing', False)
    flash_attention = model_config.get('flash_attention', None)
    model_max_length = model_config.get('model_max_length', None)

    logger.info(f"Loading policy model and tokenizer for {model_name!r}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True if checkpoint_path else False,
    )

    if model_max_length:
        tokenizer.model_max_length = model_max_length

    assert tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= 1
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= 1

    model_args = {
        'pretrained_model_name_or_path': (
            checkpoint_path if checkpoint_path else model_name
        ),
        'local_files_only': True if checkpoint_path else False,
        'torch_dtype': torch_dtype,
        'use_cache': False,
        'attn_implementation': (
            'flash_attention_2' if flash_attention else 'eager'
        ),
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    if load_in_4bit:
        model_args['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**model_args)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )

    return model, tokenizer


def build_longformer_classification_model_and_tokenizer(
    model_config: Dict,
    torch_dtype: torch.dtype,
) -> Tuple[LongformerForSequenceClassification, PreTrainedTokenizer]:
    """Build a binary classification model from a pretrained model."""
    model_name = model_config['pretrained_model']
    checkpoint_path = model_config.get('checkpoint_path', None)
    load_in_4bit = model_config.get('load_in_4bit', False)
    gradient_checkpointing = model_config.get('gradient_checkpointing', False)
    model_max_length = model_config.get('model_max_length', None)

    logger.info(f"Loading model and tokenizer for {model_name!r}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True if checkpoint_path else False,
    )

    if model_max_length:
        tokenizer.model_max_length = model_max_length

    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    assert tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= 1
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= 1

    model_args = {
        'pretrained_model_name_or_path': (
            checkpoint_path if checkpoint_path else model_name
        ),
        'local_files_only': True if checkpoint_path else False,
        'torch_dtype': torch_dtype,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    if load_in_4bit:
        model_args['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
    label2id = {'NEGATIVE': 0, 'POSITIVE': 1}
    model = LongformerForSequenceClassification.from_pretrained(
        **model_args, num_labels=2, id2label=id2label, label2id=label2id
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )

    return model, tokenizer


def get_trainable_param_groups(
    model: PreTrainedModel, learning_rate: float, weight_decay: float
) -> List[Dict]:
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(
                nd in name
                for nd in [
                    'bias',
                    'layer_norm.weight',
                    'layernorm.weight',
                    'norm.weight',
                ]
            ):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

    optim_groups = [
        {
            'params': nodecay_params,
            'lr': learning_rate,
            'weight_decay': 0.0,
            'name': 'nodecay',
        },
        {
            'params': decay_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'name': 'decay',
        },
    ]

    return optim_groups


def create_optimizer_and_scheduler(
    policy_model: PreTrainedModel,
    optimizer_config: Dict,
    scheduler_config: Dict,
    total_steps: int,
) -> Tuple[Optimizer, OneCycleLR]:
    """Creates the optimizer and scheduler from the given configuration."""

    optim_type = optimizer_config['type']
    opt_params = optimizer_config['params']
    lr = float(opt_params['lr'])
    eps = float(opt_params['eps'])
    weight_decay = float(opt_params['weight_decay'])
    betas = opt_params['betas']

    optim_groups = get_trainable_param_groups(policy_model, lr, weight_decay)

    optim_kwargs = {'lr': lr, 'eps': eps, 'betas': betas}

    if optim_type == 'AdamW8bit':
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(optim_groups, **optim_kwargs)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups, **optim_kwargs)
    else:
        optimizer = torch.optim.Adam(optim_groups, **optim_kwargs)

    if scheduler_config is not None:
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config['params']
        scheduler = create_scheduler(
            optimizer, max_lr=lr, total_steps=total_steps, **scheduler_params
        )
    else:
        scheduler = None
    return optimizer, scheduler


def create_scheduler(
    optimizer: Optimizer,
    max_lr: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    initial_lr_fraction: float = 0.1,
    final_lr_fraction: float = 0.01,
) -> OneCycleLR:
    """
    Creates a OneCycleLR scheduler with warmup and cosine decay.

    Args:
        optimizer: The optimizer to use
        max_lr: Maximum learning rate after warmup
        total_steps: Total number of training steps
        warmup_fraction: Fraction of total steps used for warmup (default: 0.3)
        initial_lr_fraction: Fraction of max_lr to use as the initial learning rate (default: 0.1)
        final_lr_fraction: Fraction of max_lr to use as the final learning rate (default: 0.01)

    Returns:
        OneCycleLR: a OneCycleLR scheduler with warmup and cosine decay
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=int(total_steps),
        pct_start=warmup_fraction,
        div_factor=1 / initial_lr_fraction,
        final_div_factor=1 / (initial_lr_fraction * final_lr_fraction),
        anneal_strategy='cos',
    )
