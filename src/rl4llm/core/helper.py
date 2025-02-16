from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def create_model_and_tokenizer(model_config: Dict, torch_dtype: torch.dtype) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Creates the model and tokenizer from the given configuration."""

    model_name = model_config['pretrained_model']
    load_in_4bit = model_config['load_in_4bit']
    gradient_checkpointing = model_config['gradient_checkpointing']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    assert tokenizer.eos_token_id is not None and tokenizer.eos_token_id > 1
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id > 1

    model_args = {
        'pretrained_model_name_or_path': model_name,
        'torch_dtype': torch_dtype,
        'use_cache': False,
        'attn_implementation': 'flash_attention_2',
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

    model = AutoModelForCausalLM.from_pretrained(**model_args)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    return model, tokenizer


def create_optimizer_and_scheduler(
    policy_model: PreTrainedModel, optimizer_config: Dict, scheduler_config: Dict, total_steps: int
) -> Tuple[Optimizer, OneCycleLR]:
    """Creates the optimizer and scheduler from the given configuration."""

    optim_type = optimizer_config['type']
    opt_params = optimizer_config['params']
    lr = float(opt_params['lr'])
    eps = float(opt_params['eps'])
    weight_decay = float(opt_params['weight_decay'])
    betas = opt_params['betas']

    decay_params = []
    nodecay_params = []
    for name, param in policy_model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ['bias', 'layer_norm.weight', 'layernorm.weight', 'norm.weight']):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

    optim_groups = [
        {'params': nodecay_params, 'lr': lr, 'weight_decay': 0.0, 'name': 'nodecay'},
        {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'decay'},
    ]

    optim_kwargs = {'lr': lr, 'eps': eps, 'betas': betas}

    if optim_type == 'AdamW8bit':
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(optim_groups, **optim_kwargs)
    else:
        optimizer = torch.optim.AdamW(optim_groups, **optim_kwargs)

    if scheduler_config is not None:
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config['params']
        scheduler = create_scheduler(optimizer, max_lr=lr, total_steps=total_steps, **scheduler_params)
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


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    if dim is not None:
        return (values * mask).sum(dim=dim, keepdim=True)
    else:
        return (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    if dim is not None:
        return (values * mask).sum(dim=dim, keepdim=True) / (mask.sum(dim=dim, keepdim=True) + 1e-8)
    else:
        return (values * mask).sum() / (mask.sum() + 1e-8)


def whiten(values: torch.FloatTensor, shift_mean: bool = True, dim: int = -1) -> torch.Tensor:
    # Compute the mean and variance along the specified dimension
    mean = values.mean(dim=dim, keepdim=True)
    var = values.var(dim=dim, unbiased=False, keepdim=True)

    # Perform whitening (normalize)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)

    # If shift_mean is False, add back the mean
    if not shift_mean:
        whitened += mean
    return whitened


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True, dim: int = -1) -> torch.Tensor:
    """Whiten values with masked values.

    Args:
        values: Input tensor of shape [batch_size, sequence_length]
        mask: Boolean mask of same shape as values
        shift_mean: Whether to shift the mean to zero
        dim: Dimension along which to perform whitening (default: -1 for sequence dimension)
    """
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    output = values.clone()

    valid_values = values[mask]

    # Whiten the valid values
    valid_values = whiten(valid_values, shift_mean, dim)

    output[mask] = valid_values

    return output
