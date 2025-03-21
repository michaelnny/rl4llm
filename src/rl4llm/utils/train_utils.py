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

from rl4llm.models import ClassifierModel

logger = logging.getLogger()


def build_longformer_classification_model_and_tokenizer(
    model_config: Dict,
    torch_dtype: torch.dtype,
) -> PreTrainedModel:
    """Build a binary classification model from a pretrained model."""
    model_name = model_config['pretrained_model']
    model_name = model_config['pretrained_model']
    load_in_4bit = model_config['load_in_4bit']
    gradient_checkpointing = model_config['gradient_checkpointing']

    logger.info(f"Loading model and tokenizer for {model_name!r}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    assert tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= 1
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= 1

    model_args = {
        'pretrained_model_name_or_path': model_name,
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
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    return model, tokenizer


def build_classification_model_and_tokenizer(
    model_config: Dict,
    torch_dtype: torch.dtype,
) -> ClassifierModel:
    """Build a binary classification model from a pretrained model."""
    model, tokenizer = build_model_and_tokenizer(model_config, torch_dtype)
    dropout_prob = model_config.get('dropout_prob', 0.0)
    classifier_model = ClassifierModel(model, dropout_prob)
    return classifier_model, tokenizer


def build_model_and_tokenizer(model_config: Dict, torch_dtype: torch.dtype) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Creates the model and tokenizer from the given configuration."""

    model_name = model_config['pretrained_model']
    load_in_4bit = model_config['load_in_4bit']
    gradient_checkpointing = model_config['gradient_checkpointing']
    flash_attention = model_config.get('flash_attention', None)

    logger.info(f"Loading model and tokenizer for {model_name!r}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    assert tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= 1
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= 1

    model_args = {
        'pretrained_model_name_or_path': model_name,
        'torch_dtype': torch_dtype,
        'use_cache': False,
        'attn_implementation': 'flash_attention_2' if flash_attention else 'eager',
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
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    return model, tokenizer


def get_trainable_param_groups(model: PreTrainedModel, learning_rate: float, weight_decay: float) -> List[Dict]:

    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ['bias', 'layer_norm.weight', 'layernorm.weight', 'norm.weight']):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

    optim_groups = [
        {'params': nodecay_params, 'lr': learning_rate, 'weight_decay': 0.0, 'name': 'nodecay'},
        {'params': decay_params, 'lr': learning_rate, 'weight_decay': weight_decay, 'name': 'decay'},
    ]

    return optim_groups


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


def compute_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    total_norm = torch.tensor(0.0)
    for p in model.parameters():
        if p.grad is not None:
            # Detach the gradient tensor before computing the norm
            grad_detached = p.grad.detach()
            local_norm = torch.linalg.vector_norm(grad_detached, dtype=p.dtype)
            if total_norm.device != local_norm.device:
                total_norm = total_norm.to(local_norm.device)
            total_norm += local_norm**2
    return total_norm**0.5


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
