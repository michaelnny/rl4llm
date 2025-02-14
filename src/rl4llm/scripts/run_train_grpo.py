"""Script to run RL GRPO training loop."""

import argparse
import os
import sys
from traceback import format_exc
from typing import Dict, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer, set_seed

from rl4llm.core.grpo import GRPOConfig, GRPOTrainer
from rl4llm.data import load_and_combine_datasets
from rl4llm.utils import load_yaml_config_file, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='RL GRPO fine-tuning')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/grpo_train_config.yaml',
        # required=True,
        help='Path to the yaml file contains all the essential configuration',
    )
    # # Include DeepSpeed configuration arguments
    # parser.add_argument(
    #     '--local_rank',
    #     type=int,
    #     default=-1,
    #     help='Required by deepspeed for local rank passed from distributed launcher, not used by our script',
    # )
    return parser.parse_args()


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

    scheduler_type = scheduler_config['type']
    scheduler_params = scheduler_config['params']

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

    scheduler = create_scheduler(optimizer, max_lr=lr, total_steps=total_steps, **scheduler_params)
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


def main():
    """Starts RL GRPO training loop."""

    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    logger = setup_logger()
    grpo_config = GRPOConfig(**config['grpo_config'])
    set_seed(grpo_config.seed)

    train_ds, _ = load_and_combine_datasets(config['datasets'])

    torch_dtype = torch.bfloat16
    device = torch.device('cuda')

    policy_model, tokenizer = create_model_and_tokenizer(config['model'], torch_dtype)

    # # compute the total update steps for LR scheduler
    # total_steps = int(
    #     grpo_config.max_iterations * grpo_config.rollout_size / (grpo_config.batch_size * grpo_config.gradient_accumulate_steps)
    # )

    optimizer, scheduler = create_optimizer_and_scheduler(
        policy_model,
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler'],
        total_steps=grpo_config.max_iterations,
    )

    trainer = GRPOTrainer(
        config=grpo_config,
        policy_model=policy_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ds=train_ds,
        device=device,
        torch_dtype=torch_dtype,
    )

    try:
        trainer.train(hyper_params=config)
    except KeyboardInterrupt:
        logger.info('\nKeyboardInterrupt received in main loop. Shutting down...')
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(format_exc())
        sys.exit(1)
    finally:
        logger.info('Exiting main program.')


if __name__ == '__main__':
    main()
