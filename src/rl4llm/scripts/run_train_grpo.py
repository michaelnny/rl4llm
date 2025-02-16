"""Script to run RL GRPO fine-tuning on a single GPU."""

import argparse
import sys
from traceback import format_exc

import torch

from rl4llm.core.grpo import GRPOConfig, GRPOTrainer
from rl4llm.core.helper import (
    create_model_and_tokenizer,
    create_optimizer_and_scheduler,
)
from rl4llm.data import load_and_combine_datasets
from rl4llm.utils import load_yaml_config_file, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='RL GRPO fine-tuning')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/grpo_train_config.yaml',
        # required=True,
        help='Path to the yaml file contains all the essential configuration',
    )
    return parser.parse_args()


def main():
    """Starts RL GRPO training loop."""

    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    seed = int(config.get('job').get('seed', 142))
    artifacts_path = config.get('job').get('artifacts_path')
    max_samples = config.get('job').get('max_samples', None)
    set_seed(seed)

    logger = setup_logger()
    grpo_config = GRPOConfig(**config['grpo_config'])

    train_ds, _ = load_and_combine_datasets(config['datasets'])

    if max_samples is not None and max_samples < len(train_ds):
        logger.info(f"Randomly select {max_samples} training samples")
        train_ds = train_ds.shuffle().select(range(max_samples))

    torch_dtype = torch.bfloat16
    device = torch.device('cuda')

    policy_model, tokenizer = create_model_and_tokenizer(config['model'], torch_dtype)

    optimizer, scheduler = create_optimizer_and_scheduler(
        policy_model,
        optimizer_config=config['optimizer'],
        scheduler_config=config.get('scheduler', None),
        total_steps=grpo_config.max_steps,
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
        artifacts_path=artifacts_path,
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
