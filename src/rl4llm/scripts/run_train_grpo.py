"""Script to run RL GRPO fine-tuning on a single GPU."""

import argparse
import cProfile
import os
import pstats
import sys
from traceback import format_exc

import torch

from rl4llm.core.grpo import GRPOConfig, GRPOTrainer
from rl4llm.data import load_multiple_datasets
from rl4llm.graders import FormatGrader, MathGrader
from rl4llm.utils import (
    build_model_and_tokenizer,
    create_optimizer_and_scheduler,
    load_yaml_config_file,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description='RL GRPO fine-tuning')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/explore_grpo_config.yaml',
        # required=True,
        help='Path to the yaml file contains all the essential configuration',
    )
    return parser.parse_args()


def main():
    """Starts RL GRPO training loop."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This script only supports run on GPU with BF16 mode.')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    seed = int(config.get('job').get('seed', 142))
    artifacts_path = config.get('job').get('artifacts_path')
    datasets = config.get('job').get('datasets')
    max_train_samples = config.get('job').get('max_train_samples', None)
    max_test_samples = config.get('job').get('max_test_samples', None)
    set_seed(seed)

    logger = setup_logger()
    grpo_config = GRPOConfig(**config['grpo_config'])

    train_ds, test_ds = load_multiple_datasets(datasets)

    if max_train_samples is not None and max_train_samples < len(train_ds):
        logger.info(f"Randomly select {max_train_samples} training samples")
        train_ds = train_ds.shuffle().select(range(max_train_samples))
    else:
        logger.info(f'Number of training samples: {len(train_ds)}')

    if max_test_samples is not None and max_test_samples < len(test_ds):
        logger.info(f"Randomly select {max_test_samples} testing samples")
        test_ds = test_ds.shuffle().select(range(max_test_samples))
    else:
        logger.info(f'Number of testing samples: {len(test_ds)}')

    device = torch.device('cuda')
    torch_dtype = torch.bfloat16

    # compute the total update steps for LR scheduler
    total_update_steps = int(
        grpo_config.max_steps * grpo_config.rollout_size / (grpo_config.batch_size * grpo_config.gradient_accumulate_steps)
    )

    policy_model, tokenizer = build_model_and_tokenizer(config['model'], torch_dtype)

    optimizer, scheduler = create_optimizer_and_scheduler(
        policy_model,
        optimizer_config=config['optimizer'],
        scheduler_config=config.get('scheduler', None),
        total_steps=total_update_steps,
    )

    trainer = GRPOTrainer(
        config=grpo_config,
        math_grader=MathGrader(),
        format_grader=FormatGrader(config['coherent_classification_model'], torch_dtype, device),
        policy_model=policy_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ds=train_ds,
        test_ds=test_ds,
        device=device,
        torch_dtype=torch_dtype,
        artifacts_path=artifacts_path,
        logger=logger,
    )

    # # Create a profiler instance
    # profiler = cProfile.Profile()
    # profiler.enable()

    def handle_exit():
        trainer.on_exit()

        # if profiler is not None:
        #     profiler.disable()
        #     # Save profiling stats
        #     stats = pstats.Stats(profiler)
        #     stats.sort_stats('cumulative').dump_stats('profile_stats.prof')
        #     stats.print_stats()  # Print to console for immediate feedback

    try:
        trainer.train(log_hyper_params=config)
    except KeyboardInterrupt:
        logger.info('\nKeyboardInterrupt received in main loop. Shutting down...')
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(format_exc())
        sys.exit(1)
    finally:
        handle_exit()
        logger.info('Exiting main program.')


if __name__ == '__main__':
    main()
