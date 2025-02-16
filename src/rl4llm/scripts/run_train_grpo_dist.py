"""Script to run RL GRPO fine-tuning on multiple GPUs using DeepSpeed."""

import argparse
import os
import sys
from traceback import format_exc

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim.lr_scheduler import OneCycleLR

from rl4llm.core.helper import create_model_and_tokenizer
from rl4llm.data import load_and_combine_datasets
from rl4llm.utils import load_yaml_config_file, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='RL GRPO fine-tuning')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/ppo_train_config.yaml',
        # required=True,
        help='Path to the yaml file contains all the essential configuration',
    )
    # Include DeepSpeed configuration arguments
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Required by deepspeed for local rank passed from distributed launcher, not used by our script',
    )
    return parser.parse_args()


def main():
    """Starts RL GRPO training loop."""

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed(verbose=False)

    local_rank = int(os.environ['LOCAL_RANK'])  # Get LOCAL_RANK from environment variable set by DeepSpeed launcher
    world_size = int(os.environ['WORLD_SIZE'])

    seed = int(config.get('job', {}).get('seed', 142)) + local_rank
    set_seed(seed)

    logger = setup_logger() if local_rank == 0 else None

    train_ds, _ = load_and_combine_datasets(config['datasets'])

    # Set device for each process using local_rank
    torch.cuda.set_device(local_rank)

    try:
        pass
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
