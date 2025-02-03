"""Runs the first-stage SFT on collected reasoning samples."""

import argparse
import os
import sys
from traceback import format_exc

import deepspeed
import torch
import torch.distributed as dist

from rl4llm.core.sft_learner import SFTLearner
from rl4llm.utils import load_yaml_config_file, set_seed, setup_tracker_and_logger


def parse_args():
    parser = argparse.ArgumentParser(description='SFT for LLM on single GPU')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/sft_train_config.yaml',
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


def main(config_file=None):

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Parse arguments if no `config_file` is provided programmatically
    if config_file is None:
        args = parse_args()
    else:
        # Create a namespace manually if called with `config_file`
        args = argparse.Namespace(config_file=config_file)

    config = load_yaml_config_file(args.config_file)

    local_rank = int(os.environ['LOCAL_RANK'])  # Get LOCAL_RANK from environment variable set by DeepSpeed launcher
    world_size = int(os.environ['WORLD_SIZE'])

    seed = int(config.get('job', {}).get('seed', 142)) + local_rank
    set_seed(seed)

    tracker, logger = setup_tracker_and_logger(config, local_rank)

    if tracker:
        tracker.log_params(config)

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed()

    # Set device for each process using local_rank
    torch.cuda.set_device(local_rank)

    # Initialize the SFTLearner
    learner = SFTLearner(config=config, local_rank=local_rank, tracker=tracker, logger=logger)
    dist.barrier()
    
    def handle_exit():
        learner.on_exit()

    try:
        learner.train()
    except KeyboardInterrupt:
        logger.info('\nKeyboardInterrupt received in main loop. Shutting down...')
        handle_exit()
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(format_exc())
        handle_exit()
        sys.exit(1)
    finally:
        logger.info('Exiting main program.')
        handle_exit()


if __name__ == '__main__':
    main()
