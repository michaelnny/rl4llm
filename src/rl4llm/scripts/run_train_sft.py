"""Runs the first-stage SFT on collected reasoning samples."""

import argparse
import sys
from traceback import format_exc


from rl4llm.core.sft_trainer import SFTTrainer


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

    # Parse arguments if no `config_file` is provided programmatically
    if config_file is None:
        args = parse_args()
    else:
        # Create a namespace manually if called with `config_file`
        args = argparse.Namespace(config_file=config_file)

    trainer = SFTTrainer.from_config(args.config_file)
    logger = trainer.logger

    def handle_exit():
        trainer.on_exit()

    try:
        trainer.train()
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
