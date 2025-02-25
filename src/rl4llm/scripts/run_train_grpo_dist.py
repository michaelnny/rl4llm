"""Script to run RL GRPO fine-tuning on multiple GPUs using DeepSpeed."""

import argparse
import os
import sys
from copy import deepcopy
from traceback import format_exc

import deepspeed
import torch
import torch.distributed as dist

from rl4llm.core.grpo_dist import GRPOConfig, GRPOTrainer
from rl4llm.data import load_and_combine_datasets
from rl4llm.utils import (
    DummyLogger,
    create_model_and_tokenizer,
    get_trainable_param_groups,
    load_yaml_config_file,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description='RL GRPO fine-tuning')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/ds_grpo_train_config.yaml',
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
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This script only supports run on GPU with BF16 mode.')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    deepspeed_config = config['deepspeed_config']

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed(verbose=False)

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed = int(config.get('job', {}).get('seed', 143))
    artifacts_path = config.get('job').get('artifacts_path')
    datasets = config.get('job').get('datasets')
    max_train_samples = config.get('job').get('max_train_samples', None)
    max_test_samples = config.get('job').get('max_test_samples', None)
    set_seed(seed)

    # Set device for each process using local_rank
    torch.cuda.set_device(local_rank)

    logger = setup_logger() if local_rank == 0 else DummyLogger()

    train_ds, test_ds = load_and_combine_datasets(datasets)

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

    # Ensure the number of samples in the test dataset can be evenly divided by the world_size
    if len(test_ds) % world_size != 0:
        new_test_sample_size = (len(test_ds) // world_size) * world_size
        logger.info(f"Adjusting test dataset size to {new_test_sample_size} to be evenly divisible by world size {world_size}")
        test_ds = test_ds.select(range(new_test_sample_size))

    # shard datasets across ranks, so each rank only works on a small subset of the data
    shared_train_ds = train_ds.shard(world_size, local_rank)
    shared_test_ds = test_ds.shard(world_size, local_rank)
    logger.info(
        f"Rank {local_rank} has {len(shared_train_ds)} training and {len(shared_test_ds)} testing samples after sharding"
    )

    torch_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    dist.barrier()
    policy_model, tokenizer = create_model_and_tokenizer(config['model'], torch_dtype)

    policy_engine, *_ = deepspeed.initialize(
        model=policy_model,
        model_parameters=get_trainable_param_groups(
            policy_model, deepspeed_config['optimizer']['params']['lr'], deepspeed_config['optimizer']['params']['weight_decay']
        ),
        config_params=deepspeed_config,
    )

    grpo_config = GRPOConfig(**config['grpo_config'], batch_size=deepspeed_config['train_micro_batch_size_per_gpu'])

    trainer = GRPOTrainer(
        config=grpo_config,
        policy_engine=policy_engine,
        tokenizer=tokenizer,
        train_ds=shared_train_ds,
        test_ds=shared_test_ds,
        device=device,
        torch_dtype=torch_dtype,
        artifacts_path=artifacts_path,
        logger=logger,
    )

    dist.barrier()

    try:
        trainer.train(log_hyper_params=config)
    except KeyboardInterrupt:
        logger.info('\nKeyboardInterrupt received in main loop. Shutting down...')
        trainer.on_exit()
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(format_exc())
        trainer.on_exit()
        sys.exit(1)
    finally:
        trainer.on_exit()
        logger.info('Exiting main program.')


if __name__ == '__main__':
    main()
