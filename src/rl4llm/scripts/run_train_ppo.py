"""Runs RL PPO."""

import argparse
import os
import sys
from traceback import format_exc

import deepspeed
import torch
import torch.distributed as dist

from rl4llm.core.actor import Actor
from rl4llm.core.ppo_learner import PPOLearner
from rl4llm.data import load_and_combine_datasets
from rl4llm.envs import VectorEnvWrapper
from rl4llm.types import DecodingConfig
from rl4llm.utils import load_yaml_config_file, set_seed, setup_tracker_and_logger


def parse_args():
    parser = argparse.ArgumentParser(description='RL PPO fine-tuning')
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
    deepspeed.init_distributed(verbose=False)

    # Set device for each process using local_rank
    torch.cuda.set_device(local_rank)

    learner = PPOLearner(config=config, local_rank=local_rank, tracker=tracker, logger=logger)

    train_config = config['training_config']
    actor_config = config['actor']
    env_config = config['environment']
    eval_config = config['evaluator']

    train_decoding = DecodingConfig(**actor_config['decoding'])
    eval_decoding = DecodingConfig(**eval_config['decoding'])

    train_ds, test_ds = load_and_combine_datasets(config['datasets'])

    env_kwargs = {'stop_tokens': learner.stop_tokens, 'seed': seed, **env_config}

    train_env_kwargs = {
        'datasets': train_ds,
        **env_kwargs,
    }
    eval_env_kwargs = {
        'datasets': test_ds,
        **env_kwargs,
    }
    train_env = VectorEnvWrapper(**train_env_kwargs)
    eval_env = VectorEnvWrapper(**eval_env_kwargs)

    actor = Actor(config=config, local_rank=local_rank, dtype=learner.dtype, tracker=tracker, logger=logger)

    best_eval_accuracy = 0.0

    train_rollout_episodes = train_config.get('rollout_episodes', 1024) // world_size
    num_iters = train_config.get('rollout_iterations', 10000)
    eval_rollout_episodes = eval_config.get('rollout_episodes', 500)
    eval_enabled = eval_config.get('enabled', False)
    eval_interval = eval_config.get('interval', 100)

    def handle_exit():
        learner.on_exit()

    latest_state_dict = None
    iter_c = 0
    try:
        # kick start the training
        while iter_c < num_iters:
            logger.info(f"Start iteration {iter_c}")

            # move training model to cpu for inference
            learner.offload_for_inference()

            if latest_state_dict is not None:
                actor.sync_model_weights(latest_state_dict)
                dist.barrier()

            episodes, _ = actor.generate_samples(
                vector_env=train_env,
                decoding=train_decoding,
                max_episodes=train_rollout_episodes,
            )

            actor.offload_for_training()
            dist.barrier()

            # step 2: Train on collected episodes
            learner.train(episodes)

            # get latest weights to pass to actor for generation
            latest_state_dict = learner.get_lasted_policy_weights()
            iter_c += 1

            # step 3: evaluation and checkpoint
            if eval_enabled and iter_c >= 1 and iter_c % eval_interval == 0 and local_rank == 0:
                logger.info('Run evaluation')

                if latest_state_dict is not None:
                    actor.sync_model_weights(latest_state_dict)

                _, eval_stats = actor.generate_samples(
                    vector_env=eval_env,
                    decoding=eval_decoding,
                    max_episodes=min(eval_rollout_episodes, len(test_ds)),
                    for_evaluator=True,
                )

                if 'accuracy' in eval_stats and eval_stats['accuracy'] > best_eval_accuracy:
                    best_eval_accuracy = eval_stats['accuracy']
                    logger.info(f"Best policy model with eval accuracy {best_eval_accuracy:.4f}")
                    learner.save_policy_model(tag="best")

            # save data to external files
            tracker.flush()
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
