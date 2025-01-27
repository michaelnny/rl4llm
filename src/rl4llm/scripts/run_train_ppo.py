"""Runs RL PPO."""

import argparse
import os
import sys
from traceback import format_exc

import deepspeed
import deepspeed.comm as dist

from rl4llm.core.actor import EgreedyActor
from rl4llm.core.ppo_trainer import PPOTrainer
from rl4llm.data import load_and_combine_datasets
from rl4llm.envs import VectorEnvWrapper
from rl4llm.generations import LLMGenerator
from rl4llm.types import DecodingConfig, ExplorationConfig, Episode


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

    trainer = PPOTrainer.from_config(args.config_file)
    logger = trainer.logger
    config = trainer.config

    train_config = config['training_config']
    actor_config = config['actor']
    env_config = config['environment']
    eval_config = config['evaluator']

    train_decoding = DecodingConfig(**actor_config['decoding'])
    train_exploration = ExplorationConfig(**actor_config['exploration'])
    eval_decoding = DecodingConfig(**eval_config['decoding'])
    eval_exploration = None

    train_ds, test_ds = load_and_combine_datasets(config['datasets'])

    env_kwargs = {'stop_tokens': trainer.stop_tokens, 'seed': trainer.seed, **env_config}

    train_env_kwargs = {
        'datasets': train_ds,
        **env_kwargs,
    }
    eval_env_kwargs = {
        'datasets': test_ds,
        **env_kwargs,
    }
    env_train = VectorEnvWrapper(**train_env_kwargs)
    env_eval = VectorEnvWrapper(**eval_env_kwargs)

    generator = LLMGenerator(trainer.policy_model, trainer.tokenizer)

    actor_kwargs = {
        'tracker': trainer.tracker,
        'generator': generator,
    }
    train_actor_kwargs = {
        'for_evaluator': False,
        'decoding_config': train_decoding,
        'exploration_config': train_exploration,
        **actor_kwargs,
    }
    eval_actor_kwargs = {
        'for_evaluator': True,
        'decoding_config': eval_decoding,
        'exploration_config': eval_exploration,
        **actor_kwargs,
    }

    train_actor = EgreedyActor(**train_actor_kwargs)
    eval_actor = EgreedyActor(**eval_actor_kwargs)

    best_eval_accuracy = 0.0

    train_rollout_episodes = train_config.get('rollout_episodes', 1024)
    num_iters = train_config.get('rollout_iterations', 10000)
    eval_rollout_episodes = eval_config.get('rollout_episodes', 500)
    eval_enabled = eval_config.get('enabled', False)
    eval_interval = eval_config.get('interval', 100)

    def handle_exit():
        trainer.on_exit()

    try:
        # kick start the training
        while trainer.get_iteration_count() < num_iters:
            logger.info(f"Start iteration {trainer.get_iteration_count()}")

            # step 1: run actor in the MDP env using the current policy to generate training samples
            episodes, _ = train_actor.generate_samples(
                env_train,
                max_episodes=train_rollout_episodes,
            )

            # # **Step 2: Gather Episodes across ranks**
            # gathered_episodes_list = dist.all_gather_object(episodes)
            # aggregated_episodes = []
            # for rank_episodes in gathered_episodes_list:
            #     aggregated_episodes.extend(rank_episodes)

            # step 3. Train on all_episodes
            trainer.train(episodes)

            # step 4: evaluation and checkpoint
            iter_c = trainer.get_iteration_count()

            if eval_enabled and iter_c >= 1 and iter_c % eval_interval == 0:
                logger.info('Run evaluation')

                _, eval_stats = eval_actor.generate_samples(
                    env_eval, max_episodes=min(eval_rollout_episodes, len(test_ds)), correct_answer_rate=0.0
                )

                if 'accuracy' in eval_stats and eval_stats['accuracy'] > best_eval_accuracy:
                    best_eval_accuracy = eval_stats['accuracy']
                    logger.info(f"Best policy model with eval accuracy {best_eval_accuracy:.4f}")
                    trainer.save_checkpoint(is_best=True)

            # save data to external files
            trainer.tracker.flush()
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
