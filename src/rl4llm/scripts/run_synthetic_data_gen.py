"""Generate synthetic data for training a classifier to detect incoherent or nonsensical responses from LLM"""

import argparse
import sys
from traceback import format_exc

import torch

from rl4llm.data import load_and_combine_datasets
from rl4llm.generations import SyntheticDataGenerator
from rl4llm.utils import build_model_and_tokenizer, load_yaml_config_file, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic data generation')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/synthetic_data_config.yaml',
        # required=True,
        help='Path to the yaml file contains all the essential configuration',
    )
    return parser.parse_args()


def main():
    """Starts loop."""
    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    seed = int(config.get('job').get('seed', 142))
    artifacts_path = config.get('job').get('artifacts_path')
    datasets = config.get('job').get('datasets')
    max_train_samples = config.get('job').get('max_train_samples', None)
    max_test_samples = config.get('job').get('max_test_samples', None)
    n_samples = config.get('job').get('n_samples', 10)
    min_new_tokens = config.get('job').get('min_new_tokens', 50)
    max_new_tokens = config.get('job').get('max_new_tokens', 1024)
    system_prompt = config.get('job').get('system_prompt', None)
    dual_language_system_prompts = config.get('job').get('dual_language_system_prompts', [])
    set_seed(seed)

    logger = setup_logger()

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

    torch_dtype = torch.float16
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
    else:
        device = torch.device('cpu')

    model, tokenizer = build_model_and_tokenizer(config['model'], torch_dtype)

    generator = SyntheticDataGenerator(model=model, tokenizer=tokenizer, device=device, output_dir=artifacts_path)

    def handle_exit():
        # save datasets
        generator.close()

    try:
        logger.info('Generating training data...')
        generator.generate_dataset(
            train_ds.to_list(),
            system_prompt=system_prompt,
            n_samples=n_samples,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            is_train=True,
        )

        logger.info('Generating test data...')
        generator.generate_dataset(
            test_ds.to_list(),
            system_prompt=system_prompt,
            n_samples=n_samples,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            is_train=False,
        )

        # additionally, add samples with multiple languages
        if dual_language_system_prompts:
            for prompt in dual_language_system_prompts:
                logger.info(f'Generating training data with dual language system prompt: {prompt}...')
                dual_system_prompt = system_prompt + '\n\n' + prompt
                max_train_size = int(len(train_ds) * 0.2)
                generator.generate_dataset(
                    train_ds.shuffle().select(range(max_train_size)).to_list(),
                    system_prompt=dual_system_prompt,
                    n_samples=(n_samples // 2),
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    positive_only=True,
                    is_train=True,
                )
                max_test_size = int(len(test_ds) * 0.1)
                generator.generate_dataset(
                    test_ds.shuffle().select(range(max_test_size)).to_list(),
                    system_prompt=dual_system_prompt,
                    n_samples=(n_samples // 2),
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    positive_only=True,
                    is_train=False,
                )

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
