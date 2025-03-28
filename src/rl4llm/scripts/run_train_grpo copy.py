"""Script to run RL GRPO fine-tuning on a single GPU."""

import argparse
import cProfile
import os
import pstats
import sys
from traceback import format_exc
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer

from rl4llm.core.grpo import GRPOConfig, GRPOTrainer
from rl4llm.data import load_multiple_datasets
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
        default='./configs/grpo_config.yaml',
        # required=True,
        help='Path to the yaml file contains all the essential configuration',
    )
    return parser.parse_args()


PROMPT_TEMPLATE_EASY = """Question:
{query}

Answer:
Let's think step by step.
"""

PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Please first think about the reasoning process step by step, and put your final answer within \\boxed{{}}.

Question:
{query}<|im_end|>
<|im_start|>assistant
"""


def preprocess_dataset(dataset: List[Dict], tokenizer: PreTrainedTokenizer, model_name: str) -> List[Dict]:
    """Pre-tokenize the entire dataset and return a list of tokenized inputs."""

    if any([k in model_name for k in ['0.5B', '1B', '1.5B']]):
        template = PROMPT_TEMPLATE_EASY
    else:
        template = PROMPT_TEMPLATE

    tokenized_data = []
    for item in dataset:
        question = item['question']
        ground_truth = item['ground_truth']
        task_type = item['task_type']
        prompt = template.format(query=question)
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=False,
            padding=False,
            max_length=tokenizer.model_max_length,
        )

        tokenized_data.append(
            {
                'input_ids': inputs['input_ids'].squeeze(0),  # Shape: [seq_len]
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'question': question,
                'ground_truth': ground_truth,
                'task_type': task_type,
            }
        )

    return tokenized_data


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
    model_name = config['model']['pretrained_model']
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

    logger.info('Preprocessing datasets...')
    train_ds = preprocess_dataset(train_ds, tokenizer, model_name)
    test_ds = preprocess_dataset(test_ds, tokenizer, model_name)

    trainer = GRPOTrainer(
        config=grpo_config,
        policy_model=policy_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ds=train_ds,
        test_ds=test_ds,
        device=device,
        torch_dtype=torch_dtype,
        artifacts_path=artifacts_path,
        coherent_model_config=config.get('coherent_model'),
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
