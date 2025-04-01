"""Script to run RL GRPO fine-tuning on a single GPU."""

import argparse
import cProfile
import os
import pstats
import sys
from traceback import format_exc
from typing import Any, Dict, List, Union

import deepspeed
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizer

from rl4llm.data import load_multiple_datasets
from rl4llm.envs import BaseRewardFunction, LLMEnv
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.logging import LoggingManager
from rl4llm.trainers.grpo_trainer import (
    DistributedManager,
    GRPOConfig,
    GRPOTrainer,
)
from rl4llm.utils import load_yaml_config_file, set_seed
from rl4llm.utils.dataset_utils import shard_dataset
from rl4llm.utils.model_utils import (
    build_model_and_tokenizer,
    create_optimizer_and_scheduler,
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
    # Include DeepSpeed configuration arguments
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Required by deepspeed for local rank passed from distributed launcher, not used by our script',
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


def preprocess_dataset(dataset: List[Dict], model_name: str) -> List[Dict]:
    """Pre-tokenize the entire dataset and return a list of tokenized inputs."""

    if any([k in model_name for k in ['0.5B', '1B', '1.5B']]):
        template = PROMPT_TEMPLATE_EASY
    else:
        template = PROMPT_TEMPLATE

    processed_data = []
    for item in dataset:
        question = item['question']
        prompt = template.format(query=question)

        processed_data.append({'prompt': prompt, **item})

    return processed_data


class AccuracyRewardFunction(BaseRewardFunction):
    name: str = 'reward_function'

    def __call__(
        self,
        completions: List[str],
        ground_truths: List[Union[str | float | int]],
        **kwargs: Dict[str, Any],
    ) -> List[float]:
        """Implements the reward function.

        Args:
            completions (List[str]): LLM generated completion texts.
            ground_truths (List[Union[str | float | int]]): Ground truth for the problem.
            **kwargs (Dict[str, Any]): Any additional data.

        Returns:
            List[float]: A list of scalar rewards.
        """
        return [
            math_problem_grader(full_answer=answer, ground_truth=truth)
            for answer, truth in zip(completions, ground_truths)
        ]


# def preprocess_dataset(
#     dataset: List[Dict], tokenizer: PreTrainedTokenizer, model_name: str
# ) -> List[Dict]:
#     """Pre-tokenize the entire dataset and return a list of tokenized inputs."""

#     if any([k in model_name for k in ["0.5B", "1B", "1.5B"]]):
#         template = PROMPT_TEMPLATE_EASY
#     else:
#         template = PROMPT_TEMPLATE

#     tokenized_data = []
#     for item in dataset:
#         question = item["question"]
#         ground_truth = item["ground_truth"]
#         task_type = item["task_type"]
#         prompt = template.format(query=question)
#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=False,
#             padding=False,
#             max_length=tokenizer.model_max_length,
#         )

#         tokenized_data.append(
#             {
#                 "input_ids": inputs["input_ids"].squeeze(0),  # Shape: [seq_len]
#                 "attention_mask": inputs["attention_mask"].squeeze(0),
#                 "question": question,
#                 "ground_truth": ground_truth,
#                 "task_type": task_type,
#             }
#         )

#     return tokenized_data


def main():
    """Starts RL GRPO training loop."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            'This script only supports run on GPU with BF16 mode.'
        )

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()

    config = load_yaml_config_file(args.config_file)

    seed = int(config.get('job').get('seed', 142))
    artifacts_path = config.get('job').get('artifacts_path')
    datasets = config.get('job').get('datasets')
    max_train_samples = config.get('job').get('max_train_samples', None)
    max_test_samples = config.get('job').get('max_test_samples', None)
    model_name = config['model']['pretrained_model']
    deepspeed_config = config['deepspeed_config']

    set_seed(seed)

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed(verbose=False)

    dist_manager = DistributedManager()
    logger_manager = LoggingManager(
        config, dist_manager, log_dir=artifacts_path, sample_file_format='jsonl'
    )

    torch_dtype = torch.bfloat16

    grpo_config = GRPOConfig(
        **config['grpo_config'],
        mini_batch_size=deepspeed_config['train_micro_batch_size_per_gpu'],
    )

    train_dataset, eval_dataset = load_multiple_datasets(datasets)

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle().select(range(max_train_samples))

    if max_test_samples is not None and max_test_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle().select(range(max_test_samples))

    shared_train_dataset = shard_dataset(
        train_dataset,
        dist_manager.world_size,
        dist_manager.global_rank,
    )
    shared_eval_dataset = shard_dataset(
        eval_dataset,
        dist_manager.world_size,
        dist_manager.global_rank,
    )

    policy_model, tokenizer = build_model_and_tokenizer(
        config['model'], torch_dtype
    )

    shared_train_dataset = preprocess_dataset(shared_train_dataset, model_name)
    shared_eval_dataset = preprocess_dataset(shared_eval_dataset, model_name)

    policy_engine, *_ = deepspeed.initialize(
        model=policy_model,
        model_parameters=policy_model.parameters(),
        config_params=deepspeed_config,
    )

    train_env = LLMEnv(
        dataset=shared_train_dataset,
        batch_size=1,
        tokenizer=tokenizer,
        reward_functions=[AccuracyRewardFunction()],
    )

    trainer = GRPOTrainer(
        config=grpo_config,
        tokenizer=tokenizer,
        policy_engine=policy_engine,
        dist_manager=dist_manager,
        logger=logger_manager,
        train_env=train_env,
        eval_env=None,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        artifacts_path=artifacts_path,
        seed=seed,
    )

    trainer.train()

    trainer.on_exit()


if __name__ == '__main__':
    main()
