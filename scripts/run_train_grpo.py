"""Script to run RL GRPO fine-tuning on a single GPU."""

import argparse
import cProfile
import os
import pstats
import sys
from functools import partial
from traceback import format_exc
from typing import Any, Dict, List, Union

import deepspeed
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizer

from rl4llm.data import load_multiple_datasets
from rl4llm.envs import BaseRewardFunction, ExploreLLMEnv, LLMEnv
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.logging import LoggingManager
from rl4llm.trainers.grpo_trainer import (
    DistributedManager,
    GRPOConfig,
    GRPOTrainer,
)
from rl4llm.utils import load_yaml_config_file, set_seed
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
{question}

Answer:
Let's think step by step.
"""

PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Please first think about the reasoning process step by step, and put your final answer within \\boxed{{}}.

Question:
{question}<|im_end|>
<|im_start|>assistant
"""


def apply_prompt_template(item: Dict, template: str) -> Dict:
    """Apply the prompt template for sample, assume the template has a 'question' place holder"""
    question = item['question']

    prompt = template.format(question=question)

    return {'prompt': prompt}


class AccuracyRewardFunction(BaseRewardFunction):

    def __init__(self, name='accuracy_reward'):
        super().__init__(name)

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
    logger = LoggingManager(
        dist_manager, log_dir=artifacts_path, sample_file_format='jsonl'
    )

    torch_dtype = torch.bfloat16

    grpo_config = GRPOConfig(**config['grpo_config'])

    deepspeed_config['train_micro_batch_size_per_gpu'] = (
        grpo_config.train_micro_batch_size
    )
    deepspeed_config['train_batch_size'] = grpo_config.train_batch_size

    train_dataset, eval_dataset = load_multiple_datasets(datasets)

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle().select(range(max_train_samples))

    if max_test_samples is not None and max_test_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle().select(range(max_test_samples))

    policy_model, tokenizer = build_model_and_tokenizer(
        config['model'], torch_dtype
    )

    if any([k in model_name for k in ['0.5B', '1B', '1.5B']]):
        template = PROMPT_TEMPLATE_EASY
    else:
        template = PROMPT_TEMPLATE

    # Define the function with fixed template using partial
    apply_prompt = partial(apply_prompt_template, template=template)

    train_dataset = train_dataset.map(apply_prompt)
    eval_dataset = eval_dataset.map(apply_prompt)

    policy_engine, *_ = deepspeed.initialize(
        model=policy_model,
        model_parameters=policy_model.parameters(),
        config_params=deepspeed_config,
    )

    # for Explore LLM Env config
    replace_source_tokens = []
    # # Determine which tokens should be replaced based on format
    if grpo_config.xml_format:
        replace_source_tokens.append(tokenizer.encode('</think>')[0])
        replace_source_tokens.append(tokenizer.encode(' </think>')[0])
        replace_source_tokens.append(tokenizer.encode(':</think>')[0])
        replace_source_tokens.append(tokenizer.encode('.</think>')[0])
    else:
        replace_source_tokens.append(tokenizer.eos_token_id)

    replace_target_tokens = [tokenizer.encode(f' {kwd}')[0] for kwd in ['Wait']]
    replace_prevent_patterns = []
    explore_skip = 0

    if grpo_config.xml_format:
        replace_prevent_patterns.extend(
            [
                tokenizer.encode('</think>'),
                tokenizer.encode(' </think>'),
                tokenizer.encode('<answer>'),
            ]
        )
        explore_skip = len(tokenizer.encode('<think>'))

    if grpo_config.group_temperature:
        temperature = torch.linspace(
            grpo_config.min_temperature,
            grpo_config.max_temperature,
            steps=grpo_config.group_size,
            dtype=torch_dtype,
        )
        temperature = torch.round(temperature, decimals=2)

    explore_env_args = {
        'temperature': temperature,
        'explore_steps': grpo_config.explore_steps,
        'explore_top_k': grpo_config.explore_top_k,
        'explore_skip': explore_skip,
        'explore_decay_rate': grpo_config.explore_decay_rate,
        'replace_source_tokens': replace_source_tokens,
        'replace_target_tokens': replace_target_tokens,
        'replace_prevent_patterns': replace_prevent_patterns,
        'replace_max_per_seq': grpo_config.replace_max_per_seq,
        'replace_prob': grpo_config.replace_prob,
    }

    train_env = ExploreLLMEnv(
        dataset=train_dataset,
        batch_size=1,
        group_size=grpo_config.group_size,
        tokenizer=tokenizer,
        reward_functions=[AccuracyRewardFunction()],
        rank=dist_manager.local_rank,
        world_size=dist_manager.world_size,
        **explore_env_args,
    )

    # train_env = LLMEnv(
    #     dataset=train_dataset,
    #     batch_size=1,
    #     group_size=grpo_config.group_size,
    #     tokenizer=tokenizer,
    #     reward_functions=[AccuracyRewardFunction()],
    #     rank=dist_manager.local_rank,
    #     world_size=dist_manager.world_size,
    # )
    eval_env = LLMEnv(
        dataset=eval_dataset,
        batch_size=grpo_config.eval_batch_size,
        group_size=1,
        tokenizer=tokenizer,
        reward_functions=[AccuracyRewardFunction()],
        rank=dist_manager.local_rank,
        world_size=dist_manager.world_size,
    )

    trainer = GRPOTrainer(
        config=grpo_config,
        tokenizer=tokenizer,
        policy_engine=policy_engine,
        dist_manager=dist_manager,
        logger=logger,
        train_env=train_env,
        eval_env=eval_env,
        artifacts_path=artifacts_path,
        seed=seed,
    )

    # Create a profiler instance
    profiler = None

    # profiler = cProfile.Profile()
    # profiler.enable()

    def handle_exit():

        trainer.on_exit()

        if profiler is not None:
            try:
                profiler.disable()
                # Save profiling stats
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative').dump_stats('profile_stats.prof')
                stats.print_stats()  # Print to console for immediate feedback
            except Exception as _e:
                pass

    trainer.train(config)

    # try:
    #     trainer.train(config)
    # except KeyboardInterrupt:
    #     logger.info(
    #         '\nKeyboardInterrupt received in main loop. Shutting down...'
    #     )
    # except Exception as e:
    #     logger.error(f"An unexpected error occurred: {e}")
    #     logger.error(format_exc())
    # finally:
    #     handle_exit()
    #     logger.info('Exiting main program.')


if __name__ == '__main__':
    main()
