"""Script to run RL GRPO fine-tuning on a single GPU."""

import argparse
import os
from functools import partial
from typing import Any, Dict, List, Union

import deepspeed
import torch
import vllm
from transformers import PreTrainedTokenizer

from rl4llm.core.base_env import BaseRewardFunction
from rl4llm.data import load_multiple_datasets
from rl4llm.envs import (
    ExploreHFEnv,
    ExploreVLLMEnv,
    HFEnv,
    InferenceEnv,
    VLLMEnv,
)
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.inference.sgl_client import SGLangClient
from rl4llm.logging import LoggingManager
from rl4llm.trainers.grpo_trainer import (
    DistributedManager,
    GRPOConfig,
    GRPOTrainer,
)
from rl4llm.utils import load_yaml_config_file, set_seed
from rl4llm.utils.model_utils import build_model_and_tokenizer


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
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if len(ground_truths) == 1:
            ground_truths = [ground_truths] * len(completions)
        if len(completions) != len(ground_truths):
            raise ValueError(
                'Completion and ground truth have mismatch elements'
            )

        return [
            math_problem_grader(full_answer=answer, ground_truth=truth)
            for answer, truth in zip(completions, ground_truths)
        ]


def prepare_explore_processor_config(
    tokenizer: PreTrainedTokenizer, grpo_config: GRPOConfig
) -> Dict:
    """Creates the exploration logits processor needed config"""

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

    replace_target_tokens = [
        tokenizer.encode(f' {kwd}')[0]
        for kwd in ['Wait', 'But', 'Hmm', 'Actually', 'However']
    ]
    replace_prevent_patterns = []
    explore_skip_n = 0

    if grpo_config.xml_format:
        replace_prevent_patterns.extend(
            [
                tokenizer.encode('</think>'),
                tokenizer.encode(' </think>'),
                tokenizer.encode('<answer>'),
            ]
        )
        explore_skip_n = len(tokenizer.encode('<think>'))

    if grpo_config.group_temperature:
        temperature = torch.linspace(
            grpo_config.min_temperature,
            grpo_config.max_temperature,
            steps=grpo_config.group_size,
        )
        temperature = torch.round(temperature, decimals=2)

    return {
        'temperature': temperature,
        'explore_steps': grpo_config.explore_steps,
        'explore_top_k': grpo_config.explore_top_k,
        'explore_skip_n': explore_skip_n,
        'explore_decay_rate': grpo_config.explore_decay_rate,
        'replace_source_tokens': replace_source_tokens,
        'replace_target_tokens': replace_target_tokens,
        'replace_prevent_patterns': replace_prevent_patterns,
        'replace_max_per_seq': grpo_config.replace_max_per_seq,
        'replace_prob': grpo_config.replace_prob,
    }


def main():
    """Starts RL GRPO training loop."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This script only supports run on GPU.')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # IMPORTANT: need to disable V1 engine to access the true model
    # in order to use an easy way to update model weights with vllm 0.8.x
    os.environ['VLLM_USE_V1'] = '0'

    args = parse_args()
    job_config = load_yaml_config_file(args.config_file)

    seed = int(job_config.get('seed', 142))
    log_config = job_config.get('logging')
    artifacts_path = log_config.get('output_dir')
    datasets_config = job_config.get('dataset')
    max_train_samples = datasets_config.get('max_train_samples', None)
    max_test_samples = datasets_config.get('max_test_samples', None)
    model_config = job_config['model']
    model_name = model_config['pretrained_model']
    deepspeed_config = job_config['deepspeed']
    grpo_config = GRPOConfig(**job_config['grpo'])

    set_seed(seed)

    local_rank = int(os.environ.get('LOCAL_RANK'))
    device = torch.device(f"cuda:{local_rank}")
    torch_dtype = torch.bfloat16

    # # A hacky way to pass the min and max temperatures before apply the patch
    # if grpo_config.group_temperature:
    #     os.environ['VLLM_MIN_TEMPERATURE'] = str(grpo_config.min_temperature)
    #     os.environ['VLLM_MAX_TEMPERATURE'] = str(grpo_config.max_temperature)
    #     from rl4llm.patches import vllm_group_temperature_patch

    # # IMPORTANT: must initialize vLLM before deepspeed
    # vllm_engine = vllm.LLM(
    #     model=model_name,
    #     tensor_parallel_size=1,
    #     device=device,
    #     gpu_memory_utilization=0.7,  # for single GPU using 0.7 seems to work
    #     max_seq_len_to_capture=4096,
    #     enable_sleep_mode=True,  # important to enable sleep mode
    #     seed=seed,
    # )
    # vllm_engine.sleep()

    SGLANG_HOST = '127.0.0.1'
    SGLANG_PORT = 30000

    inference_client = SGLangClient(
        host=SGLANG_HOST,
        port=SGLANG_PORT,
    )

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed(verbose=False)

    dist_manager = DistributedManager()
    logger = LoggingManager(dist_manager, **log_config)

    deepspeed_config['train_micro_batch_size_per_gpu'] = (
        grpo_config.train_micro_batch_size
    )
    deepspeed_config['train_batch_size'] = grpo_config.train_batch_size

    train_dataset, eval_dataset = load_multiple_datasets(
        datasets_config['names']
    )

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle().select(range(max_train_samples))

    if max_test_samples is not None and max_test_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle().select(range(max_test_samples))

    policy_model, tokenizer = build_model_and_tokenizer(
        model_config, torch_dtype
    )
    policy_model = policy_model.to(dist_manager.device)

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

    # explore_env_args = prepare_explore_processor_config(tokenizer, grpo_config)
    # train_env = ExploreVLLMEnv(
    #     dataset=train_dataset,
    #     batch_size=1,  # always set batch size to 1 for training
    #     group_size=grpo_config.group_size,
    #     tokenizer=tokenizer,
    #     reward_functions=[AccuracyRewardFunction()],
    #     rank=dist_manager.local_rank,
    #     world_size=dist_manager.world_size,
    #     **explore_env_args,
    # )

    train_env = InferenceEnv(
        dataset=train_dataset,
        batch_size=1,  # always set batch size to 1 for training
        group_size=grpo_config.group_size,
        tokenizer=tokenizer,
        reward_functions=[AccuracyRewardFunction()],
        rank=dist_manager.local_rank,
        world_size=dist_manager.world_size,
    )
    eval_env = InferenceEnv(
        dataset=eval_dataset,
        batch_size=grpo_config.eval_batch_size,
        group_size=1,  # always set group size to 1 for evaluation
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
        artifacts_path=artifacts_path,
        train_env=train_env,
        eval_env=eval_env,
        inference_client=inference_client,
        seed=seed,
    )

    trainer.train(job_config)


if __name__ == '__main__':
    main()
