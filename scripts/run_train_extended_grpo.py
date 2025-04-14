"""Script to run RL extended GRPO fine-tuning."""

import argparse
import os
from functools import partial
from typing import Any, Dict, List, Union

import deepspeed
import torch
from transformers import PreTrainedTokenizer

from rl4llm.core.base_env import BaseRewardFunction
from rl4llm.core.distributed import DistributedOps
from rl4llm.data import load_multiple_datasets
from rl4llm.envs import (
    ExploreInferenceEnv,
    ExploreLocalLLMEnv,
    InferenceEnv,
    LocalLLMEnv,
)
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.inference.sgl_client import SGLangClient
from rl4llm.trainers.extended_grpo_trainer import (
    ExtendedGRPOConfig,
    ExtendedGRPOTrainer,
)
from rl4llm.utils import load_yaml_config_file, set_seed
from rl4llm.utils.model_utils import build_policy_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='RL extended GRPO fine-tuning')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/extended_grpo_config.yaml',
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
    # Include inference server specific configuration arguments
    parser.add_argument(
        '--use-infer-server',
        action='store_true',
        help='Connect to an inference server (default: False)',
    )
    parser.add_argument(
        '--infer-host',
        type=str,
        default='localhost',
        help='Inference server hostname or IP (default: localhost)',
    )
    parser.add_argument(
        '--infer-port',
        type=str,
        default='30000',
        help='Inference server port (default: 30000)',
    )
    parser.add_argument(
        '--infer-cohost-mode',
        action='store_true',
        help='Enable if inference server is sharing devices with training (default: False)',
    )

    args = parser.parse_args()

    if args.infer_cohost_mode and args.infer_host not in (
        '0.0.0.0',
        'localhost',
    ):
        raise ValueError(
            f"When using host '{args.infer_host}', you must explicitly set --infer-cohost-mode"
        )

    return args


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


def reward_transform_fn(reward_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform multiple rewards to single reward for a group of samples"""
    accuracy_rewards = reward_dict['accuracy_reward']  # [group_size]

    return accuracy_rewards


def prepare_explore_processor_config(
    tokenizer: PreTrainedTokenizer,
    grpo_config: ExtendedGRPOConfig,
    xml_format: bool = False,
) -> Dict:
    """Creates the exploration logits processor needed config"""

    special_tokens = [
        f' {kwd}' for kwd in ['Wait', 'But', 'Hmm', 'Actually', 'However']
    ]
    explore_skip_n = len(tokenizer.encode('<think>')) if xml_format else 0

    if grpo_config.group_temperature:
        temperatures = torch.linspace(
            grpo_config.min_temperature,
            grpo_config.max_temperature,
            steps=grpo_config.group_size,
        )
        temperatures = torch.round(temperatures, decimals=2)
    else:
        temperatures = (
            torch.ones((grpo_config.group_size,)) * grpo_config.temperature
        )

    return {
        'temperatures': temperatures,
        'explore_steps': grpo_config.explore_steps,
        'explore_top_k': grpo_config.explore_top_k,
        'explore_skip_n': explore_skip_n,
        'explore_decay_rate': grpo_config.explore_decay_rate,
        'continue_special_tokens': special_tokens,
        'continue_max_retry': grpo_config.continue_max_retry,
        'continue_prob': grpo_config.continue_prob,
    }


def main():
    """Starts RL GRPO training loop."""
    if not torch.cuda.is_available():
        raise RuntimeError('This script requires supports CUDA.')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()
    job_config = load_yaml_config_file(args.config_file)

    seed = int(job_config.get('seed', 142))
    log_config = job_config.get('logging')
    datasets_config = job_config.get('dataset')
    max_train_samples = datasets_config.get('max_train_samples', None)
    max_test_samples = datasets_config.get('max_test_samples', None)
    model_config = job_config['model']
    model_name = model_config['pretrained_model']
    deepspeed_config = job_config['deepspeed']
    grpo_config = ExtendedGRPOConfig(**job_config['grpo'])

    set_seed(seed)

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed(verbose=False)
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    bf16_enabled = deepspeed_config.get('bf16', {}).get('enabled')
    torch_dtype = torch.bfloat16 if bf16_enabled else torch.float16

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

    policy_model, tokenizer = build_policy_model_and_tokenizer(
        model_config, torch_dtype
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

    # Create reference model and optionally use deepspeed sharding
    ref_model = None
    if grpo_config.kl_loss_coef > 0:
        ref_model, _ = build_policy_model_and_tokenizer(
            model_config, torch_dtype
        )
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

    reference_deepspeed_config = job_config.get('reference_deepspeed')
    if ref_model is not None and reference_deepspeed_config is not None:
        zero3_enabled = (
            reference_deepspeed_config.get('zero_optimization', {}).get('stage')
            == 3
        )
        if zero3_enabled:
            ref_model, *_ = deepspeed.initialize(
                model=ref_model,
                model_parameters=[],
                config_params=reference_deepspeed_config,
            )
            ref_model.eval()

    explore_env_args = prepare_explore_processor_config(tokenizer, grpo_config)
    inference_client = None

    env_reward_functions = [AccuracyRewardFunction()]
    eval_env_cls = LocalLLMEnv
    train_env_cls = ExploreLocalLLMEnv
    if args.use_infer_server:
        inference_client = SGLangClient(
            host=args.infer_host,
            port=args.infer_port,
            cohost_mode=args.infer_cohost_mode,
        )
        eval_env_cls = InferenceEnv
        train_env_cls = ExploreInferenceEnv

    train_env = train_env_cls(
        dataset=train_dataset,
        batch_size=1,  # always set batch size to 1 for training
        group_size=grpo_config.group_size,
        tokenizer=tokenizer,
        reward_functions=env_reward_functions,
        rank=local_rank,
        world_size=world_size,
        **explore_env_args,
    )

    eval_env = eval_env_cls(
        dataset=eval_dataset,
        batch_size=grpo_config.eval_batch_size,
        group_size=1,  # always set group size to 1 for evaluation
        tokenizer=tokenizer,
        reward_functions=env_reward_functions,
        rank=local_rank,
        world_size=world_size,
    )

    trainer = ExtendedGRPOTrainer(
        config=grpo_config,
        tokenizer=tokenizer,
        policy_engine=policy_engine,
        log_config=log_config,
        train_env=train_env,
        eval_env=eval_env,
        inference_client=inference_client,
        ref_model=ref_model,
        reward_transform_fn=reward_transform_fn,
        seed=seed,
    )

    trainer.train(job_config)


if __name__ == '__main__':
    main()
