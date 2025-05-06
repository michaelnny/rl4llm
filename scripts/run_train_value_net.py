"""Script to run RL value model training."""

import argparse
import os
from functools import partial
from typing import Any, Dict, List, Union

import deepspeed
import torch

from rl4llm.core.base_env import BaseRewardFunction, ChatMessage
from rl4llm.data import load_multiple_datasets
from rl4llm.envs import HfMDPEnv, SglMDPEnv
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.inference.sgl_client import SGLangClient
from rl4llm.trainers.value_net_trainer import ValueNetConfig, ValueNetTrainer
from rl4llm.utils import load_yaml_config_file, set_seed
from rl4llm.utils.model_utils import build_value_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='RL RL value model training')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/value_net_config.yaml',
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


def prepare_initial_chat_messages(item: Dict) -> Dict:
    """Build chat-style messages for initial state"""
    messages = [
        {'role': 'user', 'content': item['question'].strip()},
        {'role': 'assistant', 'content': "Let's think step by step"},
    ]
    return {'messages': messages}


class AccuracyRewardFunction(BaseRewardFunction):
    def __init__(self, name='accuracy_reward'):
        super().__init__(name)

    def __call__(
        self,
        messages: List[ChatMessage],
        ground_truth: Union[str | float | int],
        **kwargs: Dict[str, Any],
    ) -> List[float]:
        """Implements the reward function.

        Args:
            messages (List[ChatMessage]]: Full chat history for the sample.
            ground_truth (Union[str | float | int]): Ground truth for the problem.
            **kwargs (Dict[str, Any]): Any additional data.

        Returns:
            List[float]: A list of scalar rewards.
        """

        # get last completion
        completion = messages[-1].content

        return math_problem_grader(
            full_answer=completion,
            ground_truth=ground_truth,
            min_score=-1.0,
            max_score=1.0,
        )


def reward_transform_fn(reward_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform multiple rewards to single reward for a group of samples"""
    accuracy_rewards = reward_dict['accuracy_reward']  # [group_size]

    return accuracy_rewards


def main():
    """Starts value model training loop."""
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
    model_config = job_config['value_model']
    model_name = model_config['pretrained_model']
    deepspeed_config = job_config['deepspeed']
    value_config = ValueNetConfig(**job_config['value_net_config'])

    set_seed(seed)

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed(verbose=False)
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    bf16_enabled = deepspeed_config.get('bf16', {}).get('enabled')
    torch_dtype = torch.bfloat16 if bf16_enabled else torch.float16
    deepspeed_config['train_micro_batch_size_per_gpu'] = (
        value_config.train_micro_batch_size
    )
    deepspeed_config['train_batch_size'] = value_config.train_batch_size

    train_dataset, eval_dataset = load_multiple_datasets(
        datasets_config['names']
    )

    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle().select(range(max_train_samples))

    # if max_test_samples is not None and max_test_samples < len(eval_dataset):
    #     eval_dataset = eval_dataset.shuffle().select(range(max_test_samples))

    train_dataset = train_dataset.map(prepare_initial_chat_messages)
    # eval_dataset = eval_dataset.map(prepare_initial_chat_messages)

    value_model, tokenizer = build_value_model_and_tokenizer(
        model_config, torch_dtype
    )

    value_engine, *_ = deepspeed.initialize(
        model=value_model,
        model_parameters=value_model.parameters(),
        config_params=deepspeed_config,
    )

    env_reward_functions = [AccuracyRewardFunction()]
    inference_client = None
    env_cls = HfMDPEnv
    if args.use_infer_server:
        inference_client = SGLangClient(
            host=args.infer_host,
            port=args.infer_port,
            cohost_mode=args.infer_cohost_mode,
        )
        env_cls = SglMDPEnv

    train_env = env_cls(
        dataset=train_dataset,
        batch_size=1,  # always set batch size to 1 for training
        group_size=value_config.group_size,
        tokenizer=tokenizer,
        reward_functions=env_reward_functions,
        rank=local_rank,
        world_size=world_size,
    )

    trainer = ValueNetTrainer(
        config=value_config,
        tokenizer=tokenizer,
        value_engine=value_engine,
        log_config=log_config,
        train_env=train_env,
        inference_client=inference_client,
        reward_transform_fn=reward_transform_fn,
        seed=seed,
    )

    trainer.train(job_config)


if __name__ == '__main__':
    main()
