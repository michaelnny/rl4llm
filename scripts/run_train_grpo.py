"""Script to run RL GRPO fine-tuning."""

import argparse
import os
from typing import Any, Dict, List, Union

import deepspeed
import torch

from rl4llm.core.base_env import BaseRewardFunction, ChatMessage
from rl4llm.data import load_multiple_datasets
from rl4llm.envs import HfMDPEnv, SglMDPEnv
from rl4llm.graders.math_grader import math_problem_grader
from rl4llm.inference.sgl_client import SGLangClient
from rl4llm.trainers.grpo_trainer import GRPOConfig, GRPOTrainer
from rl4llm.utils import load_yaml_config_file, set_seed
from rl4llm.utils.model_utils import build_policy_model_and_tokenizer


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


def apply_custom_chat_template(tokenizer):
    """This could be useful for training on base-model or for special use cases (like with special pre-filling for generation)"""
    # Define a Jinja2 chat template string

    jinja_chat_template = (
        '{% for message in messages %}'
        "{% if message.role == 'user' %}"
        'Question: {{ message.content }}\n\n'
        "Answer: Let's think step by step.\n"
        "{% elif message.role == 'assistant' %}"
        '{{ message.content }}'
        "{% elif message.role == 'system' %}"
        '{{ message.content }}\n\n'
        '{% endif %}'
        '{% endfor %}'
    )

    tokenizer.chat_template = jinja_chat_template


def prepare_initial_chat_messages(item: Dict) -> Dict:
    """Build chat-style messages for initial state"""
    messages = [
        {'role': 'user', 'content': item['question'].strip()},
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
    grpo_config = GRPOConfig(**job_config['grpo'])

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

    # Load and pre-processing dataset
    train_dataset, eval_dataset = load_multiple_datasets(
        datasets_config['names']
    )
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.shuffle().select(range(max_train_samples))
    if max_test_samples is not None and max_test_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle().select(range(max_test_samples))

    train_dataset = train_dataset.map(prepare_initial_chat_messages)
    eval_dataset = eval_dataset.map(prepare_initial_chat_messages)

    # Create models
    policy_model, tokenizer = build_policy_model_and_tokenizer(
        model_config, torch_dtype
    )

    # Use our own template for base model training
    apply_custom_chat_template(tokenizer)

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

    # Create envs
    env_args = {
        'reward_functions': [AccuracyRewardFunction()],
        'reward_transform_fn': reward_transform_fn,
        'tokenizer': tokenizer,
        'rank': local_rank,
        'world_size': world_size,
    }

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
        group_size=grpo_config.group_size,
        **env_args,
    )
    eval_env = env_cls(
        dataset=eval_dataset,
        batch_size=grpo_config.eval_batch_size,
        group_size=1,  # always set group size to 1 for evaluation
        **env_args,
    )

    trainer = GRPOTrainer(
        config=grpo_config,
        tokenizer=tokenizer,
        policy_engine=policy_engine,
        log_config=log_config,
        train_env=train_env,
        eval_env=eval_env,
        inference_client=inference_client,
        ref_model=ref_model,
        seed=seed,
    )

    trainer.train(job_config)


if __name__ == '__main__':
    main()
