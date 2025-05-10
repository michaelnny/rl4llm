"""Script to run RL GRPO fine-tuning with tools."""

import argparse
import os
import re
from typing import Any, Dict, List, Union

import deepspeed
import torch

from rl4llm.core.base_env import BaseRewardFunction, ChatMessage
from rl4llm.data import load_multiple_datasets
from rl4llm.envs.sgl_tool_env import ENV_TOOL_SCHEMAS, SglToolMDPEnv
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
        default='./configs/grpo_tools_config.yaml',
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
    ]
    return {'messages': messages}


class AccuracyRewardFunction(BaseRewardFunction):
    """Implements the accuracy reward for math problems"""

    def __init__(self, name='accuracy_reward'):
        super().__init__(name)

    def __call__(
        self,
        messages: List[ChatMessage],
        ground_truth: Union[str | float | int],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Implements the reward function.

        Args:
            messages (List[ChatMessage]]: Full chat history for the sample.
            ground_truth (Union[str | float | int]): Ground truth for the problem.
            **kwargs (Dict[str, Any]): Any additional data.

        Returns:
            float: A scalar rewards.
        """

        # get last completion
        completion = messages[-1].content

        return math_problem_grader(
            full_answer=completion,
            ground_truth=ground_truth,
            min_score=-1.0,
            max_score=1.0,
        )


class ToolUsageRewardFunction(BaseRewardFunction):
    """
    A simplified reward function for tool usage, designed for an environment
    where 'code_execution_tool' is the primary or only tool.

    - Penalizes clear execution failures (from the tool dispatcher or Python execution).
    - Applies a small cost for every tool call.
    - Gives a small bonus if a tool call (assumed to be code_execution_tool)
      executes without any reported errors.
    """

    def __init__(
        self,
        name: str = 'tool_usage_reward',
        cost_per_tool_call: float = 0.0,
        tool_call_error_penalty: float = -0.1,
        tool_call_success_bonus: float = 0.25,
    ):
        # Ensure error penalty is negative, cost is small (can be 0 or slightly neg/pos), success is positive
        if not (tool_call_error_penalty < 0):
            raise ValueError('tool_call_error_penalty should be negative.')
        if not (tool_call_success_bonus > 0):
            raise ValueError('tool_call_success_bonus should be positive.')

        super().__init__(name)
        self.cost_per_tool_call = cost_per_tool_call
        self.tool_call_error_penalty = tool_call_error_penalty
        self.tool_call_success_bonus = tool_call_success_bonus

        # Simplified regex: looks for the standard "Error: " prefix.
        # Adding '^' ensures it checks the beginning of the string.
        self._error_prefix = 'Error: '
        self._error_indicators_regex = re.compile(
            r'^{}'.format(re.escape(self._error_prefix))
        )

    def _is_execution_error(self, tool_output_content: str) -> bool:
        """Checks if the tool output string indicates an execution error."""
        return bool(self._error_indicators_regex.search(tool_output_content))

    def __call__(
        self,
        messages: List[ChatMessage],
        ground_truth: Union[str, float, int],
        **kwargs: Any,
    ) -> float:
        episode_tool_reward = 0.0

        for msg in messages:
            if msg.role == 'tool':
                # 1. Apply the cost for any tool message encountered
                episode_tool_reward += self.cost_per_tool_call

                # 2. Check for any execution error
                if self._is_execution_error(msg.content):
                    episode_tool_reward += self.tool_call_error_penalty
                else:
                    episode_tool_reward += self.tool_call_success_bonus

        return episode_tool_reward


def reward_transform_fn(reward_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform multiple rewards to single reward for a group of samples"""
    # Start with accuracy reward, assuming it's the primary component
    if 'accuracy_reward' not in reward_dict:
        raise ValueError('accuracy_reward is missing from reward_dict.')

    final_reward = reward_dict['accuracy_reward'].clone()

    # Add tool usage reward if present
    if 'tool_usage_reward' in reward_dict:
        tool_rewards = reward_dict['tool_usage_reward']
        if isinstance(tool_rewards, torch.Tensor):
            tool_rewards = tool_rewards.to(final_reward.device)
        else:  # Ensure tool_rewards is a tensor if it's not
            tool_rewards = torch.tensor(
                tool_rewards,
                dtype=final_reward.dtype,
                device=final_reward.device,
            )

        final_reward += tool_rewards
        # For debugging:
        # logger.debug(f"Accuracy: {reward_dict['accuracy_reward']}, Tool: {tool_rewards}, Final: {final_reward}")

    return final_reward


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

    env_tool_config = job_config['env_tool_config']
    env_max_steps = env_tool_config.get('env_max_steps', 5)
    cost_per_tool_call = env_tool_config.get('cost_per_tool_call', -0.05)
    tool_call_error_penalty = env_tool_config.get(
        'tool_call_error_penalty', -0.2
    )
    tool_call_success_bonus = env_tool_config.get(
        'tool_call_success_bonus', 0.1
    )

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
        'reward_functions': [
            AccuracyRewardFunction(),
            ToolUsageRewardFunction(
                cost_per_tool_call=cost_per_tool_call,
                tool_call_error_penalty=tool_call_error_penalty,
                tool_call_success_bonus=tool_call_success_bonus,
            ),
        ],
        'reward_transform_fn': reward_transform_fn,
        'tokenizer': tokenizer,
        'rank': local_rank,
        'world_size': world_size,
        'max_steps': env_max_steps,
    }

    inference_client = SGLangClient(
        host=args.infer_host,
        port=args.infer_port,
        cohost_mode=args.infer_cohost_mode,
    )

    train_env = SglToolMDPEnv(
        dataset=train_dataset,
        batch_size=1,  # always set batch size to 1 for training
        group_size=grpo_config.group_size,
        **env_args,
    )
    eval_env = SglToolMDPEnv(
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
