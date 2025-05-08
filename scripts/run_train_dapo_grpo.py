"""Script to run RL DAPO fine-tuning using GRPO as base trainer."""

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
from rl4llm.trainers.dapo_grpo_trainer import DAPOConfig, DAPOTrainer
from rl4llm.utils import load_yaml_config_file, set_seed
from rl4llm.utils.model_utils import build_policy_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='RL DAPO GRPO fine-tuning')
    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/dapo_config.yaml',
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


class LengthPenaltyRewardFunction(BaseRewardFunction):
    """Implements the soft overlong penalty reward as in DAPO"""

    def __init__(
        self,
        tokenizer: Any,
        L_max: int,
        L_cache: int,
        name: str = 'length_penalty_reward',
    ):
        if L_max <= 0 or L_cache <= 0 or L_cache < L_max:
            raise ValueError('L_max and L_cache must be positive integers')
        super().__init__(name)
        self.tokenizer = tokenizer
        self.L_max = L_max
        self.L_cache = L_cache
        if self.L_cache <= 0:
            # To prevent division by zero if L_cache is not positive.
            # If L_cache is 0, the penalty becomes a hard step: 0 if <= L_max, -1 if > L_max.
            # The paper implies L_cache > 0 for the linear ramp.
            print(
                'Warning: L_cache is not positive. The linear penalty ramp will not apply as intended.'
            )

    def __call__(
        self,
        messages: List[ChatMessage],
        ground_truth: Union[str | float | int],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Implements the length penalty reward function.

        Args:
            messages (List[ChatMessage]]: Full chat history for the sample.
            ground_truth (Union[str | float | int]): Ground truth for the problem.
            **kwargs (Dict[str, Any]): Any additional data.

        Returns:
            float: A scalar rewards.
        """
        completion_text = messages[-1].content

        # Get token length. Ensure this matches how your LLM counts tokens.
        # For many Hugging Face tokenizers, `encode` returns token IDs.
        token_ids = self.tokenizer.encode(
            completion_text, add_special_tokens=False
        )
        response_length = len(token_ids)

        penalty = 0.0
        if response_length > self.L_max:
            penalty = -1.0
        elif self.L_cache > 0 and response_length > (self.L_max - self.L_cache):
            # This is the linear ramp part: L_max - L_cache < |y| <= L_max
            # R_length(y) = (L_max - L_cache - |y|) / L_cache
            # This formula results in 0 when |y| = L_max - L_cache
            # and -1 when |y| = L_max
            penalty = (self.L_max - self.L_cache - response_length) / float(
                self.L_cache
            )

        return penalty


def reward_transform_fn(reward_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Transform multiple rewards to single reward for a group of samples.
    The paper states: "This penalty is added to the original rule-based correctness reward"
    """
    accuracy_rewards = reward_dict['accuracy_reward']  # [group_size]

    if 'length_penalty_reward' in reward_dict:
        length_penalties = reward_dict['length_penalty_reward']  # [group_size]
        # Ensure both are on the same device if they are tensors
        if isinstance(accuracy_rewards, torch.Tensor) and isinstance(
            length_penalties, torch.Tensor
        ):
            length_penalties = length_penalties.to(accuracy_rewards.device)

        final_reward = accuracy_rewards + length_penalties
        # print(f"Accuracy: {accuracy_rewards.item()}, Length Penalty: {length_penalties.item()}, Final: {final_reward.item()}") # For debugging
        return final_reward
    else:
        # print(f"Accuracy: {accuracy_rewards.item()}, No Length Penalty, Final: {accuracy_rewards.item()}") # For debugging
        return accuracy_rewards


def main():
    """Starts RL DAPO GRPO training loop."""
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
    dapo_config = DAPOConfig(**job_config['dapo_grpo'])

    set_seed(seed)

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed(verbose=False)
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    bf16_enabled = deepspeed_config.get('bf16', {}).get('enabled')
    torch_dtype = torch.bfloat16 if bf16_enabled else torch.float16
    deepspeed_config['train_micro_batch_size_per_gpu'] = (
        dapo_config.train_micro_batch_size
    )
    deepspeed_config['train_batch_size'] = dapo_config.train_batch_size

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
    if dapo_config.kl_loss_coef > 0:
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
            LengthPenaltyRewardFunction(
                tokenizer=tokenizer,
                L_max=dapo_config.length_max,
                L_cache=dapo_config.length_cache,
            ),
        ],
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
        group_size=dapo_config.group_size,
        **env_args,
    )
    eval_env = env_cls(
        dataset=eval_dataset,
        batch_size=dapo_config.eval_batch_size,
        group_size=1,  # always set group size to 1 for evaluation
        **env_args,
    )

    trainer = DAPOTrainer(
        config=dapo_config,
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
