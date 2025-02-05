from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator

# for MDP environment


class ChatTurn(BaseModel):
    """A chat turn"""

    role: str = Field(..., min_length=3, description='The role of the chat turn')
    content: str = Field(..., min_length=5, description='The chat content')

    @field_validator('role')
    def validate_role(cls, value):
        if value not in ['system', 'assistant', 'user']:
            raise ValueError(f"Invalid role: {value}")
        return value


EnvState = List[ChatTurn]


class TokenUsage(BaseModel):
    """Collect token usage for logging"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StateNode(BaseModel):
    """A node representing a state in the environment."""

    state_id: str = Field(..., min_length=3, description='State id')
    user_prompt: str = Field(..., min_length=10, description='User turn content')
    system_prompt: Optional[str] = Field(None, description='System-level instructions')
    should_grade: bool = Field(False, description='Should grade answer')
    should_augment: bool = Field(False, description='Should augment last answer')
    is_start: bool = Field(False, description='Is start state')
    transitions: List[Tuple['StateNode', float]] = Field([], description='State transition')

    def add_transition(self, next_state: 'StateNode', probability: float) -> None:
        """Add a transition and normalize probabilities"""
        self.transitions.append((next_state, probability))
        total = sum(prob for _, prob in self.transitions)
        if total > 0:
            self.transitions = [(state, prob / total) for state, prob in self.transitions]


class EnvAction(BaseModel):
    """An action in the MDP environment"""

    text: str = Field(..., min_length=2, description='The text of the action')
    reasoning: Optional[str] = Field(None, description='The reasoning tokens')
    exploring_steps: Optional[int] = Field(0, description='The number of starting steps for exploration')
    temperature: Optional[float] = Field(0.0, description='The temperature for decoding')
    usage: Optional[TokenUsage] = Field(None, description='The token usage during generation')


class Transition(BaseModel):
    """A transition step in the MDP environment"""

    state: StateNode = Field(..., description='The state node in MDP')
    action: EnvAction = Field(..., description='The policy model generated action')
    reward: float = Field(0.0, description='Reward for the action')
    is_done: Optional[bool] = Field(False, description='Terminal state mark')


class Episode(BaseModel):
    """An episode in the MDP environment"""

    question: str = Field(..., min_length=2, description='Question or task full text')
    ground_truth: Union[str, int, float] = Field(..., description='Ground truth answer')
    task_type: str = Field('GSM', description='Task or dataset type')
    short_answer: Optional[Union[str, int, float]] = Field(
        None, description='Extracted short answer from model generated action'
    )
    graded_reward: Optional[float] = Field(0, description='Graded reward for the episode')
    transitions: List[Transition] = Field([], description='Episode transitions')

    def count_total_rewards(self) -> int:
        """Count total rewards in the episode"""
        if not self.transitions:
            return 0
        return sum([t.reward for t in self.transitions])

    def count_total_tokens(self) -> int:
        """Count total tokens in the episode"""
        if not self.transitions:
            return 0
        return sum([t.action.usage.total_tokens for t in self.transitions])

    def count_prompt_tokens(self) -> int:
        """Count total prompt tokens in the episode"""
        if not self.transitions:
            return 0
        return sum([t.action.usage.prompt_tokens for t in self.transitions])

    def count_completion_tokens(self) -> int:
        """Count total completion tokens in the episode"""
        if not self.transitions:
            return 0
        return sum([t.action.usage.completion_tokens for t in self.transitions])

    def record_transition(self, state: StateNode, action: EnvAction, reward: float, done: bool):
        """Record a transition in the episode, which includes full env state and the action taken"""
        t = Transition(
            state=state,
            action=action,
            reward=reward,
            is_done=done,
        )
        self.transitions.append(t)

    def get_chat_messages_for_logging(self) -> List[Dict]:
        """This should be only used for logging"""
        # get all chat messages except 'system' message
        messages = []
        for i, t in enumerate(self.transitions):
            # for user turn
            if i == 0:
                # add original question for the initial step
                question = self.question.strip()
                try:
                    user_content = t.state.user_prompt.format(question=question).strip()
                except Exception as _e:
                    return f"{t.state.user_prompt.strip()}\n\n{question}"
            else:
                user_content = t.state.user_prompt
            messages.append({'role': 'user', 'content': user_content})
            # for assistant turn
            assistant_content = t.action.text
            messages.append({'role': 'assistant', 'content': assistant_content})

        return messages


# for training


class DecodingConfig(BaseModel):
    """LLM decoding configuration for generation"""

    max_new_tokens: Optional[int] = Field(4096, ge=100, description='Maximum number of new tokens to generate')
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description='Sampling temperature for generation')
    top_k: Optional[int] = Field(0, ge=0, le=50000, description='Sampling top-k for generation')
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description='Sampling top-p for generation')
    do_sample: Optional[bool] = Field(True, description='Enable sampling for generation')


class ExplorationConfig(BaseModel):
    """Exploration configuration for epsilon-greedy policy"""

    init_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Initial exploration epsilon')
    min_epsilon: Optional[float] = Field(0.0, ge=0.0, le=1.0, description='Minimum exploration epsilon after decay')
    decay_steps: Optional[int] = Field(0, ge=0, le=500000, description='Number of steps for epsilon decay')
    max_explore_steps: Optional[int] = Field(
        0, ge=0, le=100, description='Maximum number of exploring start steps for generation'
    )


class ProcessedEpisode(BaseModel):
    """Episode data ready for PPO"""

    token_ids: Optional[np.ndarray] = Field(
        None, description='Full token ids of the episode include both user and assistant turns'
    )
    rewards: Optional[np.ndarray] = Field(
        None,
        description='Same size as token ids, mostly 0s, except for assistant turns we have scalar reward at the end-of-turn',
    )
    loss_masks: Optional[np.ndarray] = Field(
        None, description='Same size as token ids, user turns are 0s, assistant turns are 1s'
    )

    class Config:
        arbitrary_types_allowed = True


class BaseTrainingConfig(BaseModel):
    """Common training configuration"""

    checkpoint_enabled: Optional[bool] = Field(False, description='Enable to save model checkpoint')
    checkpoint_interval: Optional[int] = Field(0, ge=0, description='Frequency to save model checkpoint')
    checkpoint_keep_n: Optional[int] = Field(3, ge=1, description='Keep most recent N model checkpoints')
    num_epochs: int = Field(1, ge=1, le=100, description='Number of epochs to go through the dataset')
    value_loss_coef: float = Field(0.1, ge=0.0, le=1.0, description='Value function loss coefficient')
    gamma: float = Field(1.0, ge=0.0, le=1.0, description='Fallback default discount factor for compute returns')
    dynamic_discount: Optional[bool] = Field(False, description='Use dynamic discount')

    """For compute dynamic discount factor"""
    min_gamma: float = Field(0.999, ge=0.0, le=1.0, description='Minimum discount factor for compute returns')
    max_gamma: float = Field(0.9999, ge=0.0, le=1.0, description='Maximum discount factor for compute returns')
    max_expected_length: int = Field(
        10000, ge=1000, le=50000, description='Maximum sequence length when compute dynamic discount factor'
    )


class SFTConfig(BaseTrainingConfig):
    """For supervised fine-tuning training configuration"""

    policy_loss_coef: float = Field(1.0, ge=0.0, le=1.0, description='Policy loss coefficient')
    augment_rate: float = Field(0.5, ge=0.0, le=1.0, description='Rate to generate augmented samples')


class PPOConfig(BaseTrainingConfig):
    """For PPO training configuration"""

    gae_lambda: float = Field(0.95, ge=0.0, le=1.0, description='GAE lambda for advantages')
    policy_clip_eps: float = Field(0.2, ge=0.0, le=1.0, description='PPO policy loss clip epsilon')
    value_clip_eps: float = Field(0.2, ge=0.0, le=1.0, description='PPO value loss clip epsilon')
    normalize_rewards: bool = Field(False, description='Normalized rewards before compute advantages')
    normalize_advantages: bool = Field(True, description='Normalized rewards before compute PPO policy loss')
    kl_loss_coef: float = Field(0.01, ge=0.0, le=1.0, description='Token-level KL divergence coefficient')


class SFTSample(BaseModel):
    """SFT sample for training"""

    input_tokens: torch.Tensor = Field(..., description='A long tensor for token sequences from t=0, 1, ..., T-1')
    target_tokens: torch.Tensor = Field(..., description='A long tensor for token sequences from t=1, 2, ..., T-1, T')
    mc_returns: torch.Tensor = Field(
        ..., description='A float tensor for monte carlo returns corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    loss_masks: torch.Tensor = Field(
        ...,
        description='A boolean tensor (0s user tokens, 1s assistant tokens) corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    correctness: torch.Tensor = Field(
        ...,
        description='A bool tensor for mark if the sample has correct answer of not',
    )

    class Config:
        arbitrary_types_allowed = True


class PPOSample(BaseModel):
    """PPO transition for training"""

    states: torch.Tensor = Field(..., description='A long tensor for token sequences from t=0, 1, ..., T-1')
    actions: torch.Tensor = Field(..., description='A long tensor for token sequences from t=1, 2, ..., T-1, T')
    rewards: torch.Tensor = Field(
        ..., description='A float tensor for rewards corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    loss_masks: torch.Tensor = Field(
        ...,
        description='A boolean tensor (0s user tokens, 1s assistant tokens) corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    pi_logprobs: Optional[torch.Tensor] = Field(
        None, description='A float tensor for action logprobs corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    ref_logprobs: Optional[torch.Tensor] = Field(
        None,
        description='A float tensor for action logprobs from reference model corresponding to token sequences from t=1, 2, ..., T-1, T',
    )
    kl: Optional[torch.Tensor] = Field(
        None, description='A float tensor for token-level KL estimate corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    values: Optional[torch.Tensor] = Field(
        None, description='A float tensor for state values estimate corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    returns: Optional[torch.Tensor] = Field(
        None, description='A float tensor for returns estimate corresponding to token sequences from t=1, 2, ..., T-1, T'
    )
    advantages: Optional[torch.Tensor] = Field(
        None, description='A float tensor for GAE advantages estimate corresponding to token sequences from t=1, 2, ..., T-1, T'
    )

    class Config:
        arbitrary_types_allowed = True
