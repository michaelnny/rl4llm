# RL4LLM: A Research-Friendly RL Framework for LLM Fine-Tuning

This framework provides a modular and extensible platform for fine-tuning Large Language Models (LLMs) using Reinforcement Learning (RL). It is designed to simplify RL research for LLMs by abstracting low-level details.

> [!IMPORTANT]
> Currently validated on single-node setups with smaller LLMs. Contributions for broader testing and feature enhancements are welcome.

## Key Features

*   **Modular Design**: Clear separation of concerns, allowing researchers to focus on algorithmic innovations.
*   **Customizable Environments**: Easily adapt or create new environments for diverse tasks (e.g., tool-use, multi-turn dialogue) by extending `BaseMDPEnv`.
*   **Flexible Inference**: Supports efficient inference engines like SGLang and standard HuggingFace Transformers.
*   **Optimized Training**: Integrates with DeepSpeed for efficient model training.

## Framework Overview

The system typically involves a training process that interacts with an inference server for generating environment rollouts. Model weights are synchronized between these components.

```
┌──────────────────────┐                           ┌────────────────────────────────────┐
│ Inference Server     │                           │ DeepSpeed Training Server          │
│ (e.g., SGLang)       │                           │                                    │
│                      │ <─────── Rollout ───────> │ ┌────────────┐      ┌────────────┐ │
│                      │          Requests         │ │ Inference  │      │ Inference  │ │
│                      │                           │ │ Client     │      │ Client     │ │
│                      │                           │ └────────────┘      └────────────┘ │
│                      │                           │ ┌────────────┐      ┌────────────┐ │
│                      │                           │ │ Env 0      │      │ Env 1      │ │
│                      │                           │ └────────────┘      └────────────┘ │
│                      │                           │       │                   │        │
│                      │                           │       ▼                   ▼        │
│                      │                           │ ┌────────────┐      ┌────────────┐ │
│                      │                           │ │ Rank 0     │      │ Rank 1     │ │
│                      │                           │ └────────────┘      └────────────┘ │
│                      │                           └────────────────────────────────────┘
│                      │                                   │
│                      │                                   │ Save Model Weights
│                      │                                   ▼
│                      │                          ┌──────────────────────┐
│                      │<───── Load Weights ──────│ Shared File System   │
└──────────────────────┘                          └──────────────────────┘
```
> [!TIP]
> The inference server and training can also be co-hosted on the same machine.

## Supported RL Algorithms

| Algorithm                                                    | Paper                                                                 | Key Features                                                                                              |
| :----------------------------------------------------------- | :-------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **Proximal Policy Optimization (PPO)**                       | [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)              | SGLang inference, DeepSpeed training, Value Model bootstrapping.                                          |
| **Group Relative Policy Optimization (GRPO)**                | [Shao et al., 2024](https://arxiv.org/abs/2402.03300)                   | SGLang inference, DeepSpeed training.                                                                     |
| **Decoupled Clip and Dynamic sAmpling Policy Opt. (DAPO)** | [Yu et al., 2025](https://arxiv.org/abs/2503.14476) | SGLang inference, DeepSpeed training. Based on GRPO.|

## Running Experiments

The `scripts/` directory contains example scripts for running various RL algorithms and tasks. Each script typically involves:
1.  Launching an inference server (e.g., SGLang).
2.  Running the main training script with `deepspeed`, pointing to a configuration file.

**For detailed instructions and specific examples, please refer to the `README.md` file within the `scripts/` directory.**

## Customization

The framework is designed for easy extension.

### 1. Custom Reward Functions

Define task-specific rewards by subclassing `BaseRewardFunction`.

> [!IMPORTANT]
> Reward functions are designed to compute a **terminal reward** once an episode (e.g., a full multi-turn dialogue or task completion) is finished.


```python
from typing import Any, Dict, List, Union
from rl4llm.core.base_env import BaseRewardFunction, ChatMessage

class CustomReward(BaseRewardFunction):
    def __init__(self, name: str = "custom_reward_name", **kwargs):
        super().__init__(name)
        # Initialize any parameters specific to your reward

    def __call__(
        self,
        messages: List[ChatMessage],
        ground_truth: Union[str, float, int],
        **kwargs: Dict[str, Any],
    ) -> float:
        # Implement your reward logic here
        # Example: Access the last model completion
        # completion = messages[-1].content
        # return score based on completion and ground_truth
        return 0.0 # Placeholder
```

If your task involves multiple distinct reward signals (e.g., accuracy and length), you can define multiple `BaseRewardFunction` instances. Then, a `reward_transform_fn` must be provided to the environment to combine these signals into a single scalar reward for the RL algorithm.

```python
# Example reward_transform_fn
def my_reward_transformer(reward_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # reward_dict contains outputs from all registered reward functions
    # e.g., reward_dict = {'accuracy_reward': tensor([0.8]), 'length_reward': tensor([0.5])}
    combined_reward = 0.7 * reward_dict['accuracy_reward'] + 0.3 * reward_dict['length_reward']
    return combined_reward
```

### 2. Custom Environments

Create new environments by subclassing `BaseMDPEnv` and implementing the `_run_interaction_loop` method. This method defines how the agent interacts with the LLM to generate trajectories.

```python
from typing import Any, Dict, Optional
import torch
from rl4llm.core.base_env import BaseMDPEnv, EnvState

class CustomEnv(BaseMDPEnv):
    @torch.inference_mode()
    def _run_interaction_loop(
        self,
        env_state: EnvState,       # Initial state of the environment batch
        llm: Any,                  # LLM interface (e.g., inference client)
        sampling_params: Dict[str, Any], # Parameters for LLM generation
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        # Implement your custom interaction logic here.
        # This involves:
        # 1. Preparing prompts from env_state.
        # 2. Calling the llm to get completions.
        # 3. Processing completions and updating env_state (e.g., adding new messages).
        # 4. Determining if episodes are done.
        return env_state # Return the updated environment state
```
Refer to existing environments in `rl4llm/envs/` for practical examples (e.g., `SglToolMDPEnv` for tool use).

## SGLang Integration

*   **Co-hosting**: For running SGLang and training on the same server, use the `--enable-memory-saver` flag with the SGLang server (requires `pip install torch-memory-saver`).
*   **Weight Synchronization**: Model weights are typically synchronized via checkpoint files. Ensure a shared file system if the inference server and training run on different machines. The path is set in `log_config.output_dir`.

## Logging

The framework includes a logging manager for metrics (TensorBoard support), episode data, and basic resource monitoring. See `BaseTrainer` for usage.

## Known Issues

*   SGLang with `--enable-memory-saver` might occasionally hang, possibly due to CUDA OOM. Consider reducing memory allocation or using GPUs with more VRAM.
*   WandB logging is not fully tested.

## Contributing

We welcome contributions. Please submit issues or pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this framework in your research, please consider citing:

```bibtex
@software{rl4llm_project_2025,
  author = {Michael Hu and Contributors},
  title = {{RL4LLM}: A Research-Friendly Reinforcement Learning Framework for LLM Fine-Tuning},
  url = {https://github.com/michaelnny/rl4llm},
  version = {0.1.0},
  year = {2025}
}
