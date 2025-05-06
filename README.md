# Reinforcement Learning Framework for LLM Fine-Tuning

This project provides a light-weight, research-friendly framework to fine-tune Large Language Models (LLMs) using Reinforcement Learning (RL). It simplifies new algorithm research by managing low-level tasks, with clear component separation design, allowing researchers to focus on algorithm development.

## Key Features

- **Customizable Environments**: Easily generate samples for various tasks.
- **Efficient Inference**: Support for fast SGLang inference and native HuggingFace transformer model generation.
- **Scalable Training**: Uses DeepSpeed for optimized model training.
- **Clean Architecture**: Clearly separates low-level operations from algorithmic logic.

> [!IMPORTANT]
> Currently only tested on single-node setups with a tiny size LLM. Need support/volunteers to help with testing and improvements.


## Supported RL Algorithms

| Algorithm | Key Features |
|-----------|--------------|
| **Proximal Policy Optimization (PPO)** | - SGLang for high-performance inference<br>- DeepSpeed for training<br>- Value Model bootstrapping |
| **Group Relative Policy Optimization (GRPO)** | - SGLang for high-performance inference<br>- DeepSpeed for training |


## Framework Overview

The core of the framework was the modular components, especially the RL `Envs`, Inference Server/Client. This makes it possible to quickly adapt training on new tasks, and also makes it extremely easy to work with special inference engines like `SGLang`.

Here’s a simple diagram:

```
┌──────────────────────┐                           ┌────────────────────────────────────┐
│ SGLang Inference     │                           │ DeepSpeed Training Server          │
│ Server               │                           │                                    │
│                      │                           │ ┌────────────┐      ┌────────────┐ │
│                      │ <─────── Rollout ───────> │ │ Inference  │      │ Inference  │ │
│                      │          Requests (HTTP)  │ │ Client     │      │ Client     │ │
│                      │                           │ └────────────┘      └────────────┘ │
│                      │                           │ ┌────────────┐      ┌────────────┐ │
│                      │                           │ │ Env 0      │      │ Env 1      │ │
│                      │                           │ │ (Dataset 0)│      │ (Dataset 1)│ │
│                      │                           │ └────────────┘      └────────────┘ │
│                      │                           │       │                   │        │
│                      │                           │       ▼                   ▼        │
│                      │                           │ ┌────────────┐      ┌────────────┐ │
│                      │                           │ │ Rank 0     │      │ Rank 1     │ │
│                      │                           │ │ (Master)   │      │            │ │
│                      │                           │ └────────────┘      └────────────┘ │
│                      │                           └────────────────────────────────────┘
│                      │                                   │
│                      │                                   │ Save Model
│                      │                                   │ Weights
│                      │                                   ▼
│                      │                          ┌──────────────────────┐
│                      │<───── Load Weights ──────│ Shared File System   │
│                      │         Path             └──────────────────────┘
└──────────────────────┘
```

> [!TIP]
> Check the example at `scripts` on how to use the `SGLangClient` that uses HTTP to call the inference server, and an `SglMDPEnv` that can handle sample generation.

> [!TIP]
> Following the modular design, we can also run the SGLang inference server and deepspeed training on the single server as in `co-hosting mode`.


## Sample Generation Environments

We adapt the MDP environment concept from classic RL, where sample generation was handled inside the MDP environment. The idea is the MDP env will collect the rollout and return the episode samples in a unified data structure. This avoids cluttering the actual RL algorithm.

This is especially useful in the context of LLM as we often need to handle specific tasks (MATH, coding, tools); some are more complex and require special handling. This makes adapting RL to new tasks easy as we don't need to make any changes to the RL algorithm and training code. Instead, we only need to focus on building the environment.

### Dataset Structure

Each environment expects a dataset with at least two fields:
- `messages` (chat-style message)
- `ground_truth` (correct answer used for rewards)

---

### Customizing Reward Functions

You can easily define your own reward functions by subclassing the provided `BaseRewardFunction`.

Example:

```python

from rl4llm.core.base_env import ChatMessage, BaseRewardFunction

class AccuracyRewardFunction(BaseRewardFunction):

    def __init__(self, name='accuracy_reward'):
        super().__init__(name)

    def __call__(
        self,
        messages: List[ChatMessage],
        ground_truth: Union[str | float | int],
        **kwargs: Dict[str, Any],
    ) -> List[float]:

        # get last completion
        completion = messages[-1].content

        return math_problem_grader(
            full_answer=completion,
            ground_truth=ground_truth,
        )

```

For task using multiple rewards, consider use a `reward_transform_fn` to transformer them into a single signal. For example here's an very simple example of handling mixed rewards.

```python
def reward_transform_fn(reward_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform multiple rewards to single reward for a batch of samples"""
    accuracy_rewards = reward_dict['accuracy_reward']  # [batch_size]
    format_reward = reward_dict['format_reward']  # [batch_size]

    return 0.8 * accuracy_rewards + 0.2 * format_reward


train_env = SglMDPEnv(
    ...
    reward_transform_fn=reward_transform_fn,
)
```

---

### Creating Custom Environments

It's very easy to build a custom environment, for example, multi-step scenarios or custom sampling with logits processors. All we need to do is extend the `BaseMDPEnv` class and implement the `rollout` logic.

```python

from rl4llm.core.base_env import BaseMDPEnv, EnvState

class MyCustomEnv(BaseMDPEnv):

    @torch.inference_mode()
    def _run_interaction_loop(
        self,
        env_state: EnvState,
        llm: Any,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        # Your own loop
```

> [!TIP]
> Checking the examples at `rl4llm.envs.explore_env.py` for a comprehensive example of custom environment using custom logits processor with SGLang inference engine.


## Fast Generation with SGLang

Launch an efficient inference server to accelerate sample generation. You can either run the SGLang engine on the same server, or on separate servers. We have a simplified FastAPI server adapted from the original SGLang HTTP server located at `rl4llm.inference.sgl_http_server`, which you can launch with the custom script `rl4llm.inference.launch_sgl_server`.

> [!NOTE]
> If running SGLang inference and deepspeed training on the same server with co-hosting mode, make sure use the `--enable-memory-saver`, this requires install the `pip install torch-memory-saver`.

> [!NOTE]
> We use model checkpoint file to sync the weights between training instance and the SGLang inference engine. If you run inference engine and training in separate servers, make sure you have a shared file system between them. The weights saving path is defined at the `log_config.output_dir` when launching the trainer.


## Centralized Logging & Monitoring

Track performance easily using the built-in logging manager. It supports:
- Metric aggregation (mean, max, min, etc.)
- Basic resource monitoring (CPU/GPU)
- Logging samples to files and metrics to Tensorboard

More example of the logging manager can be found at the `BaseTrainer.train` and `BaseTrainer.log_batch_episodes`.


## Know Issues

- When running SGLang with `--enable-memory-saver`, sometimes the inference server will hangs when we try to release/resume the memory. The most likely cause is due to CUDA OOM, try reduce the memory fraction or using more powerful GPU.


## License

This project is licensed under the MIT License, see the LICENSE file for details.


## Contribute and Collaboration

We are looking for contributions and collaboration to make the framework more robust, and also to expand it to support additional features and algorithms. Issue reports and PRs are welcome.


## Citing our work

If you reference or use our project in your research, please cite our work:


```bibtex
@software{the_rl4llm_project,
  title = {{RL 4 LLM}: A research friendly Reinforcement Learning Framework for LLM Fine-Tuning},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/rl4llm},
  version = {0.1.0},
  year = {2025},
}
```
