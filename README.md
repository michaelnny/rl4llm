# Reinforcement Learning Framework for LLM Fine-Tuning

This project provides an easy-to-use, research-friendly framework to fine-tune Large Language Models (LLMs) using Reinforcement Learning (RL). It simplifies new algorithm research by managing low-level tasks, with clear component separation design, allowing researchers to focus on algorithm development.

## Key Features

- **Customizable Environments**: Easily generate samples for various tasks.
- **Efficient Inference**: Support for fast SGLang inference and native HuggingFace transformer model generation.
- **Scalable Training**: Uses DeepSpeed for optimized model training.
- **Clean Architecture**: Clearly separates low-level operations from algorithmic logic.

> [!IMPORTANT]
> Currently only tested on single-node setups with a tiny size LLM. Need support/volunteers to help with heavy testing and improvements.



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
> Following the modular design, we can also run the SGLang inference server and deepspeed training on the single server as in `co-hosting mode`.


### Example of start GRPO training with SGLang inference on a single server:

**Step 1**: Launch the SGLang inference server with `--enable-memory-saver`

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-0.5B \
    --host localhost \
    --port 30000 \
    --tp 1 \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.5 \
    --enable-memory-saver
```


**Step 2**: Launch the training script and set the inference server arguments

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_grpo.py \
    --config-file ./configs/grpo_config.yaml \
    --use-infer-server \
    --infer-host localhost \
    --infer-port 30000 \
    --infer-cohost-mode
```


## Sample Generation Environments

We adapt the MDP environment concept from classic RL, where sample generation was handled inside the MDP environment. The idea is the MDP env will collect the rollout and return the episode samples in a unified data structure. This avoids cluttering the actual RL algorithm.

This is especially useful in the context of LLM as we often need to handle specific tasks (MATH, coding, tools); some are more complex and require special handling. This makes adapting RL to new tasks easy as we don't need to make any changes to the RL algorithm and training code. Instead, we only need to focus on building the environment.

### Dataset Structure

Each environment expects a dataset with at least two fields:
- `prompt` (pre-formatted input for the model)
- `ground_truth` (correct answer used for rewards)

---

### Customizing Reward Functions

You can easily define your own reward functions by subclassing the provided `BaseRewardFunction`.

Example:

```python
from rl4llm.core.base_env import BaseRewardFunction

class AccuracyRewardFunction(BaseRewardFunction):

    def __init__(self, name='accuracy_reward'):
        super().__init__(name)

    def __call__(self, completions, ground_truths, **kwargs):
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if len(ground_truths) == 1:
            ground_truths = [ground_truths] * len(completions)
        if len(completions) != len(ground_truths):
            raise ValueError(
                'Completion and ground truth have mismatch elements'
            )

        for completion, truth in zip(completions, ground_truths):
            rewards.append(random.rand()) # random scores
        return rewards
```

For task using multiple rewards, consider use a `reward_transform_fn` to transformer them into a single signal. For example here's an very simple example of handling mixed rewards.

```python
def reward_transform_fn(reward_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform multiple rewards to single reward for a batch of samples"""
    accuracy_rewards = reward_dict['accuracy_reward']  # [batch_size]
    format_reward = reward_dict['format_reward']  # [batch_size]

    return 0.8 * accuracy_rewards + 0.2 * format_reward


trainer = GRPOTrainer(
    ...
    reward_transform_fn=reward_transform_fn,
)
```

---

### Creating Custom Environments

It's very easy to build a custom environment, for example, multi-step scenarios or custom sampling with logits processors. All we need to do is extend the `BaseEnv` class and implement the `rollout` logic.

```python

from rl4llm.core.base_env import BaseEnv, EpisodeData

class MyCustomEnv(BaseEnv):

    def rollout(self,
        llm: Any,
        sampling_params: Dict,
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[EpisodeData]:
        # Your own sampling logic
```

> [!TIP]
> Checking the examples at `rl4llm.envs.explore_env.py` for a comprehensive example of custom environment using SGLang inference engine and HF model.


## Fast Generation with SGLang

Launch an efficient inference server to accelerate sample generation. You can either run the SGLang engine on the same server, or on separate servers. We have a simplified FastAPI server adapted from the original SGLang HTTP server located at `rl4llm.inference.sgl_http_server`, which you can launch with the custom script `rl4llm.inference.launch_sgl_server`.

> [!NOTE]
> If running SGLang inference and deepspeed training on the same server with co-hosting mode, make sure use the `--enable-memory-saver`, this requires install the `pip install torch-memory-saver`.

> [!NOTE]
> We use model checkpoint file to sync the weights between training instance and the SGLang inference engine. If you run inference engine and training in separate servers, make sure you have a shared file system between them. The weights saving path is defined at the `artifacts_path` when launching the trainer.


### How it works - InferenceEnv and InferenceClient

In order to handle the communications for sample generation and weight updates, we created a simple `SGLangClient` that uses HTTP to call the inference server, and an `InferenceEnv` that can handle sample generation using the inference client.

> [!TIP]
> Check the example at `scripts/run_train_grpo.py` on how to use them.


## Centralized Logging & Monitoring

Track performance easily using the built-in logging manager. It supports:
- Metric aggregation (mean, max, min, etc.)
- Basic resource monitoring (CPU/GPU)
- Logging samples to files and metrics to Tensorboard

More example of the logging manager can be found at the `BaseTrainer.train` and `BaseTrainer.log_batch_episodes`.


## Know Issues

- When running SGLang with `--enable-memory-saver`, sometimes the server will hangs when we try to release/resume the memory. The most likely cause is due to CUDA OOM.


## License

This project is licensed under the MIT License, see the LICENSE file for details.


## Contribute and Collaboration

We are looking for contributions and collaboration to make the framework more robust, and also to expand it to support additional features and algorithms. Issue reports and PRs are welcome.


## Citing our work

If you reference or use our project in your research, please cite our work:


```bibtex
@software{rl4llm2025github,
  title = {{RL 4 LLM}: A research friendly Reinforcement Learning Framework for LLM Fine-Tuning},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/rl4llm},
  version = {0.1.0},
  year = {2025},
}
```
