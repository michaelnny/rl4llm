# Reinforcement Learning Framework for LLM Fine-Tuning

This project provides an easy-to-use, research-friendly framework to fine-tune Large Language Models (LLMs) using Reinforcement Learning (RL). It simplifies new algorithm research by managing low-level tasks, allowing researchers to focus on algorithm development.

## Key Features

- **Customizable Environments (MDPs)**: Easily generate and manage samples.
- **Efficient Inference**: Support for HuggingFace Transformers, vLLM, and fast SGLang inference servers.
- **Scalable Training**: Uses DeepSpeed for optimized model training.
- **Clean Architecture**: Clearly separates low-level operations from algorithmic logic.

> **Note:** Currently only tested on single-node setups.



## Quick Setup Guide

### Step 1: Install NVIDIA CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6 cuda-drivers
```

### Step 2: Install Project Dependencies

```bash
cd /project/rl4llm
pip install -r requirements.txt
```



## Framework Overview

The RL framework operates by separating sample generation (`Envs`) from the main training loop. Here’s a simple diagram:

```
Sample Generation (Envs)
          │
          ▼
   Training Loop (RL Algorithm)
          │
          ▼
    Logging & Monitoring
```

### Dataset Structure

Each environment expects a dataset with two fields:
- `prompt` (formatted input for the model)
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
            reward = random.rand()
            rewards.append(reward)
        return rewards
```

For multiple rewards, use a `reward_transform_fn` to combine them into a single signal.

---

### Creating Custom Environments

Extend the `BaseEnv` class to implement unique logic, multi-step scenarios, or custom sampling:

```python

from rl4llm.core.base_env import BaseEnv, EpisodeData

class MyCustomEnv(BaseEnv):

    def rollout(self,
        llm: Any,
        sampling_params: Dict,
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[EpisodeData]:
        # Your sampling logic
```



## Fast Sample Generation with SGLang

Launch an efficient inference server to accelerate sample generation:

### Start the inference server:

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-0.5B \
    --host localhost --port 30000 \
    --enable-memory-saver
```

### Begin training using the inference server:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_grpo.py \
    --config-file ./configs/grpo_config.yaml \
    --use-infer-server --infer-host localhost --infer-port 30000 --infer-cohost-mode
```



## Centralized Logging & Monitoring

Track performance easily using the built-in logging manager. It supports:
- Metric aggregation (mean, max, min, etc.)
- Resource monitoring (CPU/GPU)
- Logging samples and metrics to files or Tensorboard

### Logging example:

```python
from rl4llm.core.distributed import DistributedManager
from rl4llm.logging import LoggingManager

dist_manager = DistributedManager()
logger = LoggingManager(dist_manager, **log_config)

for step in range(100):
    with logger.timer('generation'):
        logger.log_scalar('objective/reward', 0.5)
        logger.log_sample('train', {"prompt": "How many workdays in a week?", "completion": "5", "reward": 1.0})

    with logger.timer('train'):
        logger.log_scalar('train/loss', 0.2)

    logger.aggregate_and_log(step)

logger.close()
```



# License

This project is licensed under the MIT License, see the LICENSE file for details

# Citing our work

If you reference or use our project in your research, please cite our work:



```bibtex
@software{rl4llm2025github,
  title = {{RL 4 LLM}: A research friendly Reinforcement Learning Framework for LLM Fine-Tuning},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/rl4llm},
  version = {1.0.0},
  year = {2025},
}
```
