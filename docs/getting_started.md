# Getting Started with RL4LLM

Welcome to RL4LLM, a research-friendly Reinforcement Learning framework for LLM fine-tuning. This guide will help you set up your environment and run your first experiment.

## 1. Environment Setup

RL4LLM uses `pyproject.toml` for dependency management. To set up your environment, navigate to the root of the project and install the dependencies using `pip`:

```bash
pip install -e .[dev]
```

This command installs the core dependencies and the development dependencies (including `pytest` and `black`).

## 2. Running Experiments

Experiments are configured using YAML files located in the `configs/` directory and executed via scripts in the `scripts/` directory.

### General Workflow

Most training scenarios using SGLang for inference follow a two-step process:

1.  **Launch the SGLang Inference Server**: This dedicated server handles LLM generation requests during the RL algorithm's rollout phase.
2.  **Launch the Training Script**: This script, typically run with `deepspeed`, executes the RL fine-tuning logic, communicating with the SGLang server for model inferences.

### 2.1. Launching the SGLang Inference Server

The SGLang server is started using the `rl4llm.inference.launch_sgl_server` module.

**Command Structure:**

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path <MODEL_NAME_OR_PATH> \
    --host <HOST_ADDRESS> \
    --port <PORT_NUMBER> \
    --tp <TENSOR_PARALLEL_SIZE> \
    [--additional-sglang-options]
```

**Key SGLang Server Arguments & Tips:**

*   `--model-path`: HuggingFace model identifier or local path (e.g., `Qwen/Qwen2.5-0.5B`).
*   `--host`, `--port`: Network address for the server (e.g., `localhost`, `30000`).
*   `--tp`: Tensor parallelism degree (e.g., `1` for a single GPU).
*   `--enable-memory-saver`: **Recommended** when co-hosting the inference server and training on the same GPU(s). This helps reduce memory footprint. Requires `pip install torch-memory-saver`.
*   `--mem-fraction-static`: Adjusts the static GPU memory fraction for SGLang (e.g., `0.5` for 50%). Tune based on your GPU capacity and training needs.
*   `--chunked-prefill-size`: Can be useful for managing memory with long sequences (e.g., `8192`).
*   `--enable-custom-logit-processor`: Required if your training script uses custom logits processing during generation (e.g., for advanced exploration techniques).

### 2.2. Launching Training Scripts

Training scripts are executed using `deepspeed`.

**Command Structure:**

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=<GPU_IDS> NCCL_P2P_DISABLE=1 deepspeed --num_gpus=<NUM_GPUS> \
    scripts/<YOUR_TRAINING_SCRIPT.py> \
    --config-file ./configs/<YOUR_CONFIG_FILE.yaml>
```

**Key Training Script Arguments:**

*   `--config-file`: Path to the YAML configuration for the specific algorithm and task (e.g., `./configs/grpo_config.yaml`). All inference server details are now configured within this YAML file.
*   `CUDA_VISIBLE_DEVICES`, `--num_gpus`: Standard DeepSpeed/CUDA settings to specify GPU usage. The examples below use a single GPU.

---

## Examples

Below are common use-cases. Remember to adjust model paths, config files, and SGLang server parameters as needed for your specific setup and experiments.

### Example 1: GRPO Fine-Tuning (Standard)

This demonstrates basic GRPO fine-tuning on a single server where SGLang and training share resources.

**Step 1**: Launch the SGLang inference server.

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

**Step 2**: Launch the GRPO training script.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 \
    scripts/run_train_grpo.py \
    --config-file ./configs/grpo_config.yaml
```

---

### Example 2: GRPO Fine-Tuning with Tool-Use

This example adapts GRPO for tasks requiring tool usage. Ensure the base model is suitable for tool invocation (e.g., an "Instruct" or "Chat" fine-tuned model).

**Step 1**: Launch the SGLang inference server.

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host localhost \
    --port 30000 \
    --tp 1 \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.5 \
    --enable-memory-saver
```

**Step 2**: Launch the GRPO tool-use training script.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 \
    scripts/run_train_grpo_tools.py \
    --config-file ./configs/grpo_tools_config.yaml
```

---

### Example 3: PPO Fine-Tuning (Two-Stage Process)

PPO training often benefits from a pre-trained value model. This example outlines the typical two-stage approach, often used for tasks like mathematical reasoning (e.g., GSM8K).

**Stage 1: Bootstrap Value Model**

The aim is to train a value head for your base model. This involves generating rollouts (often with a fixed policy) and using task-specific rewards to train the value function.

> [!IMPORTANT]
> The value model and the policy model (used in Stage 2) must share the same base architecture and tokenizer for compatibility.

**Step 1.1**: Launch the SGLang inference server.

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

**Step 1.2**: Launch the value network training script.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 \
    scripts/run_train_value_net.py \
    --config-file ./configs/value_net_config.yaml
```

**Stage 2: PPO Training**

After Stage 1, you will have a trained value model checkpoint. Update your PPO configuration file (e.g., `ppo_config.yaml`) to point to this value model checkpoint.

**Step 2.1**: Launch the SGLang inference server.

*(You might need to adjust `--mem-fraction-static` based on the combined memory needs of the policy and value models during PPO training).*

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-0.5B \
    --host localhost \
    --port 30000 \
    --tp 1 \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.3 \
    --enable-memory-saver
```

**Step 2.2**: Launch the PPO training script.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 \
    scripts/run_train_ppo.py \
    --config-file ./configs/ppo_config.yaml
```
