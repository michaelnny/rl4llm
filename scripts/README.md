
## Example 1: GRPO Fine-Tuning with SGLang inference on a single server

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


## Example 2: Extend GRPO Fine-Tuning with SGLang inference

**Step 1**: Launch the SGLang inference server with `--enable-memory-saver`

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-0.5B \
    --host localhost \
    --port 30000 \
    --tp 1 \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.5 \
    --enable-memory-saver \
    --enable-custom-logit-processor
```


**Step 2**: Launch the training script and set the inference server arguments

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_extended_grpo.py \
    --config-file ./configs/extended_grpo_config.yaml \
    --use-infer-server \
    --infer-host localhost \
    --infer-port 30000 \
    --infer-cohost-mode
```




## Example 3:  PPO Fine-Tuning on GSM8K Training with SGLang inference on a single server

**Stage 1: Bootstrap Value Model**

Initialize a random value head for the model. Run a fixed policy to generate rollout samples with rule-based rewards. Compute Monte Carlo (MC) returns and use them to train the value model.

> [!IMPORTANT]
> Ensure the value and policy models share the same tokenizer, as policy-generated tokens are used to train the value model.

**Step 1**: Start inference server:
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

**Step 2**: Start training script:
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_value_net.py \
    --config-file ./configs/value_net_config.yaml \
    --use-infer-server \
    --infer-host localhost \
    --infer-port 30000 \
    --infer-cohost-mode
```

**Stage 2: PPO Training**

Adapt the checkpoint files path inside the yaml config file for the value model. Then begin PPO training using the value model checkpoint from Stage 1.

**Step 1**: Start inference server:
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

**Step 2**: Start training script:
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_ppo.py \
    --config-file ./configs/ppo_config.yaml \
    --use-infer-server \
    --infer-host localhost \
    --infer-port 30000 \
    --infer-cohost-mode
```
