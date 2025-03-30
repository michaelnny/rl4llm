
## TODO

Use the OpenAI MATH split dataset for training and evaluation

https://github.com/openai/prm800k/tree/main/prm800k/math_splits



## Preparation


```bash

# Install packages
cd /project/rl4llm

pip install -r requirements.txt

```





Start training job



```bash

PYTHONPATH=src CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_grpo.py --config-file ./configs/grpo_config.yaml


```


To monitoring the job, open tensorboard

```bash

tensorboard --logdir ./runs --bind_all --samples_per_plugin=text=10000

```
