
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

nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/grpo_config.yaml" > standard_grpo.log &


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/group_grpo_config.yaml" > group_grpo.log &


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/explore_grpo_config.yaml" > explore_grpo.log &


```


```bash

PYTHONPATH=src CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_grpo_config.yaml


```


To monitoring the job, open tensorboard

```bash

tensorboard --logdir ./runs --bind_all --samples_per_plugin=text=10000

```