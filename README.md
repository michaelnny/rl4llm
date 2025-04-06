
## TODO

Use the OpenAI MATH split dataset for training and evaluation

https://github.com/openai/prm800k/tree/main/prm800k/math_splits



## Preparation

**Install NVIDIA CUDA 12.6**
```bash

# Install NVIDIA CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Install NVIDIA Driver
sudo apt-get install -y cuda-drivers

```


**Install packages**
```bash

cd /project/rl4llm

pip install -r requirements.txt

```





Start training job



```bash

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_grpo.py --config-file ./configs/grpo_config.yaml

```


To monitoring the job, open tensorboard

```bash

tensorboard --logdir ./runs --bind_all --samples_per_plugin=text=10000

```
