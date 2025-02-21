
## TODO

Use the OpenAI MATH split dataset for training and evaluation

https://github.com/openai/prm800k/tree/main/prm800k/math_splits



## Preparation


```bash

# Install CUDA Toolkit and MPICH for deepspeed
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 libmpich-dev


# Install packages
cd /project/rl4llm

pip install -r requirements.txt

```



Start training job

```bash

PYTHONPATH=src python -m rl4llm.scripts.run_train_grpo

```


```bash

PYTHONPATH=src TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_grpo_train_config.yaml


```


To monitoring the job, open tensorboard

```bash

tensorboard --logdir ./runs --bind_all --samples_per_plugin=text=10000

```












Run on remote server

```bash

# Login to remote server
ssh -p 19187 root@175.155.64.221 -L 8080:localhost:8080


# Install CUDA Toolkit and MPICH for deepspeed
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 libmpich-dev


# On local machine, copy project files to remote server
rsync -avz -e "ssh -p 19187" --exclude='.*' --exclude='__pycache__/' --exclude='tests' --exclude='old_runs' --exclude='runs' ./rl4llm root@175.155.64.221:/project/


# Install packages
cd /project/rl4llm

pip install -r requirements.txt



```

Run training script

```bash

cd /project/rl4llm




PYTHONPATH=src TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=4 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_grpo_train_config.yaml



nohup sh -c "PYTHONPATH=src NCCL_P2P_DISABLE=1 deepspeed --num_gpus=4 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_grpo_train_config.yaml" &




pkill -f "run_train_grpo"


```


Copy experiment runs logs from remove to local machine

```bash

rsync -avz -e "ssh -p 19187"  --exclude='checkpoints' root@175.155.64.221:/project/rl4llm/runs ./rl4llm

```
