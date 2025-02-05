```bash

# Login to remote server
ssh -p 37541 root@86.57.175.52 -L 8080:localhost:8080


# Install CUDA Toolkit and MPICH for deepspeed
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 libmpich-dev


# On local machine, copy project files to remote server
rsync -avz -e "ssh -p 37541" --exclude='.*' --exclude='__pycache__/' --exclude='tests' --exclude='runs' ./rl4llm root@86.57.175.52:/project/


# Install packages
cd /project/rl4llm

pip install -r requirements.txt



```

Run training script

```bash

cd /project/rl4llm


PYTHONPATH=src TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=4 src/rl4llm/scripts/run_train_sft.py --config-file ./configs/sft_train_config.yaml




nohup sh -c "PYTHONPATH=src NCCL_P2P_DISABLE=1 deepspeed --num_gpus=4 src/rl4llm/scripts/run_train_sft.py --config-file ./configs/sft_train_config.yaml" &



PYTHONPATH=src TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=2 src/rl4llm/scripts/run_train_ppo.py --config-file ./configs/ppo_train_config.yaml



nohup sh -c "PYTHONPATH=src NCCL_P2P_DISABLE=1 deepspeed --num_gpus=2 src/rl4llm/scripts/run_train_ppo.py --config-file ./configs/ppo_train_config.yaml" &




pkill -f "src/rl4llm/scripts/run_train"

pkill -f "python"

```

To monitoring the job, open tensorboard

```bash

rsync -avz -e "ssh -p 37541" --exclude='.pt' --exclude='checkpoints' root@86.57.175.52:/project/rl4llm/runs ./rl4llm


tensorboard --logdir ./rl4llm/runs --samples_per_plugin=text=1000

```
