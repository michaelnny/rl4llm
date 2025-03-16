
## Run on Hyperbolic labs

```bash

# Login to remote server
ssh ubuntu@timely-banana-squid.1.cricket.hyperbolic.xyz -p 31466

# Install MPICH for deepspeed and remote file transfers
sudo apt update
sudo apt install -y python3-pip libmpich-dev rsync zstd

pip3 install packaging torch==2.5.* torchaudio==2.5.*


# On local machine, copy project files to remote server
rsync -avz -e "ssh -p 31466" --exclude='.*' --exclude='__pycache__/' --exclude='notebooks' --exclude='tests' --exclude='old_runs' --exclude='runs' ./rl4llm ubuntu@timely-banana-squid.1.cricket.hyperbolic.xyz:/home/ubuntu


# Install packages
cd /home/ubuntu/rl4llm

pip install -r requirements.txt

```

Run training script

```bash

cd /home/ubuntu/rl4llm




PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/grpo_config.yaml


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/grpo_config.yaml" >> standard_grpo.log &


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/explore_grpo_config.yaml" >> explore_grpo.log &


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/discount_grpo_config.yaml" >> discount_grpo.log &



# to download the model and quick smock runs
PYTHONPATH=src TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_grpo_config.yaml



# multiple GPUs
nohup sh -c "PYTHONPATH=src NCCL_P2P_DISABLE=1 deepspeed --num_gpus=4 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_grpo_config.yaml" >> enhanced_grpo.log &





pkill -f "run_train_grpo"


```


Copy experiment runs logs from remove to local machine

```bash

rsync -avz -e "ssh -p 31883" --exclude='checkpoints' --exclude='samples' ubuntu@worrisome-cherry-tiger.1.cricket.hyperbolic.xyz:/home/ubuntu/rl4llm/runs ./


rsync -avz -e "ssh -p 31883"  --exclude='checkpoints' ubuntu@worrisome-cherry-tiger.1.cricket.hyperbolic.xyz:/home/ubuntu/rl4llm/runs ./

```


```bash
# compress checkpoint files before transfer
tar -I 'zstd --ultra -22 -T0' -cvf model_checkpoint_iteration_50.tar.zst iteration_50/

split -b 1G model_checkpoint_iteration_50.tar.zst model_checkpoint_iteration_50.tar.zst.part-



# copy from remote server to local machine using parallel transfers
ssh -p 31883 ubuntu@worrisome-cherry-tiger.1.cricket.hyperbolic.xyz "ls /home/ubuntu/rl4llm/runs/enhanced_grpo_qwen2.5_7b_math/checkpoints/model_checkpoint_iteration_50.tar.zst.part-*" | \
parallel -j8 "scp -P 31883 ubuntu@worrisome-cherry-tiger.1.cricket.hyperbolic.xyz:{} ./"


# to decompress after transfer
cat model_checkpoint_iteration_50.tar.zst.part-* > model_checkpoint_iteration_50.tar.zst
tar -I zstd -xvf model_checkpoint_iteration_50.tar.zst
```



###  Problems with Hyperbolic POD

- Missing common default libraries like `rsync` for file copy, `libmpich-dev` for distributed training
- After install deepspeed, you need to re-login SSH, otherwise we get `deepspeed` command not found










## Run on VastAI
```bash

ssh -p 19764 root@175.155.64.160 -L 8080:localhost:8080



# Required for flash attention
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 libmpich-dev




rsync -avz -e "ssh -p 19764" --exclude='.*' --exclude='__pycache__/' --exclude='notebooks' --exclude='tests' --exclude='old_runs' --exclude='runs' --exclude='*.md' ./rl4llm root@175.155.64.160:/workspace/



cd /workspace/rl4llm


pip3 install -r requirements.txt




nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/grpo_config.yaml" >> standard_grpo.log &


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/explore_grpo_config.yaml" >> explore_grpo.log &


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=2 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/discount_grpo_config.yaml" >> discount_grpo.log &


nohup sh -c "PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 python3 -m rl4llm.scripts.run_train_grpo --config-file ./configs/discount_explore_grpo_config.yaml" >> discount_explore_grpo.log &





nohup sh -c "PYTHONPATH=src NCCL_P2P_DISABLE=1 deepspeed --num_gpus=4 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_grpo_config.yaml" > standard_grpo.log &




nohup sh -c "PYTHONPATH=src NCCL_P2P_DISABLE=1 deepspeed --num_gpus=4 src/rl4llm/scripts/run_train_grpo_dist.py --config-file ./configs/ds_explore_grpo_config.yaml" > explore_grpo.log &



```



Copy experiment runs logs from remove to local machine

```bash

rsync -avz -e "ssh -p 19764" --exclude='checkpoints' --exclude='samples' --exclude='samples' root@175.155.64.160:/workspace/rl4llm/runs ./rl4llm


rsync -avz -e "ssh -p 19764"  --exclude='checkpoints' root@175.155.64.160:/workspace/rl4llm/runs ./rl4llm

```
