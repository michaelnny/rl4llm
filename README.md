


```bash

# Login to remote server
ssh -p 46844 root@171.240.133.62 -L 8080:localhost:8080


# Install CUDA Toolkit and MPICH for deepspeed
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 libmpich-dev


# On local machine, copy project files to remote server

rsync -avz -e "ssh -p 46844" --exclude='.*' --exclude='__pycache__/' --exclude='tests' --exclude='runs' ./rl4llm root@171.240.133.62:/project/

# update a small files
rsync -avz -e "ssh -p 46844" ./rl4llm/src/rl4llm/core root@171.240.133.62:/project/rl4llm/src/rl4llm


rsync -avz -e "ssh -p 46844" ./rl4llm/src/rl4llm/scripts root@171.240.133.62:/project/rl4llm/src/rl4llm


rsync -avz -e "ssh -p 46844" ./rl4llm/src/rl4llm/envs root@171.240.133.62:/project/rl4llm/src/rl4llm


rsync -avz -e "ssh -p 46844" ./rl4llm/configs root@171.240.133.62:/project/rl4llm


# Install packages
cd /project/rl4llm

pip install -r requirements.txt

```


Run training script
```bash

cd /project/rl4llm


PYTHONPATH=src deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_sft.py --config-file ./configs/sft_train_config.yaml


nohup sh -c "PYTHONPATH=src deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_sft.py --config-file ./configs/sft_train_config.yaml" &



PYTHONPATH=src deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_ppo.py --config-file ./configs/ppo_train_test_config.yaml



nohup sh -c "PYTHONPATH=src deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_ppo.py --config-file ./configs/ppo_train_config.yaml" &




pkill -f "src/rl4llm/scripts/run_train_ppo.py"

```


To monitoring the job, open tensorboard
```bash

rsync -avz -e "ssh -p 46844" --exclude='.pt' --exclude='checkpoints' root@171.240.133.62:/project/rl4llm/runs ./rl4llm


tensorboard --logdir ./rl4llm/runs --samples_per_plugin=text=1000

```