


```bash

# Login to remote server
ssh -p 24371 root@115.124.123.238 -L 8080:localhost:8080


# Install CUDA Toolkit and MPICH for deepspeed
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4 libmpich-dev


# On local machine, copy project files to remote server
scp -r -P 24371 ./rl4llm root@115.124.123.238:/project

# update a small files
scp -r -P 24371 ./rl4llm/src/rl4llm/core root@115.124.123.238:/project/src/rl4llm


scp -r -P 24371 ./rl4llm/src/rl4llm/scripts root@115.124.123.238:/project/src/rl4llm


scp -r -P 24371 ./rl4llm/src/rl4llm/envs root@115.124.123.238:/project/src/rl4llm


scp -r -P 24371 ./rl4llm/configs root@115.124.123.238:/project


# Install packages
cd /project

pip install -r requirements.txt

```


Run training script
```bash

cd /project


PYTHONPATH=src deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_ppo.py --config-file ./configs/ppo_train_test_config.yaml


PYTHONPATH=src deepspeed --num_gpus=1 src/rl4llm/scripts/run_train_ppo.py --config-file ./configs/ppo_train_config.yaml


```