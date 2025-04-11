
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



## Start Training

### Option 1: Use SGLang inference server + DeepSpeed

We can use SGLang inference server to speed up the sample generation process. To do so, we need to first launch the SGLang HTTP server. It supports co-host the inference engine and training models on the same server/GPUs. To do so, we only need to use the `--enable-memory-saver` option. Which requires using the torch memory saver package, you can install it with `pip install torch-memory-saver`

```bash

PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-0.5B \
    --host localhost \
    --port 30000 \
    --mem-fraction-static 0.5 \
    --chunked-prefill-size 8192 \
    --tp 1 \
    --enable-memory-saver

```


--enable-custom-logit-processor



Then, we can start the training job by passing in the options to the launch script



```bash

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_grpo.py --config-file ./configs/grpo_config.yaml --use-infer-server --infer-host localhost --infer-port 30000 --infer-cohost-mode

```


### Optional 2: Use DeepSpeed model for both inference and training

```bash

PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 deepspeed --num_gpus=1 scripts/run_train_grpo.py --config-file ./configs/grpo_config.yaml

```


To monitoring the job, open tensorboard

```bash

tensorboard --logdir ./runs --bind_all --samples_per_plugin=text=10000

```
