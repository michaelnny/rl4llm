

## Launch the SGLang server

Before start the training job, launch the SGLang inference server with command. This will start a FastAPI HTTP server in the background. And you should keep it open for the entire training session. Remember use the same model name/path


```bash

PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-1.5B \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1

```


## Important for Co-host SGLang with training models


If you want to co-host the inference engine and training models on the same devices, use the `--enable-memory-saver`, which requires using the torch memory saver package, you can install it with `pip install torch-memory-saver`

```bash

PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-1.5B \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1 \
    --enable-memory-saver

```


During training step, we can first release the GPU memory allocated by the inference engine. But it's important to do this in a specific order, otherwise, it might break the inference engine (server crash, or random weights).

Here's what a example training loop would look like this:

```python

# training loop
for _ in range(100):
    # 1. Call inference server to generate M samples

    # 2. Call the inference server to release memory

    # 3. Update the policy model

    # 4. Call the inference server to resume memory, then load weights from disk

```


## For multi-server/nodes distributed settings

We use local file based method to update the weights for the SGLang inference engine. This means if you are using multiple servers/nodes, you have to make sure the inference server and the master node/rank have access to the file system where we store the checkpoint files.

This path is defined under the job config file under `logging/output_dir`, where we will save the weights under the subfolder `checkpoints`.
