# Inference Module

## Launch the SGLang Server

Before starting the training job, launch the SGLang inference server with the command. This will start a FastAPI HTTP server in the background, and you should keep it open for the entire training session. Remember to use the same model name/path.

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-1.5B \
    --host localhost \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1
```

> **Note:** Avoid using general IP like `0.0.0.0`, instead using the host name or real IP when launching the script.


### Important for Co-hosting SGLang with Training Models

If you want to co-hosting the inference engine and training models on the same devices, use the `--enable-memory-saver` option, which requires using the torch memory saver package. You can install it with `pip install torch-memory-saver`.

```bash
PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-1.5B \
    --host localhost \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1 \
    --enable-memory-saver
```

> **Note:** During the training step, we can first release the GPU memory allocated by the inference engine. However, it's important to do this in a specific order; otherwise, it might break the inference engine (server crash or random weights).

Here's what an example training loop would look like:

```python
# Training loop
for _ in range(100):
    # 1. Call inference server to generate M samples

    # 2. Call the inference server to release memory

    # 3. Update the policy model

    # 4. Call the inference server to resume memory, then load weights from disk
```
