
```bash

python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1


python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1 \
    --enable-memory-saver

```



```

PYTHONPATH=src python -m rl4llm.inference.launch_sgl_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1 \
    --enable-memory-saver

```



## Important findings

- first release memory
- then resume memory
- then load weights from disk
