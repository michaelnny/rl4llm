"""Simple script to launch an SGLang Server for inference"""

import argparse
import os

from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server


# def parse_args():
#     parser = argparse.ArgumentParser(description="RL GRPO fine-tuning")
#     parser.add_argument(
#         "--model-path",
#         type=str,
#         required=True,
#         help="Model name, or path to the model checkpoint",
#     )
#     parser.add_argument(
#         "--tp-size",
#         type=int,
#         required=False,
#         default=1,
#         help="Tensor parallel size, default 1",
#     )
#     return parser.parse_args()


# args = parse_args()


# server_process, port = launch_server_cmd(
#     "python -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --host 0.0.0.0"
# )

# wait_for_server(f"http://localhost:{port}")


# enable_memory_saver


python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1




python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size 8192 \
    --tp 1 \
    --enable-memory-saver
