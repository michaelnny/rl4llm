import torch
from trl import GRPOTrainer
from vllm import LLM

group_size = 4

temperature = torch.linspace(0.1, 1.0, steps=group_size)

print(temperature)
