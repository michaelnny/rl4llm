import torch

group_size = 4

temperature = torch.linspace(0.1, 1.0, steps=group_size)

print(temperature)