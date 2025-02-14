import numpy as np
import torch

group_size = 8
max_temp = 0.7

temperature = torch.linspace(0.0, max_temp, steps=group_size)

print(temperature)


temperature = torch.pow(torch.linspace(0.0, 1.0, steps=group_size), 2) * max_temp

print(temperature)
