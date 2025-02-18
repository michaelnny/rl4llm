import numpy as np
import torch

group_size = 32
max_temp = 0.9

temperature = torch.linspace(0.0, max_temp, steps=group_size)

print(temperature)


temperature = torch.pow(torch.linspace(0.0, 1.0, steps=group_size), 0.5) * max_temp

print(temperature)
