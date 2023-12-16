import numpy as np
import torch

A = torch.tensor([
    [0, 1, 0],
    [1, 2, 1],
    [0, 1, 0],
])

print(torch.gradient(A))