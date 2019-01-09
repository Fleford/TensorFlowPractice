import torch
import numpy as np

t1 = torch.Tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])

t2 = torch.Tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])

t3 = torch.Tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])

t = torch.stack((t1, t2, t3))
print(t.shape)
t = t.reshape(3, 1, 4, 4)
print(t)
print(t.flatten(start_dim=1))
