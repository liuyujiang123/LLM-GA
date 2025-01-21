import torch

x = torch.zeros((3, 448, 448))
y = torch.torch.empty_like(x).uniform_(-0.05, 0.05)
print(torch.linalg.norm(y - x))