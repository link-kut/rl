import torch

x = torch.tensor([[1], [2], [3]])
print(x.size())
print(x)
x = x.expand(3, 4)
print(x.size())
print(x)