import torch
from torch.utils.benchmark import Timer

x = torch.randn(1_000_000, device="cuda")

print(Timer("x.square()", globals={"x": x}).blocked_autorange())
print(Timer("x ** 3", globals={"x": x}).blocked_autorange())