import torch


a = torch.rand((200, 200),device="cuda:1")
b = torch.rand((200, 200),device="cuda:1")

for i in range(1_000_000_000):
	a * b
