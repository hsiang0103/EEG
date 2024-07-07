import torch

# 假設你有一個形狀為 [100, 2] 的張量
x = torch.randn(100, 2)

# 使用 torch.max 找出每一行的最大值的索引
_, indices = torch.max(x, dim=1)

# 現在，indices 是一個形狀為 [100] 的張量，包含每一行的最大值的索引
print(indices)  # 輸出：torch.Size([100])

print(x)