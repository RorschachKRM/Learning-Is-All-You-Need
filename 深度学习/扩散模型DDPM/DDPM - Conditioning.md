---
title: 条件注入机制
tags:
  - DDPM
  - Condition
  - Embedding
related:
  - "[[DDPM - General Framework#6. 模块四：条件注入机制（Conditioning）]]"
---
# one-hot
```python
# 类别one-hot编码
class_onehot = onehot[labels].to(device)  # (B,10)
```
- `onehot` 是一个预定义的 `(10, 10)` 单位矩阵（或 one-hot 查找表）。



# 方位角傅里叶编码
```python
# 方位角傅里叶编码 (5阶, 10维)
real_angle = torch.deg2rad(az)
B = x0.shape[0]
real_az_vec = torch.zeros(B, 10, device=device)
for i in range(B):
    real_az_vec[i, 0] = torch.cos(real_angle[i])
    real_az_vec[i, 1] = torch.sin(real_angle[i])
    real_az_vec[i, 2] = torch.cos(2 * real_angle[i])
    real_az_vec[i, 3] = torch.sin(2 * real_angle[i])
    real_az_vec[i, 4] = torch.cos(3 * real_angle[i])
    real_az_vec[i, 5] = torch.sin(3 * real_angle[i])
    real_az_vec[i, 6] = torch.cos(4 * real_angle[i])
    real_az_vec[i, 7] = torch.sin(4 * real_angle[i])
    real_az_vec[i, 8] = torch.cos(5 * real_angle[i])
    real_az_vec[i, 9] = torch.sin(5 * real_angle[i])
```
- **谐波编码（Fourier features / positional encoding）**：将单个角度值 $\theta$ 映射为 10 维向量： $$[\cos\theta, \sin\theta, \cos 2\theta, \sin 2\theta, ..., \cos 5\theta, \sin 5\theta]$$
- 共 5 个谐波阶数，每阶 cos + sin 共 10 维
- 例如方位角 60° → rad(60) ≈ 1.047 → `[cos(1.047), sin(1.047), cos(2.094), sin(2.094), ...]`
- **为什么这样做？** 
	- 角度是周期性的（$0°$ 和 $360°$ 应该是同一个方向），直接输入角度值无法表达这种周期性。使用 sin/cos 编码可以自然地捕捉角度的周期特性，类似 Transformer 中的位置编码。
- **性能注意**：这里用了一个 Python for 循环逐样本处理，batch 大时会有性能瓶颈。可以改为向量化操作



