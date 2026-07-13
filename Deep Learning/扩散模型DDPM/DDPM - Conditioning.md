---
title: 条件注入机制
tags:
  - DDPM
  - Condition
  - Embedding
related:
  - "[[DDPM - General Framework#6. 模块四：条件注入机制（Conditioning）]]"
---
# 类别one-hot编码
```python
# label one-hot template
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(
	1, 
	torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 
	1
)
```

```python
# 类别one-hot编码
class_onehot = onehot[labels].to(device)  # (B,10)
```
- `onehot` 是一个预定义的 `(10, 10)` 单位矩阵（或 one-hot 查找表）。



# 方位角傅里叶编码
**单个角度值映射到高维连续空间**，让网络能学到角度的周期性特征。

傅里叶编码核心思想：把一个角度 `θ` 映射为一组不同频率的 `(cos, sin)` 对：
```
θ  →  [cos(1θ), sin(1θ), cos(2θ), sin(2θ), cos(3θ), sin(3θ), cos(4θ), sin(4θ), cos(5θ), sin(5θ)]
```
共 5 个频率 × 2(cos+sin) = 10 维。

### 为什么用多个频率？
不同频率捕捉不同尺度的方向敏感性：

|频率|周期|捕捉的特征|
|---|---|---|
|1×|360° 一圈|整体方位趋势（朝东 vs 朝西）|
|2×|180° 半圈|前后对称特征（SAR 侧视的左右模糊）|
|3×|120°|中等细节，如车辆侧面结构|
|4×|90°|更细的局部散射变化|
|5×|72°|精细的角闪烁（speckle 随角度快速变化）|
这和 **SAR 物理本质**匹配：雷达回波强度对观测角的依赖包含多种频率成分——镜面反射变化慢（低频），边缘绕射变化快（高频）。


```python
# 方位角傅里叶编码 (5阶, 10维)

# 1. 角度转弧度：PyTorch 三角函数接受弧度
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
最终 `real_az_vec` 是 `(B, 10)` 的张量，作为一个条件向量、和类别 one-hot (`B, 10`) 一起拼成条件信号送入 UNet 指导扩散过程。



