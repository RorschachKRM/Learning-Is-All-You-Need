---
tags: [Deep-Learning, 模型组件, normalization, WeightNorm]
aliases:
  - WeightNorm
  - WN
  - 权重归一化
related:
  - "[[网络归一化层]]"
  - "[[Spectral Normalization]]"
---

# Weight Normalization

Weight Normalization（WN，权重归一化）不归一化输入激活，而是把权重向量重新参数化为方向和长度，以解耦二者的优化。

## 计算方式

$$
\mathbf w=g\frac{\mathbf v}{\lVert\mathbf v\rVert}
$$

- $\mathbf v$：决定权重方向；
- $g$：决定权重长度；
- $\mathbf v$ 和 $g$ 都是可学习参数。

前向传播时根据 $g$ 和 $\mathbf v$ 计算实际权重 $\mathbf w$，优化器更新的是重参数化后的可学习变量。

## 作用与特点

- 将权重的方向与尺度解耦，可能改善优化条件；
- 不依赖 batch 统计量，训练和推理逻辑一致；
- 不像 BN 那样对激活进行标准化；
- 不维护 running mean 或 running variance；
- 曾用于 RNN、生成模型、强化学习和部分卷积架构。

## PyTorch 用法

现代 parametrization API：

```python
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

linear = weight_norm(nn.Linear(512, 512), name="weight")
```

移除参数化：

```python
from torch.nn.utils.parametrize import remove_parametrizations

remove_parametrizations(linear, "weight", leave_parametrized=True)
```

不同 PyTorch 版本可能同时存在旧的 `torch.nn.utils.weight_norm` 接口。保存和加载权重时，应确保创建模型时使用相同的参数化方式。

## 与 SpectralNorm 的区别

WeightNorm 把权重写成“长度 × 单位方向”，主要是一种优化重参数化；[[Spectral Normalization]] 则用最大奇异值约束整个线性变换的放大能力，常用于 Lipschitz 控制和 GAN 稳定性。两者名称相似，但目标并不相同。

## 注意事项

- WN 不保证激活值具有固定均值或方差；
- 它不能简单替代 BN/LN 的全部作用；
- 导出部署模型前，可根据后端要求将参数化合并回普通权重；
- 与预训练权重或编译工具配合时，要确认 state dict 的参数命名。

