---
tags: [Deep-Learning, 模型组件, normalization, LRN, CNN]
aliases:
  - LRN
  - Local Response Norm
  - 局部响应归一化
related:
  - "[[网络归一化层]]"
---

# Local Response Normalization

Local Response Normalization（LRN，局部响应归一化）是一种在相邻通道之间制造竞争的早期神经网络组件，因 AlexNet 而知名。

## 核心思想

LRN 借鉴生物神经系统中的侧抑制：某个位置上响应较强的通道，会抑制相邻通道的响应，使局部突出激活更加显著。

一种常见形式为：

$$
b^i_{x,y}=a^i_{x,y}
\left(k+\alpha
\sum_{j=\max(0,i-n/2)}^{\min(C-1,i+n/2)}
(a^j_{x,y})^2\right)^{-\beta}
$$

- $i$：当前通道；
- $n$：参与竞争的相邻通道数量；
- $k,\alpha,\beta$：控制缩放强度的超参数。

## PyTorch 用法

```python
import torch.nn as nn

lrn = nn.LocalResponseNorm(
    size=5,
    alpha=1e-4,
    beta=0.75,
    k=1.0,
)
```

## 历史地位与局限

- AlexNet 使用 LRN 强化相邻通道间的竞争；
- 它不是按 batch 或特征维度进行零均值、单位方差标准化；
- 计算开销较大，现代架构中收益通常有限；
- 随着 BatchNorm、LayerNorm 等方法出现，LRN 已很少作为新模型的默认选择。

LRN 现在主要用于理解经典 CNN 的发展历史或复现早期架构，而不是现代网络的常规归一化方案。

