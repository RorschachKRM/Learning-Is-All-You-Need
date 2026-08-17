---
tags: [Deep-Learning, 模型组件, normalization, SpectralNorm, GAN]
aliases:
  - SpectralNorm
  - SN
  - 谱归一化
related:
  - "[[网络归一化层]]"
  - "[[Weight Normalization]]"
---

# Spectral Normalization

Spectral Normalization（SN，谱归一化）通过权重矩阵的最大奇异值对权重进行缩放，以限制线性层或卷积层的放大能力。

## 计算方式

设权重矩阵为 $W$，其谱范数为最大奇异值 $\sigma(W)$：

$$
\bar W=\frac{W}{\sigma(W)}
$$

若需要指定其他尺度，也可以乘上目标系数。实际训练中通常通过幂迭代近似最大奇异值，而不会每一步完整进行奇异值分解。

## 主要作用

- 控制网络层的 Lipschitz 常数上界；
- 限制判别器权重快速放大；
- 提高 GAN 训练稳定性；
- 不依赖 batch 统计量。

SN 主要约束权重变换，而不是把激活值调整为零均值、单位方差，因此与 BN/LN 的目标不同。

## PyTorch 用法

```python
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

discriminator_layer = spectral_norm(
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
)
```

卷积核在内部会被视作适当展开后的矩阵。幂迭代次数越多，谱范数估计可能越准确，但计算成本也会增加。

## 与 WeightNorm 的区别

| 项目 | SpectralNorm | WeightNorm |
|---|---|---|
| 约束对象 | 整个权重矩阵的最大奇异值 | 单个权重向量的方向与长度 |
| 主要目的 | Lipschitz 控制、稳定训练 | 改善参数优化 |
| 常见领域 | GAN 判别器 | RNN、生成模型等 |
| 是否依赖 batch | 否 | 否 |

## 注意事项

- SN 会增加每次前向中的幂迭代计算；
- 它不能保证整个深度网络严格具有指定 Lipschitz 常数，整体还取决于激活函数、残差结构和所有层；
- 保存、加载或导出模型时，需要注意参数化状态；
- 不应仅因名称中含“Normalization”就把它视作 BatchNorm 的直接替代品。

