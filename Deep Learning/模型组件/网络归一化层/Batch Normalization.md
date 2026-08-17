---
tags: [Deep-Learning, 模型组件, normalization, BN]
aliases:
  - BatchNorm
  - BN
  - 批归一化
related:
  - "[[网络归一化层]]"
---

# Batch Normalization

Batch Normalization（BN，批归一化）利用 mini-batch 的统计量对激活值进行标准化，是 CNN 中最经典的归一化方法之一。

## 计算方式

对于 CNN 输入 $x\in\mathbb R^{N\times C\times H\times W}$，BN 对每个通道分别沿 $(N,H,W)$ 维度计算均值和方差：

$$
\mu_c=\frac{1}{NHW}\sum_{n,h,w}x_{nchw}
$$

$$
\operatorname{BN}(x)=\gamma_c
\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta_c
$$

$\gamma_c$ 和 $\beta_c$ 是每个通道的可学习参数。

## 训练与推理

| 阶段 | 统计量来源 |
|---|---|
| 训练 | 当前 mini-batch 的均值和方差 |
| 推理 | 训练期间累计的 running mean 和 running variance |

`running_mean`、`running_var` 和 `num_batches_tracked` 是 buffer，不由优化器直接更新。验证和推理前必须调用 `model.eval()`；恢复训练后需要调用 `model.train()`。

## 优点

- batch 统计量稳定时，通常能改善 CNN 的优化和收敛；
- 降低对参数初始化和学习率的敏感性；
- mini-batch 统计噪声可能产生轻微正则化效果；
- 在卷积网络中有成熟、高效的硬件实现。

## 局限

- 对每张设备上的实际 batch size 敏感，小 batch 时统计量噪声较大；
- 训练和推理行为不一致；
- 分布式训练中各设备的局部统计量可能不同；
- 序列长度或数据分布变化较大时使用不便。

## PyTorch 用法

```python
import torch.nn as nn

# CNN，输入一般为 [N, C, H, W]
bn2d = nn.BatchNorm2d(num_features=64)

# 一维特征或时序卷积常用
bn1d = nn.BatchNorm1d(num_features=256)
```

常见卷积块：

```python
block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
)
```

卷积后紧跟 BN 时，卷积层的 bias 往往可以关闭，因为 BN 自己包含可学习偏移量。

## 小 batch 与分布式训练

- **GroupNorm**：完全不依赖 batch，适合检测和分割中的小 batch；
- **SyncBatchNorm**：跨 GPU 同步统计量，增加有效 batch，但会引入通信开销；
- **冻结 BN**：保持已有 running statistics，不再更新，常见于预训练检测模型微调；
- **梯度累积不能增大 BN 的统计 batch**：每个 micro-batch 的 BN 仍独立计算统计量。

## 与 EMA 的关系

使用 [[EMA模型]] 时，除可训练参数外，还必须决定如何处理 BN buffer。常见做法是从在线模型直接复制 running statistics，或训练结束后重新校准统计量。不同代码库的实现可能不同。

## 补充理解

BN 最初由“缓解内部协变量偏移”解释。后续研究认为，它的收益还与重参数化、平滑优化景观和改善梯度传播有关，因此不宜只用内部协变量偏移概括全部机制。

