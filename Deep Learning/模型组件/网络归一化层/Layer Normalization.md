---
tags: [Deep-Learning, 模型组件, normalization, LN, Transformer]
aliases:
  - LayerNorm
  - LN
  - 层归一化
related:
  - "[[网络归一化层]]"
  - "[[RMSNorm]]"
---

# Layer Normalization

Layer Normalization（LN，层归一化）对每个样本独立地在指定特征维度上计算统计量，不依赖同一 batch 中的其他样本。

## 计算方式

设一个 token 的隐藏向量为 $x\in\mathbb R^d$：

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,
\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\operatorname{LN}(x)=\gamma\odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

对于 Transformer 中 `[batch, sequence, hidden]` 形状的张量，通常沿最后一个 `hidden` 维度归一化，每个 token 独立计算统计量。

## 核心特性

- 不依赖 batch size，batch 为 1 时也能正常工作；
- 训练和推理的计算方式一致，不维护 running statistics；
- 广泛用于 Transformer、RNN 和大语言模型；
- 统计维度由 `normalized_shape` 决定，而不是固定等于“除 batch 外的所有维度”。

## PyTorch 用法

```python
import torch.nn as nn

# 输入形状 [batch, sequence, hidden_size]
layer_norm = nn.LayerNorm(normalized_shape=768)
```

`normalized_shape=768` 表示对输入的最后一个维度进行归一化。若传入元组，则对最后多个维度一起归一化。

## Pre-LN 与 Post-LN

以残差子层 $F$ 为例：

```text
Post-LN: y = LN(x + F(x))
Pre-LN:  y = x + F(LN(x))
```

| 形式 | 特点 |
|---|---|
| Post-LN | 原始 Transformer 使用；深层训练可能更依赖初始化与学习率策略 |
| Pre-LN | 梯度通常更容易沿残差路径传播；现代深层 Transformer 中常见 |

Pre-LN 并不保证所有模型都不需要 warmup，也不一定在所有任务上优于 Post-LN。最终还取决于残差缩放、初始化、模型深度和训练配方。

## 与 BatchNorm 的区别

| 项目 | LayerNorm | BatchNorm |
|---|---|---|
| 统计范围 | 单个样本的特征维 | 跨 batch 的同一通道 |
| 依赖 batch | 否 | 是 |
| running statistics | 无 | 有 |
| 训练/推理行为 | 一致 | 不同 |
| 常见领域 | Transformer、RNN | CNN |

## 常见误区

- “Layer”并不意味着自动对整层张量的所有非 batch 维归一化，实际范围由实现参数指定；
- LN 不会使用 batch 内其他样本的信息；
- LN 与输入数据的 Z-Score 标准化所处位置和作用不同，通常不能相互替代。

