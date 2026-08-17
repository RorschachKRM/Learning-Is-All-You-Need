---
tags: [Deep-Learning, 模型组件, normalization, RMSNorm, LLM]
aliases:
  - Root Mean Square Layer Normalization
  - 均方根归一化
related:
  - "[[网络归一化层]]"
  - "[[Layer Normalization]]"
---

# RMSNorm

RMSNorm（Root Mean Square Layer Normalization）是一种类似 LayerNorm 的特征缩放方法。它不减去均值，而是使用输入向量的均方根调整尺度。

## 计算方式

对于 $x\in\mathbb R^d$：

$$
\operatorname{RMS}(x)=
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

$$
\operatorname{RMSNorm}(x)=
\frac{x}{\operatorname{RMS}(x)}\odot\gamma
$$

标准 RMSNorm 通常只包含可学习缩放参数 $\gamma$，不包含 LayerNorm 中的中心化步骤；具体库也可能提供额外 bias 选项。

## 与 LayerNorm 的区别

| 项目 | RMSNorm | LayerNorm |
|---|---|---|
| 减均值 | 否 | 是 |
| 缩放依据 | 均方根 | 标准差 |
| 可学习 scale | 有 | 有 |
| 可学习 bias | 常无 | 通常有 |
| 平移不变性 | 无 | 有 |
| 缩放不变性 | 有 | 有 |

## 核心特性

- 计算形式更简单，省略均值中心化；
- 不依赖 batch，训练和推理行为一致；
- 广泛用于 LLaMA、Mistral 等现代大语言模型；
- 实际速度收益取决于融合内核、硬件、张量大小和框架实现；
- 是否优于 LN 取决于模型架构和训练配方，不是通用结论。

## PyTorch 用法

较新版本的 PyTorch 提供：

```python
import torch.nn as nn

rms_norm = nn.RMSNorm(normalized_shape=4096, eps=1e-6)
```

使用旧版 PyTorch 或特定大模型框架时，可能使用模型库自带的 RMSNorm 实现。加载预训练权重时应保持 `eps`、参数名和 bias 配置一致。

## 常见位置

RMSNorm 常被放置在 Transformer 的注意力模块和前馈网络之前，即 Pre-Norm 结构：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

RMSNorm 解决的是激活尺度问题；它与 RoPE、SwiGLU、注意力机制等组件承担不同职责。

