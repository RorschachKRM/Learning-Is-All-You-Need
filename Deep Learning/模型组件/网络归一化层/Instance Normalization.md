---
tags: [Deep-Learning, 模型组件, normalization, IN, 图像生成]
aliases:
  - InstanceNorm
  - IN
  - 实例归一化
related:
  - "[[网络归一化层]]"
---

# Instance Normalization

Instance Normalization（IN，实例归一化）对每个样本、每个通道独立地在空间维度上计算统计量，最初因风格迁移任务而受到广泛关注。

## 计算方式

对于 $x\in\mathbb R^{N\times C\times H\times W}$：

$$
\mu_{nc}=\frac{1}{HW}\sum_{h,w}x_{nchw}
$$

$$
\operatorname{IN}(x_{nchw})=
\gamma_c\frac{x_{nchw}-\mu_{nc}}
{\sqrt{\sigma_{nc}^2+\epsilon}}+\beta_c
$$

统计量完全由当前样本的当前通道产生，不混合不同样本或不同通道。

## 核心特性

- 不依赖 batch size；
- 通常不需要训练阶段累计全局 running statistics；
- 会弱化单张图像各通道的整体均值、方差和对比度信息；
- 常用于风格迁移及部分生成模型；
- 对依赖绝对颜色、对比度或强度的分类任务，可能移除有用信息。

## PyTorch 用法

```python
import torch.nn as nn

instance_norm = nn.InstanceNorm2d(
    num_features=64,
    affine=True,
    track_running_stats=False,
)
```

PyTorch 的 `InstanceNorm2d` 默认 `affine=False`，与 BatchNorm 的默认行为不同；是否需要可学习仿射参数应根据架构设置。`track_running_stats=False` 时，训练与评估都使用当前输入的实例统计量。

## 与其他方法的关系

- BN：同一通道跨 batch 统计；
- IN：每个样本、每个通道只在空间维统计；
- GN：每个样本内将多个通道组成一组；
- 当 GN 每组只含一个通道时，其统计范围与 IN 接近，但仍要注意仿射参数和实现配置。

## AdaIN

Adaptive Instance Normalization（AdaIN）先对内容特征做实例归一化，再使用风格特征的均值和标准差重新缩放：

$$
\operatorname{AdaIN}(x_c,x_s)=
\sigma(x_s)\frac{x_c-\mu(x_c)}{\sigma(x_c)}+\mu(x_s)
$$

它通过替换通道统计量注入风格信息，是经典任意风格迁移方法的重要组件。

