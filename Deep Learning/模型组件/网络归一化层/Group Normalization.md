---
tags: [Deep-Learning, 模型组件, normalization, GN]
aliases:
  - GroupNorm
  - GN
  - 组归一化
related:
  - "[[网络归一化层]]"
---

# Group Normalization

Group Normalization（GN，组归一化）将通道划分为若干组，并在每个样本的每组通道及空间维度内计算统计量。它不依赖 batch，常用于小 batch 的视觉任务。

## 计算方式

对于 $x\in\mathbb R^{N\times C\times H\times W}$，将 $C$ 个通道分为 $G$ 组：

$$
\mu_{ng}=\frac{1}{(C/G)HW}
\sum_{c\in\operatorname{group}_g,h,w}x_{nchw}
$$

每个样本的每一组分别计算均值和方差，再进行标准化和仿射变换。

## 核心特性

- 不使用 batch 维统计量，batch 为 1 时也能工作；
- 训练和推理行为一致，不维护 running statistics；
- 适合目标检测、实例分割、语义分割等单设备 batch 较小的任务；
- 需要选择组数，并保证通道数可以被组数整除。

## PyTorch 用法

```python
import torch.nn as nn

group_norm = nn.GroupNorm(
    num_groups=32,
    num_channels=256,
)
```

`num_groups=32` 是常见起点而非固定规则。如果通道较少，可以尝试 8、16 或其他能够整除通道数的组数。

## 特殊情况

- `num_groups = 1`：所有通道属于同一组，统计范围覆盖每个样本的全部通道与空间维；
- `num_groups = num_channels`：每组只有一个通道，统计范围接近 InstanceNorm。

这些特殊情况在统计范围上接近 LN/IN 的某些形式，但仿射参数形状、输入布局和具体实现可能仍不同，不能在所有场景中直接视为完全等价。

## 与 BatchNorm 对比

| 项目 | GroupNorm | BatchNorm |
|---|---|---|
| 依赖 batch | 否 | 是 |
| 小 batch 稳定性 | 通常较好 | 可能较差 |
| running statistics | 无 | 有 |
| 额外选择 | 组数 | momentum、同步策略等 |
| 常见场景 | 检测、分割 | 大 batch CNN 分类 |

当单设备 batch 足够大时，BN 仍可能更高效或取得更好结果；GN 的主要价值是消除对 batch 统计量的依赖。

