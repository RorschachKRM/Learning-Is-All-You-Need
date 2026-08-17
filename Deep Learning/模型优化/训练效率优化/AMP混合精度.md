---
title: Automatic Mixed Precision，自动混合精度
tags:
  - AMP
  - PyTorch
  - Model-Optimization
---

# AMP 混合精度

## 一句话理解

**AMP（Automatic Mixed Precision，自动混合精度）**是一种训练与推理优化技术：让适合低精度的运算使用 `FP16` 或 `BF16`，让对数值精度敏感的运算继续使用 `FP32`，从而在尽量保持模型精度和训练稳定性的前提下，减少显存占用并提高计算速度。

这里的“混合”是指：**同一个模型的一次前向、反向计算中，会同时使用多种数值精度**，而不是简单地把模型中所有数据都强制转换成半精度。

**AMP 是训练工程/计算性能优化技术**，一般不会主动提升模型精度，重点是“训练得更快、更省显存”，主要是优化计算效率和显存使用，加快训练/推理、降低显存占用。

不属于模型结构优化，不会像剪枝、量化、知识蒸馏那样直接改变模型结构或部署形态。

---

## 1. 什么是数值精度？

深度学习中的权重、激活值和梯度通常用浮点数表示。常见格式如下：

| 格式 | 位数 | 大致特点 | 常见用途 |
|---|---:|---|---|
| `FP32` | 32 bit | 精度高、数值范围较大、稳定性好，但占显存较多 | 传统训练、敏感运算 |
| `FP16` | 16 bit | 占用小、速度快，但数值范围较窄，容易上溢或下溢 | GPU 混合精度训练 |
| `BF16` | 16 bit | 与 FP32 指数位相同，数值范围大，但尾数精度较低 | 新一代 GPU/TPU 训练 |

一个数值格式通常包含三部分：符号位、指数位和尾数位。

- **指数位**主要决定能表示多大或多小的数，即“数值范围”。
- **尾数位**主要决定数值能表示得多细，即“精确程度”。

`FP16` 和 `BF16` 都只占 2 字节，但二者取舍不同：

- `FP16`：尾数更精细，但指数范围较小，训练时更容易出现梯度下溢，因此通常需要梯度缩放。
- `BF16`：指数范围与 `FP32` 接近，训练更稳定，通常不需要梯度缩放；代价是单个数的有效精度比 `FP16` 略低。

> [!note]
> BF16 是否更快取决于硬件是否原生支持。例如 NVIDIA Ampere 及更新架构通常支持得较好。

---

## 2. 为什么不把所有运算都改成 FP16？

纯 FP16 训练可能遇到数值稳定性问题。

### 2.1 下溢（underflow）

反向传播中的梯度经常非常小。如果小到超出 FP16 能表示的范围，就可能被舍入成 0。大量梯度变成 0 后，模型就无法正常更新。

### 2.2 上溢（overflow）

如果数值超过 FP16 能表示的最大值，就会变成 `inf`，后续计算还可能产生 `NaN`。

### 2.3 舍入误差

某些运算需要累加大量小数，例如归一化、归约或损失计算。使用过低精度可能让误差不断累积。

因此，AMP 会根据运算类型自动选择更合适的精度：

- 矩阵乘法、卷积等计算量大且通常适合低精度的运算，优先使用 FP16/BF16；
- 某些归约、指数、对数、归一化及损失相关运算，在需要时使用 FP32；
- 优化器通常保留 FP32 权重或状态，以避免参数更新中的微小变化丢失。

可以把 AMP 理解成一位“精度调度员”：能安全加速的地方用低精度，对数值敏感的地方用高精度。

---

## 3. AMP 有什么用？

### 3.1 减少显存占用

FP16/BF16 每个元素占 2 字节，FP32 每个元素占 4 字节。激活值等中间结果使用半精度后，通常可以显著减少显存占用。

但显存占用**不会简单地固定减半**，因为：

- 某些张量仍会以 FP32 保存；
- 优化器状态可能仍为 FP32；
- 显存还包括 CUDA 上下文、临时工作区和缓存等。

节省出的显存可以用于更大的 batch size、更高分辨率、更长序列或更大的模型。

### 3.2 加快训练和推理

现代 GPU 的 Tensor Core 对 FP16/BF16 矩阵运算具有很高吞吐量。模型若主要由卷积、矩阵乘法等运算组成，AMP 往往能明显加速。

实际加速比不是固定值，取决于：

- GPU 型号及其 Tensor Core 支持；
- 模型中矩阵乘法、卷积所占比例；
- batch size 和张量维度；
- 数据加载、CPU 或通信是否成为瓶颈；
- 是否存在大量不适合半精度的算子。

### 3.3 通常能保持接近 FP32 的模型效果

AMP 并非盲目降低所有计算精度，而是保留关键运算的高精度，并配合梯度缩放等机制。因此，多数常规模型能获得与 FP32 训练接近的收敛结果。

不过，“通常接近”不等于绝对一致。浮点计算顺序和舍入方式发生变化后，即使随机种子相同，训练曲线也可能出现细微差异。

---

## 4. AMP 的两个核心机制

### 4.1 自动类型转换（autocast）

`autocast` 会在指定代码区域内，根据算子的安全性和性能特点自动决定采用低精度还是 FP32。

它只影响运算时采用的精度，不意味着永久把整个模型转换为半精度。因此，使用 AMP 时一般不要再手动对模型调用 `model.half()`。

### 4.2 梯度缩放（gradient scaling）

梯度缩放主要用于缓解 FP16 梯度下溢。

设原始损失为 $L$，AMP 在反向传播前先乘上一个较大的缩放因子 $S$：

$$
L' = S \cdot L
$$

由于梯度也会相应放大：

$$
\frac{\partial L'}{\partial w} = S \cdot \frac{\partial L}{\partial w}
$$

原本小到可能在 FP16 中变成 0 的梯度，就更容易被保留下来。在优化器更新参数前，再把梯度除以 $S$，恢复到正确尺度。

动态 `GradScaler` 还会自动调整 $S$：

- 如果连续若干步没有出现 `inf/NaN`，可逐渐增大缩放因子；
- 如果检测到梯度溢出，则跳过本次参数更新并减小缩放因子。

这不会改变理论上的梯度，只是暂时把梯度搬到 FP16 更容易表示的数值区间。

> [!tip]
> BF16 的指数范围与 FP32 接近，通常不需要 `GradScaler`；FP16 训练一般建议使用它。

---

## 5. PyTorch 中如何使用 AMP？

### 5.1 FP16 训练示例

```python
import torch

device = "cuda"
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# FP16 通常需要梯度缩放
scaler = torch.amp.GradScaler("cuda")

model.train()
for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)

    # autocast 主要包住前向传播和损失计算
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # 先对放大后的 loss 做反向传播
    scaler.scale(loss).backward()

    # scaler 会在更新前还原梯度，并在溢出时跳过更新
    scaler.step(optimizer)
    scaler.update()
```

标准顺序是：

```text
清空梯度
  → autocast 前向传播与 loss
  → scaler.scale(loss).backward()
  → scaler.step(optimizer)
  → scaler.update()
```

### 5.2 BF16 训练示例

硬件支持 BF16 时，可以写成：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

loss.backward()
optimizer.step()
```

BF16 通常不需要梯度缩放，但模型和硬件组合不同，仍应实际验证训练是否稳定。

### 5.3 推理示例

推理不需要反向传播，也不需要 `GradScaler`：

```python
model.eval()
with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        predictions = model(inputs)
```

---

## 6. 梯度裁剪时要特别注意

如果要裁剪梯度，必须先使用 `scaler.unscale_()` 恢复梯度的真实尺度，否则裁剪的是被放大后的梯度。

```python
scaler.scale(loss).backward()

scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

正确顺序是：**反向传播 → 还原梯度 → 梯度裁剪 → 参数更新**。

---

## 7. FP16 和 BF16 应该怎么选？

| 情况 | 建议 |
|---|---|
| 较老的、支持 Tensor Core 的 NVIDIA GPU | 常用 FP16 + GradScaler |
| Ampere 或更新架构，且框架和算子支持良好 | 优先尝试 BF16，也可实测对比 FP16 |
| 训练频繁出现梯度溢出或不稳定 | 尝试 BF16，或定位敏感算子并保留 FP32 |
| 只做推理 | 对比 FP16/BF16 的速度、显存和输出误差后选择 |
| CPU 训练或推理 | 是否受益取决于 CPU 的 BF16/FP16 硬件支持，不能照搬 GPU 结论 |

简单概括：

- **FP16**：有效数字精度稍好，但范围小，通常依赖梯度缩放。
- **BF16**：范围大、训练更稳，通常不依赖梯度缩放，但需要硬件良好支持。

---

## 8. 常见误区与注意事项

### 误区 1：AMP 等于把整个模型变成 FP16

不是。AMP 的关键是“按算子混合使用精度”。手动调用 `model.half()` 属于整体半精度转换，行为和风险都不同。

### 误区 2：用了 AMP，显存一定减半、速度一定翻倍

不一定。优化器状态、FP32 参数副本、非 Tensor Core 算子、数据读取及通信瓶颈都会影响最终收益。

### 误区 3：有 GradScaler 就不会出现任何 NaN

`GradScaler` 主要解决 FP16 梯度范围问题。学习率过大、数据异常、损失函数不稳定、非法数学运算等仍然可能产生 `NaN`。

### 误区 4：把反向传播也放进 autocast 上下文会更快

通常只需要让前向传播和损失计算处于 `autocast` 上下文中。反向传播会沿用前向图中相应运算的数据类型，不建议额外把 `backward()` 包进去。

### 其他注意事项

- 自定义 CUDA 算子或自定义 `autograd.Function` 需要单独确认低精度兼容性。
- 某些数值敏感模块可局部关闭 autocast，并显式转为 FP32 计算。
- 多个 loss、多个优化器、梯度累积和分布式训练时，要确保缩放、还原和 `step` 的顺序正确。
- 评估 AMP 是否有效时，应同时观察吞吐量、峰值显存、训练曲线、最终指标和是否出现 `inf/NaN`。
- 首次迭代可能包含 CUDA 初始化、内核选择和缓存开销，性能测试前应预热若干轮，并使用正确的 GPU 同步计时方式。

局部禁用 AMP 的示例：

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    features = model(inputs)

    # 假设这一段对数值精度非常敏感
    with torch.autocast(device_type="cuda", enabled=False):
        stable_result = sensitive_operation(features.float())
```

---

## 9. 如何判断 AMP 是否值得使用？

可以做一次控制变量实验：FP32 和 AMP 使用相同模型、数据、batch size 与训练步数，比较以下指标：

1. 每秒样本数或每步耗时；
2. GPU 峰值显存；
3. loss 曲线是否稳定；
4. 验证集最终指标是否与 FP32 接近；
5. 是否频繁出现溢出、跳过优化器更新或 `NaN`。

如果 AMP 节省显存后允许增大 batch size，比较时需要区分两种收益：

- 相同 batch size 下，AMP 本身带来的算力加速；
- 更大 batch size 提高硬件利用率后带来的额外吞吐提升。

---

## 10. 总结

AMP 的本质不是单纯“降低精度”，而是进行**有选择的精度分配**：

- 用 FP16/BF16 加速卷积和矩阵乘法，并减少中间张量显存；
- 用 FP32 保护数值敏感的运算和参数更新；
- FP16 配合梯度缩放，降低小梯度下溢的风险；
- 在现代 GPU 上，通常能够以很小的精度代价换取更高吞吐量和更低显存占用。

因此，AMP 已经成为现代深度学习训练中常用的基础优化手段，但最终收益和稳定性仍应通过实际基准测试确认。

