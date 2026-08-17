---
tags:
  - EMA
  - Model-Optimization
  - PyTorch
title: Exponential Moving Average，指数移动平均模型，指数移动平均模型
---

# EMA 模型

## 一句话理解

深度学习中的 **EMA 模型（Exponential Moving Average Model，指数移动平均模型）**，通常是指：训练过程中除了正常更新的在线模型（online model）外，再维护一份模型参数的指数移动平均副本。

在线模型正常参与前向传播、反向传播和优化器更新；EMA 模型不参与反向传播，只在每次训练更新后，用在线模型的最新参数缓慢更新自己。训练完成后，通常使用 EMA 模型进行验证、推理或保存。

它的核心作用是**平滑训练过程中不断抖动的模型参数，使最终模型通常更稳定、泛化效果更好**。

**EMA 是训练策略/参数平均技术**，通常不会降低单次推理成本，重点是“获得更稳定、可能效果更好的最终权重”，主要优化参数稳定性和泛化表现，平滑参数波动，使验证或推理结果更稳定。

不属于模型结构优化，不会像剪枝、量化、知识蒸馏那样直接改变模型结构或部署形态。


---

## 1. EMA 的数学原理

设第 $t$ 步训练结束后，在线模型参数为 $\theta_t$，EMA 模型参数为 $\bar{\theta}_t$，更新公式为：

$$
\bar{\theta}_t = \beta \bar{\theta}_{t-1} + (1-\beta)\theta_t
$$

其中：

- $\theta_t$：在线模型当前参数；
- $\bar{\theta}_{t-1}$：上一步的 EMA 参数；
- $\beta$：衰减系数（decay），通常非常接近 1，例如 `0.99`、`0.999`、`0.9999`；
- $1-\beta$：当前在线模型参数在本次 EMA 更新中的权重。

例如 $\beta=0.999$ 时：

$$
\bar{\theta}_t = 0.999\bar{\theta}_{t-1} + 0.001\theta_t
$$

这表示 EMA 模型保留 99.9% 的历史平滑结果，只吸收 0.1% 的最新参数，因此不会随单个训练 step 剧烈变化。

把递推公式展开，可以看到 EMA 参数实际上是许多历史参数的加权平均：

$$
\bar{\theta}_t
= (1-\beta)\theta_t
+ \beta(1-\beta)\theta_{t-1}
+ \beta^2(1-\beta)\theta_{t-2}
+ \cdots
$$

越新的参数权重越大，越旧的参数权重按指数形式逐渐减小，这正是“指数移动平均”名称的来源。

---

## 2. 为什么参数平均可能比最后一步参数更好？

神经网络训练并不是平滑地走向一个固定最优点。受 mini-batch 抽样、数据增强、Dropout、较大学习率等影响，每一步梯度都带有噪声，模型参数会在较优区域附近不断震荡。

如果直接使用最后一步的在线模型，相当于从这段震荡轨迹中随机取最后一个点；这个点可能恰好偏离了较稳定的中心区域。

EMA 将一段时间内的参数轨迹平滑起来，通常能带来以下效果：

- 降低单个 mini-batch 或单次梯度更新带来的噪声；
- 减少参数在局部较优区域附近的震荡；
- 使验证指标和推理输出更稳定；
- 在许多任务上获得略好的泛化性能；
- 在训练后期或生成模型中，显著改善输出质量和一致性。

可以将在线模型想象成一位快速移动、不断试探方向的探索者，EMA 模型则像一位缓慢跟随、记录总体趋势的观察者。

> [!important]
> EMA 不是一种新的优化器，也不会替代 Adam、AdamW 或 SGD。优化器负责更新在线模型，EMA 只对更新后的参数做额外平滑。

---

## 3. 在线模型和 EMA 模型如何分工？

典型训练过程如下：

```text
在线模型前向传播
    ↓
计算 loss
    ↓
反向传播
    ↓
optimizer.step() 更新在线模型
    ↓
用在线模型的新参数更新 EMA 模型
    ↓
进入下一次迭代
```

两份模型的职责如下：

| 项目 | 在线模型 | EMA 模型 |
|---|---|---|
| 参与训练前向传播 | 是 | 通常否 |
| 参与反向传播 | 是 | 否 |
| 由优化器直接更新 | 是 | 否 |
| 由 EMA 公式更新 | 否 | 是 |
| 需要保存梯度 | 是 | 否 |
| 验证与最终推理 | 可用 | 通常优先使用 |

EMA 模型的参数应设置为不需要梯度，更新过程也应放在 `torch.no_grad()` 环境中，以避免建立无用的计算图。

---

## 4. decay 参数应该如何理解？

$\beta$ 越大，EMA 模型更新越慢、平滑程度越强；$\beta$ 越小，EMA 模型越快跟随在线模型。

| decay | 特点 | 可能的问题 |
|---:|---|---|
| `0.9` | 跟随很快，主要反映近期参数 | 平滑作用较弱 |
| `0.99` | 中等平滑 | 适合训练步数不太长的实验 |
| `0.999` | 常见设置，平滑较强 | 前期跟随相对慢 |
| `0.9999` | 平滑很强，常见于长训练或生成模型 | 训练太短时可能明显滞后 |

EMA 的大致“有效窗口长度”可粗略理解为：

$$
N_{\text{effective}} \approx \frac{1}{1-\beta}
$$

例如：

- $\beta=0.99$，有效窗口约为 100 步；
- $\beta=0.999$，有效窗口约为 1000 步；
- $\beta=0.9999$，有效窗口约为 10000 步。

这只是帮助建立直觉的近似值，不代表 EMA 只考虑固定数量的历史步骤。理论上所有历史参数都有贡献，只是很早以前的权重已经极小。

### decay 不是越大越好

选择 decay 时需要结合总训练步数和参数更新频率：

- 训练很短却使用极大的 decay，EMA 模型可能一直没有充分跟上在线模型；
- 训练很长、每步噪声较大时，较大的 decay 通常更有价值；
- 如果每隔若干步才更新一次 EMA，应根据实际 EMA 更新次数重新考虑 decay；
- 梯度累积时，通常在真正执行 `optimizer.step()` 后才更新一次 EMA。

---

## 5. 初始化、预热与早期偏差

最常见的做法是：训练开始时，直接把在线模型完整复制为 EMA 模型：

$$
\bar{\theta}_0 = \theta_0
$$

这样简单直观，也避免 EMA 初值为 0 带来的明显偏差。

不过，训练初期参数变化很快，固定使用很大的 decay 可能让 EMA 模型过度保留随机初始化状态。常见解决办法包括：

1. **延迟启动**：训练若干步后，再创建或启用 EMA；
2. **decay 预热**：早期使用较小 decay，随后逐渐增大到目标值；
3. **定期直接同步**：预热阶段让 EMA 参数更快接近在线模型。

是否需要预热取决于训练长度、目标 decay 和任务特性。较长训练中，初始化的影响通常会逐渐衰减。

---

## 6. PyTorch 基础实现

下面是一种容易理解的完整模型副本实现：

```python
from copy import deepcopy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()

        # EMA 模型不参与反向传播
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_parameters = dict(self.ema_model.named_parameters())
        model_parameters = dict(model.named_parameters())

        for name, ema_parameter in ema_parameters.items():
            model_parameter = model_parameters[name].detach()
            ema_parameter.mul_(self.decay).add_(
                model_parameter,
                alpha=1.0 - self.decay,
            )

        # 对 BatchNorm 等模块的 buffer，这里采用直接复制策略
        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(model_buffers[name].detach())
```

训练时的调用方式：

```python
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
ema = ModelEMA(model, decay=0.999)

for inputs, targets in train_loader:
    inputs = inputs.cuda()
    targets = targets.cuda()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()

    # 必须在在线模型完成参数更新后再更新 EMA
    ema.update(model)
```

验证或推理时使用 EMA 副本：

```python
ema.ema_model.eval()
with torch.inference_mode():
    predictions = ema.ema_model(inputs)
```

### 更节省显存的实现

完整复制模型会额外占用一份参数显存。大模型训练中，也可以只维护一份 EMA 参数字典，并将其保存在 CPU 上；验证时再临时加载到模型中。这样可以节省 GPU 显存，但每步把参数传到 CPU 会增加通信开销。

因此需要在以下方案间权衡：

- GPU 上维护完整 EMA 副本：更新快，但占用更多显存；
- CPU 上维护 EMA 参数：节省显存，但更新可能更慢；
- 每隔若干步更新一次 EMA：减少更新成本，但 decay 要按更新频率调整。

---

## 7. BatchNorm 的 buffer 应该怎样处理？

模型中除了可训练参数，还有一些 buffer，例如 BatchNorm 的：

- `running_mean`；
- `running_var`；
- `num_batches_tracked`。

这些 buffer 并不由优化器更新，因此需要明确处理策略：

1. **直接从在线模型复制**：实现简单，适合许多场景；
2. **对浮点 buffer 也做 EMA**：部分实现会这样处理，但计数型 buffer 不能做浮点平均；
3. **训练结束后重新校准 BatchNorm 统计量**：用训练数据额外跑一遍前向传播，重新统计均值与方差。

不同代码库的 EMA 实现可能在这里行为不同。复现实验或加载第三方权重时，应确认它是否同时处理了 parameters 和 buffers。

---

## 8. EMA 与 AMP、梯度累积怎样配合？

EMA 与 [[AMP混合精度]] 可以同时使用。EMA 不关心在线模型是用 FP32、FP16 还是 BF16 完成前向与反向传播，它只需要在真正的优化器更新之后读取最新参数。

AMP 训练的典型顺序：

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

ema.update(model)
```

需要注意：FP16 梯度溢出时，`GradScaler` 可能跳过本次 `optimizer.step()`。严格实现可以只在优化器确实完成参数更新时再更新 EMA，避免将同一份在线参数重复计入 EMA。

梯度累积时，也应在累计完成、真正执行一次 `optimizer.step()` 后再更新 EMA，而不是每个 micro-batch 都更新。

---

## 9. EMA 与其他“平均”方法的区别

### 9.1 EMA 与简单滑动平均

简单平均会给选定区间内的所有模型参数相同权重；EMA 给较新的参数更大权重，并且只需保存一份当前 EMA 状态，不需要保存大量历史模型。

### 9.2 EMA 与 SWA

SWA（Stochastic Weight Averaging，随机权重平均）通常在训练后期以固定间隔采样若干模型检查点，并进行等权平均。EMA 则通常贯穿训练过程，对每次更新后的参数进行指数加权。

| 特性 | EMA | SWA |
|---|---|---|
| 历史权重 | 越新权重越高 | 采样点通常等权 |
| 更新频率 | 常为每个 optimizer step | 常为训练后期定期采样 |
| 额外状态 | 一份平均参数 | 一份平均参数 |
| BatchNorm | 需明确处理 buffer | 通常需要重新统计 BN |
| 常见领域 | 通用训练、检测、生成模型 | 分类等监督学习场景 |

### 9.3 EMA 与模型集成

模型集成会保留多个独立模型，并对它们的预测结果做平均，推理成本较高。EMA 是在参数空间中维护单个平滑模型，最终推理成本与普通单模型基本相同。

---

## 10. Teacher–Student 中的 EMA

在半监督学习、自监督学习和知识蒸馏相关方法中，经常使用“EMA Teacher”：

- Student 模型正常参与梯度训练；
- Teacher 模型的参数由 Student 参数的 EMA 更新；
- Teacher 为 Student 生成更稳定的伪标签、特征或训练目标。

更新公式仍然相同：

$$
\theta_{teacher} \leftarrow
\beta\theta_{teacher} + (1-\beta)\theta_{student}
$$

这里的 EMA 模型不仅用于最终推理，还直接参与训练目标的构造。Mean Teacher、部分自监督方法及目标网络方法都可以看到相似思想。

不过应注意：并不是所有“目标网络”都严格等同于标准 EMA，有些算法使用固定周期的硬更新或不同的更新策略。

---

## 11. EMA 特别常见的应用场景

EMA 常见于：

- 图像分类、目标检测、语义分割；
- 半监督学习中的 Teacher 模型；
- 自监督学习中的目标编码器；
- GAN 等生成模型；
- 扩散模型（Diffusion Model）；
- 训练噪声较大或参数波动明显的任务。

扩散模型尤其常使用 EMA 权重进行采样，因为生成质量对模型参数的小幅波动较敏感，EMA 往往能得到更稳定、更好的生成结果。

---

## 12. 保存与恢复训练

如果训练支持断点恢复，检查点中应同时保存：

- 在线模型参数；
- EMA 模型参数；
- 优化器状态；
- 学习率调度器状态；
- AMP 的 `GradScaler` 状态（如果使用）；
- 当前 epoch、global step 和 EMA 更新次数。

示例：

```python
checkpoint = {
    "model": model.state_dict(),
    "ema_model": ema.ema_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": global_step,
}
torch.save(checkpoint, "checkpoint.pt")
```

恢复时不能只加载在线模型，否则 EMA 的历史平滑状态会丢失。若旧检查点中没有 EMA 权重，可以用恢复后的在线模型重新初始化 EMA，但这相当于重新开始累计 EMA。

如果只发布推理模型，则通常只需导出 EMA 权重，并清楚标注该权重是 EMA 版本。

---

## 13. 常见误区与注意事项

### 误区 1：EMA 模型也要加入 optimizer

不需要。EMA 参数不能由梯度优化器更新，否则会破坏参数滑动平均逻辑。

### 误区 2：在 `optimizer.step()` 之前更新 EMA

应该先让在线模型完成参数更新，再让 EMA 吸收最新参数。否则 EMA 每次看到的是旧参数，整体会额外滞后一步。

### 误区 3：decay 越接近 1 越好

过大的 decay 在短训练中可能让 EMA 严重滞后。应结合总训练步数、更新频率和任务特点选择。

### 误区 4：EMA 一定优于在线模型

EMA 经常有效，但并非理论上保证提升。如果学习率、训练时长或 decay 不合适，EMA 模型也可能更差。因此应同时记录在线模型和 EMA 模型的验证指标。

### 误区 5：EMA 能解决训练发散

EMA 可以平滑参数噪声，但无法修复错误标签、学习率过大、损失函数非法、梯度爆炸等根本问题。在线模型已经严重发散时，EMA 最终通常也会受到影响。

### 其他注意事项

- EMA 参数更新应放在 `torch.no_grad()` 中；
- 分布式数据并行训练中，各 rank 的在线参数通常保持同步，但仍应确认 EMA 更新策略一致；
- 包含 BatchNorm 时，应确认 buffer 的复制、平均或重估策略；
- 模型结构改变后，参数名和形状不一致会导致 EMA 状态无法直接加载；
- 使用 `torch.compile`、参数分片或大模型并行时，优先采用与框架兼容的 EMA 实现；
- 对比实验时，应明确报告 decay、启动时机、更新频率以及最终评估使用的是在线权重还是 EMA 权重。

---

## 14. 总结

EMA 模型是在正常训练模型之外维护的一份**参数指数移动平均副本**：

$$
\bar{\theta}_t = \beta\bar{\theta}_{t-1} + (1-\beta)\theta_t
$$

它不参与反向传播，而是在每次有效的优化器更新后缓慢跟随在线模型。其主要价值是平滑训练噪声、减少参数震荡，并经常获得更稳定的验证表现和更好的泛化或生成质量。

实际使用时最重要的几点是：

1. 先执行 `optimizer.step()`，再更新 EMA；
2. 根据训练长度和更新频率选择 decay，而不是盲目设得很大；
3. 正确处理 BatchNorm 等 buffer；
4. 断点中同时保存在线模型与 EMA 状态；
5. 最终效果要通过在线权重与 EMA 权重的实际验证指标来判断。

