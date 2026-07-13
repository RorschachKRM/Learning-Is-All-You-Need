---
tags:
  - PyTorch
  - Deep-Learning
related:
  - "[[深度学习基础知识点]]"
---

## 1. 标准导入

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

- `torch`：核心库（张量、数学运算）
- `torch.nn`：网络层、损失函数
- `torch.nn.functional`：无状态的函数式操作（激活函数、填充等）

---

## 2. nn.Module —— 所有网络的基类

```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()       # 必须调用父类初始化
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

- 必须继承 `nn.Module`
- `__init__` 中定义层，`forward` 中定义前向逻辑
- `super().__init__()` 不可省略

---

## 3. nn.Linear —— 线性层 / 全连接层

```python
layer = nn.Linear(in_features=128, out_features=512)
y = layer(x)    # x: (B, 128) → y: (B, 512)
```

数学本质：$y = Wx + b$，内部自动管理权重和偏置。

---

## 4. 常用激活函数

| 函数 | 调用方式 | 特点 |
|------|---------|------|
| ReLU | `F.relu(x)` / `nn.ReLU()` | 最常用，负值截 0 |
| SiLU | `F.silu(x)` / `nn.SiLU()` | 比 ReLU 更平滑 |
| GELU | `F.gelu(x)` / `nn.GELU()` | Transformer 常用 |

`nn.Xxx()` 用于 `nn.Sequential` 中，`F.xxx()` 用于 `forward` 中直接调用。

---

## 5. nn.Sequential —— 串联多个模块

按顺序依次执行每个子模块，前一个的输出自动成为后一个的输入。

```python
self.mlp = nn.Sequential(
    nn.Linear(256, 512),
    nn.SiLU(),
    nn.Linear(512, 256),
)
y = self.mlp(x)   # x → Linear → SiLU → Linear → y
```

**可以混合自定义模块和内置层：**

```python
self.time_mlp = nn.Sequential(
    SinusoidalPosEmb(time_dim),       # 自定义模块也能放进去
    nn.Linear(time_dim, time_dim * 2),
    nn.SiLU(),
    nn.Linear(time_dim * 2, time_dim),
)
```

只要继承自 `nn.Module`，就能放进 `nn.Sequential`。

---

## 6. MLP 标准结构

```python
nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.SiLU(),
    nn.Linear(d_hidden, d_out),
)
```

公式：**Linear → 激活 → Linear**，两层夹一个非线性激活是最基础的 MLP。

---

## 7. torch.arange —— 张量创建

| 方法                          | 说明                          | 示例                                |
| --------------------------- | --------------------------- | --------------------------------- |
| `torch.arange(n)`           | 创建 `[0, 1, ..., n-1]` 的一维张量 | `torch.arange(5)` → `[0,1,2,3,4]` |
| `torch.arange(n, device=d)` | 指定设备                        | `torch.arange(5, device="cuda")`  |

---

## 8. 张量形状操作

| 操作 | 说明 | 输出形状变化 |
|------|------|-------------|
| `t[:, None]` | 在第 1 维插入一维，变列向量 | `(B,)` → `(B, 1)` |
| `t[None, :]` | 在第 0 维插入一维，变行向量 | `(D,)` → `(1, D)` |
| `t.unsqueeze(0)` | 同 `[None, :]`，更显式的写法 | |
| `t.dim()` | 获取维度数 | |
| `t.float()` | 转为 float32 类型 | |
| `t.device` | 获取张量所在设备 | 返回 `cpu`/`cuda:0` 等 |

---

## 9. Broadcasting (Broadcasting)

不同形状的张量运算时，PyTorch 自动从最后一维向前对齐扩展：

```python
t = torch.tensor([1.0, 2.0, 3.0])   # (3,)
t = t[:, None]                       # (3, 1)
emb = torch.arange(4)                # (4,)
out = t * emb                        # (3, 1) × (4,) → (3, 4)  自动广播
```

规则：从右向左，维度相等或有一方为 1 即可广播。

---

## 10. torch.cat —— 张量拼接

```python
a = torch.tensor([[1, 2]])   # (1, 2)
b = torch.tensor([[3, 4]])   # (1, 2)
torch.cat([a, b], dim=0)     # (2, 2) 按行拼
torch.cat([a, b], dim=-1)    # (1, 4) 按列拼（最后一维）
```

`dim=-1` 表示沿最后一维拼接，是最常用的写法。

---

## 11. torch.数学运算

```python
torch.exp(x)     # e^x
torch.sin(x)     # sin(x)
torch.cos(x)     # cos(x)
```

均为逐元素运算。

---

## 12. F.pad —— 张量填充

```python
F.pad(x, (0, 1))         # 只在最后一维右侧补 1 个零
F.pad(x, (1, 0, 0, 1))   # (左, 右, 上, 下) 补零
```

`pad` 参数格式：从最后一维向前，每维两个数（左填充，右填充）。


## 13. F.interpolate —— 张量上/下采样

```python
F.interpolate(x, scale_factor=2, mode="nearest")
```

调整输入张量的空间尺寸（H、W），常用于 U-Net / DDPM 的上采样或下采样阶段。

| 参数 | 说明 |
|------|------|
| `size` | 目标输出尺寸 `(H, W)`，与 `scale_factor` 二选一 |
| `scale_factor` | 缩放因子，如 `2` 表示 H、W 各放大 2 倍 |
| `mode` | 插值算法：`"nearest"` / `"bilinear"` / `"bicubic"` 等 |

**常用 mode 对比：**

| mode | 原理 | 特点 |
|------|------|------|
| `"nearest"` | 新像素直接复制最近的原像素值 | 最快，输出有块状感，但**不引入伪影** |
| `"bilinear"` | 周围 2×2 像素的加权平均 | 平滑，但可能产生棋盘伪影 |

**生成模型（DDPM / U-Net）中的最佳实践：**

`Nearest 上采样 + 3×3 Conv` 是公认最干净的上采样模式：

```python
x = F.interpolate(x, scale_factor=2, mode="nearest")  # 粗糙放大，不引入伪影
x = self.conv(x)                                       # 3×3 卷积负责平滑+特征重组
```

- Nearest 只负责尺寸翻倍，不产生灰度渐变和棋盘伪影
- 后续卷积有可学习参数，自己学出最优平滑策略
- 比 `ConvTranspose2d` 和 `bilinear` 上采样更稳定

---

## 14. PyTorch 中最常见的装饰器

```python
@torch.no_grad()         # 禁用梯度计算（推理/验证时用）
def evaluate(model, data):
    ...

@torch.inference_mode()  # 更彻底的推理模式（PyTorch 1.9+）
def predict(model, x):
    ...
```

这两个不是装饰函数，而是**上下文管理器的装饰器写法**，和 `with torch.no_grad():` 等价。

---

## 15. nn.Conv2d —— 2D 卷积层

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

CNN / U-Net / DDPM 的核心算子，用可学习的卷积核在 2D 特征图上滑动做内积。

| 参数 | 含义 |
|------|------|
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数（= 卷积核个数） |
| `kernel_size` | 卷积核尺寸，`3` 表示 3×3 |
| `stride` | 滑动步长，默认 1 |
| `padding` | 边缘零填充数，默认 0 |

**形状变化：** 输入 `(B, C_in, H, W)` → 输出 `(B, C_out, H', W')`

$$H' = \frac{H + 2 \times padding - kernel}{stride} + 1$$

**常用配置：**

| 配置 | 效果 |
|------|------|
| `kernel=3, padding=1` | **Same conv**，尺寸不变，只变换特征 — U-Net / ResNet 标配 |
| `kernel=1` | 1×1 卷积，只沿通道维做线性混合，不混合空间信息 |
| `kernel=3, stride=2, padding=1` | 下采样，H、W 减半 |
| `kernel=3, stride=1, padding=0` | 不做填充，每层尺寸缩小 2 |

**参数量：** $C_{out} \times C_{in} \times K \times K + C_{out}$（最后一项为 bias）。


## 16. DataLoader
`DataLoader` 是 PyTorch 的数据加载器，来自 `torch.utils.data.DataLoader`。
Dataset 管"怎么读一个样本"，DataLoader 管"怎么攒一批样本"。
核心作用：把 `Dataset`（你的 `SAMPLE_Dataset`）包装成**可批量迭代**的对象，自动帮你做：

| 功能        | 说明                                                  |
| --------- | --------------------------------------------------- |
| **批量打包**  | 把多个样本堆叠成 batch tensor                               |
| **随机打乱**  | `shuffle=True` 每 epoch 洗牌                           |
| **多线程加载** | `num_workers` 并行读数据，不阻塞 GPU                         |
| **自动批处理** | 通过 `collate_fn` 把 list of dict 合并成 batch of tensors |
### 用法示例
```python
from torch.utils.data import DataLoader

dataset = SAMPLE_Dataset(txt_file='data/sample_train.txt')

dataloader = DataLoader(
    dataset,
    batch_size=32,      # 每次取32个样本
    shuffle=True,       # 训练时打乱
    num_workers=4,      # 4个线程并行加载
    drop_last=True,     # 丢弃最后不够一个batch的
)

for batch in dataloader:
    images = batch['image']   # shape: (32, 1, 128, 128)
    labels = batch['label']   # shape: (32,)
    azs    = batch['az']      # shape: (32,)
    # 喂给模型训练...
```

### 数据流
```
磁盘图片 ──> SAMPLE_Dataset.__getitem__() ──> 单样本 dict
                                                      │
                                              DataLoader 收集32个
                                                      ↓
                                                  batch dict
                                          {'image': (32,1,128,128),
                                           'label': (32,),
                                           'az': (32,),
                                           'name': (...)}

```


## 17. NumPy vs Tensor 核心区别

|          | **  NumPy ndarray** | **PyTorch Tensor**                  |
| -------- | ------------------- | ----------------------------------- |
| **运行位置** | 只能 CPU              | CPU + **GPU** (`tensor.to('cuda')`) |
| **自动求导** | 不支持                 | 支持 `requires_grad=True`，自动算梯度       |
| **生态**   | 科学计算、数据处理           | 深度学习训练、推理                           |
| **底层内存** | 系统内存                | 可跑在 GPU 显存上                         |
### 举例
```python
import numpy as np
import torch

# 长得一样，用起来也像
a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
a_t  = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# 大多数操作语法相同
print(a_np.shape)   # (2, 2)
print(a_t.shape)    # (2, 2)
print(a_np + 1)     # [[2, 3], [4, 5]]
print(a_t + 1)      # [[2, 3], [4, 5]]

# 但关键区别在这里 ——
a_t = a_t.to('cuda')         # ✅ tensor 搬到 GPU 上算
a_np.to('cuda')              # ❌ numpy 做不到

# 自动求导
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2
y.backward()                 # 自动算出 dy/dx = 2*x = 6
print(x.grad)                # tensor([6.])
# numpy 完全没有这个概念
```


## 18. model.eval() & model.train()
### 一、基本定义

|方法|等价写法|`self.training`|场景|
|---|---|---|---|
|`model.train()`|`model.train(mode=True)`|`True`|训练|
|`model.eval()`|`model.train(mode=False)`|`False`|验证/推理/生成|
|`model.train(False)`|—|`False`|同 `eval()`|
它们定义在 `torch.nn.Module` 基类中，**递归地作用于模型及其所有子模块**。

### 二、源码本质

```python
class Module:
    training: bool  # 核心标志位

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)    # 递归设置所有子模块
        return self

    def eval(self):
        return self.train(False)  # 本质就是 train(False)
```
关键点：**`eval()` 没有独立的实现，它只是 `train(False)` 的别名**。

### 三、影响的具体层

所有通过 `self.training` 判断行为的层都会受影响：
#### 1. BatchNorm (BatchNorm1d / 2d / 3d)

BatchNorm 是推理时必须切 eval 的**最重要原因**。

| |`train()`|`eval()`|
|---|---|---|
|均值/方差来源|当前 batch 的统计量|训练期积累的 running_mean / running_var（冻结）|
|是否更新 running 统计量|是（指数移动平均）|否|

```python
# BatchNorm forward 的简化为：
def forward(self, x):
    if self.training:
        mean = x.mean(dim=0)           # 当前 batch 的均值
        var = x.var(dim=0, unbiased=False)  # 当前 batch 的方差
        # 更新 running mean/var（指数移动平均）
        self.running_mean = momentum * self.running_mean + (1-momentum) * mean
        self.running_var  = momentum * self.running_var  + (1-momentum) * var
    else:
        mean = self.running_mean       # 使用训练期积累的统计量
        var = self.running_var
    return (x - mean) / sqrt(var + eps) * gamma + beta
```

**为什么推理必须用 running_mean？**
训练时 batch 通常较大（如 32、64），统计量稳定；推理时可能 `batch_size=1`，单张图的均值和方差**毫无意义**。用 running_mean 是全局数据分布的无偏估计。

#### 2. Dropout (Dropout / Dropout2d / Dropout3d)

| |`train()`|`eval()`|
|---|---|---|
|行为|以概率 p 随机置零神经元|不做任何操作，直接透传|
|缩放|幸存神经元 × 1/(1-p) 补偿期望值|不缩放|

```python
# Dropout forward 简化：
def forward(self, x):
    if self.training:
        mask = Bernoulli(1 - p).sample(x.shape)  # 随机掩码
        return x * mask / (1 - p)               # 放大以保持期望不变
    else:
        return x                                  # 什么都不做
```

**为什么推理时关闭 Dropout？**
- Dropout 是一种**正则化**手段，随机丢弃迫使网络不过度依赖特定神经元
- 推理时需要**确定性输出**：同一输入必须产生同一输出
- 训练时的 `× 1/(1-p)` 缩放已经保证了期望值一致，推理时直接用全连接等价于对所有 dropout 子网络取**集成平均**

|模块|`train()` 模式|`eval()` 模式|
|---|---|---|
|**BatchNorm**|使用当前 batch 的均值和方差，并更新 running mean/var|使用训练期间积累的 running mean/var（冻结）|
|**Dropout**|随机置零神经元（引入噪声，用于正则化）|关闭，所有神经元都参与计算|
如果不调用 `.eval()`，在生成图像时：
- Dropout 会随机丢弃信息，导致每次生成的图像**不稳定**（同样的输入产生不同输出）；
- BatchNorm 统计量不准确（推理时 batch 可能很小），影响生成质量

#### 3. LayerNorm / GroupNorm / InstanceNorm

|归一化类型|`train()` vs `eval()` 行为差异|
|---|---|
|LayerNorm|**一般无差异**（逐样本归一化，不依赖 batch 统计量；但如果有 `elementwise_affine=True` 且实现了 running_statistics 的变体则可能不同）|
|GroupNorm|**无差异**（逐样本、组内归一化）|
|InstanceNorm|**PyTorch 默认实现有差异**——track_running_stats=True 时和 BatchNorm 一样区分 train/eval|
**关键区别**：LN/GN 是对单个样本内部做归一化，不依赖跨样本统计量，所以 `self.training` 通常不影响它们的行为。

#### 4. 其他受影响的模块

|层|train() 行为|eval() 行为|
|---|---|---|
|**MultiheadAttention**|需要 `batch_first` 等参数|同样可用，无行为差异|
|**LSTM/GRU** 的 dropout 参数|层间 dropout 生效|层间 dropout 关闭|
|**RReLU**|负半轴斜率为随机均匀采样|负半轴斜率固定为 `(lower+upper)/2` 的期望值|
|**BatchNorm 的 track_running_stats**|更新 running 统计量|冻结 running 统计量|

### 四、`with torch.no_grad()` 与 `eval()` 的区别

**这是最常见的混淆点——两者完全不同且通常需要同时使用：**

| |`model.eval()`|`torch.no_grad()`|
|---|---|---|
|控制什么|层的**前向传播行为**（BN 用什么统计量、Dropout 是否开启）|是否**构建计算图**|
|影响范围|仅模型自身|影响包裹区域内的所有 tensor 运算|
|是否省显存|不直接影响|**是**——不保存中间激活，大幅节省显存|