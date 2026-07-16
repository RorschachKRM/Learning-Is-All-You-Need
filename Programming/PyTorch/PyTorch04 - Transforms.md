---
title: 变换
source: https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html
tags:
  - clippings
  - PyTorch
  - Transforms
---
[基础入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || **Transforms（变换）** || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || [自动求导](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || [优化](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || [保存与加载模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

# 转换

数据并不总是以机器学习算法训练所需的最终处理形式出现。我们使用 **transforms（变换）** 来对数据进行一些操作，使其适合训练。

所有的 TorchVision 数据集都有两个参数—— 
	`transform` （用于修改特征）
	`target_transform` （用于修改标签）
——它们接收包含变换逻辑的可调用对象。 [torchvision.transforms](https://pytorch.ac.cn/vision/stable/transforms.html) 模块提供了多种开箱即用的常用变换。

FashionMNIST 的特征采用 PIL 图像格式，标签为整数。
在训练时，我们需要将==**特征处理为归一化的张量**==，将==**标签处理为独热编码（one-hot encoded）张量**==。
为了进行这些变换，我们使用 `torchvision.transforms.v2` API 以及 `torch.nn.functional.one_hot` 。

```python
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)
```
## 代码解析==target_transform==
- `v2.Lambda` ：是一个通用的变换包装器，来自 `torchvision.transforms.v2`。它接收**任意可调用对象**（函数、lambda、实现了 `__call__` 的类等），并将其包装成一个标准的 transform。每当数据集取出一条标签 `y`，就会自动传入这个 lambda 并返回变换后的结果
-  `torch.tensor(y)` ：把这个标量整数包装成一个 **0 维张量（标量张量）**，shape 为 `()`，例如 `tensor(3)`。因为 `F.one_hot` 的第一个参数必须是张量。
- `F.one_hot(..., num_classes=10)`：将**整数索引**转换为**独热编码向量**：
	- **输入**：`tensor(3)` — 一个标量，值为 3
	- **`num_classes=10`**：指定向量长度为 10（FashionMNIST 有 10 个类别）
	- **输出**：`tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])` — shape 为 `(10,)`，只有第 3 位（从 0 开始）是 1，其余为 0
- .float()：转为 `torch.float32`。
	- **什么要转 float？** 因为损失函数（如 `CrossEntropyLoss` 或 `BCEWithLogitsLoss`）期望标签为浮点类型，且模型的输出也是 float，保持 dtype 一致可以避免隐式类型转换的警告或错误。



# transforms.v2
`torchvision.transforms.v2` API 用一个两步流水线：
```python
transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
```
- [v2.ToImage](https://pytorch.ac.cn/vision/stable/generated/torchvision.transforms.v2.ToImage.html) 将 PIL 图像或 NumPy `ndarray` 转换为 `torchvision.tv_tensors.Image` 张量。
- 带有 `scale=True` 参数的 [v2.ToDtype](https://pytorch.ac.cn/vision/stable/generated/torchvision.transforms.v2.ToDtype.html) 将其转换为 `float32` 类型，并将像素强度值缩放到 \[0., 1.\] 区间。

取代了传统的 `ToTensor` 变换。


## 老版 .`ToTensor`解析
```python
transforms.ToTensor()
```
`ToTensor`把三件事揉在一起做了：

| 步骤  | 做了什么                                              |
| --- | ------------------------------------------------- |
| ①   | 将 PIL Image（`mode='L'` 或 `'RGB'`）→ `torch.Tensor` |
| ②   | 将像素值从 `[0, 255]`（整数）缩放到 `[0.0, 1.0]`（浮点）          |
| ③   | 将通道维度从 H×W×C（PIL 格式）变为 C×H×W（PyTorch 格式）          |
问题在于：这三件事捆在一起，你没法单独控制某一步。比如你想保留 `uint8` 类型？做不到。你想让图片保持 H×W×C 格式？做不到。这违反了"一个函数只做一件事"的原则。

v2 就把这三件事拆开了：
```
ToImage()       →  只负责格式转换（PIL → Tensor，C×H×W）
ToDtype()       →  只负责类型转换和数值缩放
```
需要额外的行为（如归一化、裁剪）？往 `Compose` 里继续加就行，互不干扰。


## `v2.ToImage()` 解析

**一句话：把"图片"统一成 PyTorch 能高效处理的张量格式。**
具体行为取决于输入类型：

|输入类型|转换行为|输出类型|
|---|---|---|
|`PIL.Image.Image`（灰度 `L` 或 RGB）|解码为数值数组，通道从 H×W×C → C×H×W|`torchvision.tv_tensors.Image`|
|`numpy.ndarray`（shape H×W×C）|通道重排 H×W×C → C×H×W|`torchvision.tv_tensors.Image`|
|已经是 `torch.Tensor`|包装为 `tv_tensors.Image`（不改变数据）|`torchvision.tv_tensors.Image`|
> **关键点**：`ToImage()` **不改变数值大小，不改变 dtype**。PIL 图片的像素值（0~255 整数）转换为张量后仍然是 0~255 的整数（`uint8`）。

### `tv_tensors.Image` 是什么？

它不是普通的 `torch.Tensor`，而是它的**子类**：
```python
from torchvision.tv_tensors import Image

# tv_tensors.Image 是 torch.Tensor 的子类
issubclass(Image, torch.Tensor)  # True
```
为什么需要这个子类？因为它携带了**元数据**——告诉下游的 transform "这是一张图片，通道在 dim=0"，这样后续的 `ToDtype`、`Resize`、`Normalize` 等操作能自动做出正确的行为（比如知道应该对哪个维度做缩放）。


## `v2.ToDtype()` 解析

```python
v2.ToDtype(dtype, scale=False)
```

| 参数      | 含义                       |
| ------- | ------------------------ |
| `dtype` | 目标数据类型，如 `torch.float32` |
| `scale` | **核心参数**：是否按输入类型的最大值做缩放  |
### 不缩放的情况（`scale=False`，默认值）

只做**纯类型转换**，数值不变：
```python
# uint8 的 [0, 255] → float32 的 [0.0, 255.0]
tensor([0, 128, 255], dtype=torch.uint8)
# ↓ ToDtype(scale=False)
tensor([0., 128., 255.], dtype=torch.float32)
```

### 缩放的情况（`scale=True`）

类型转换 **+ 自动缩放**。缩放规则：**将数值除以输入 dtype 的理论最大值**。
```python
# 输入是 uint8 → 除以 255
tensor([0, 128, 255], dtype=torch.uint8)
# ↓ ToDtype(scale=True)
tensor([0.0, 0.50196..., 1.0], dtype=torch.float32)
```
每种输入 dtype 对应的缩放因子：

|输入 dtype|最大值|缩放因子|
|---|---|---|
|`uint8`|255|÷ 255|
|`int16`|32767|÷ 32767|
|`float32`|1.0（假设已在 [0,1]）|÷ 1（不缩放）|
这就是 `tv_tensors.Image` 子类发挥作用的地方——`ToDtype` 能感知到输入是图片，知道应该用像素值的语义范围来缩放。

### 为什么 `scale=True` 很重要？

神经网络对输入数值范围非常敏感。原始像素值 0~255 直接喂给网络会导致：
1. **梯度爆炸/消失**：权重更新时梯度值过大，训练不稳定
2. **收敛慢**：不同特征的量纲差异巨大时，优化器需要更多的迭代
3. **数值精度问题**：float32 下大数值的加减运算精度损失更明显
缩放到 `[0.0, 1.0]` 后，输入处于一个**良好约束的范围内**，网络训练更稳定、收敛更快。


### 一个典型的完整流水线示例

```python
transform = v2.Compose([
    v2.ToImage(),                              # PIL → Tensor, uint8
    v2.RandomResizedCrop(224),                 # 随机裁剪（在 uint8 上更快）
    v2.RandomHorizontalFlip(),                 # 随机翻转（在 uint8 上更快）
    v2.ToDtype(torch.float32, scale=True),    # uint8 → float32, 缩放到 [0,1]
    v2.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化（在 float 上做）
                 std=[0.229, 0.224, 0.225]),
])
```


# Lambda 变换

Lambda 变换应用任何用户定义的 lambda 函数。
在此，我们使用 [torch.nn.functional.one_hot](https://pytorch.ac.cn/docs/stable/generated/torch.nn.functional.one_hot.html) 将整数标签转换为大小为 10（即我们数据集中的标签数量）的独热编码张量，然后将其转换为 `float` 类型以匹配预期的 dtype。
```python
target_transform = v2.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)
```

## 为什么需要这个包装器？

直接写 lambda 不行吗？比如：
```python
target_transform = lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
```

**技术上可行**，但 `v2.Lambda` 提供了三样东西：
### ① 兼容 transform 管线

v2 的所有 transform 是**有类型、有元数据的对象**，不是裸函数。`Lambda` 把你的函数包装成一个 `Transform` 对象，使其可以：
- 被 `v2.Compose` 串联
- 被 `dataset.target_transform` 统一管理
- 在序列化/反序列化时保持一致性

### ② 自动处理 batch 维度（关键！）

v2 的 transform 和普通的 `torchvision.transforms` 不同——它能**自动判断输入是单张图片还是一个 batch**，并相应调整行为。
```python
# 单样本
y = 3                                  # int
# Lambda 内部: 得到 tensor(3)

# 批量样本（如果直接对 batch 用）
y = [3, 7, 1, 5]                      # list
# Lambda 内部: 得到 tensor([3, 7, 1, 5])
```
不过对于 `target_transform`，PyTorch 的 DataLoader 默认是对单条样本调用的（`collate_fn` 在 transform 之后才组装 batch），所以这一点在这里不直接体现。但理解这个设计很有必要。



# F.one_hot()函数
```python
import torch.nn.functional as F

torch.nn.functional.one_hot(tensor, num_classes=-1)
#                           ^^^^^^ 必须是 Tensor
# num_classes：输出向量的长度（必须 > 输入中的最大值）

```

`torch.tensor(3)` 创建一个**0 维（标量）张量**，不是 1 维向量
**`torch.tensor(3)` 是标量，`torch.tensor([3])` 才是向量**。

## 处理更高维输入

`one_hot` 不仅支持标量，也支持任意维度的张量：
```python
# 二维输入
y = torch.tensor([[0, 3], [5, 9]])            # shape (2, 2)
oh = F.one_hot(y, num_classes=10)
print(oh.shape)  # torch.Size([2, 2, 10])     ← 在末尾新增一维

# 整个 batch
y = torch.tensor([0, 3, 5, 7, 9])             # shape (5,)
oh = F.one_hot(y, num_classes=10)
print(oh.shape)  # torch.Size([5, 10])        ← 每个样本的 one-hot 向量
```
规则是：**输出的 shape = 输入的 shape + (num_classes,)**。