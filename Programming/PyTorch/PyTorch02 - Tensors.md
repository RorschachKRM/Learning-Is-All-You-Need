---
title: 张量
source: https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html
tags:
  - PyTorch
  - Tensor
  - clippings
---

[基础入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || **张量 (Tensors)** || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || [数据变换 (Transforms)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html) || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || [自动求导 (Autograd)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || [模型优化](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || [保存与加载模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

# 张量

张量是一种特殊的数学结构，与数组和矩阵非常相似。在 PyTorch 中，我们使用张量来编码模型的输入、输出以及模型的参数。

张量与 [NumPy](https://numpy.com.cn/) 的 ndarray 非常相似，不同之处在于张量可以在 GPU 或其他硬件加速器上运行。事实上，张量和 NumPy 数组通常可以共享底层的内存，从而无需复制数据（请参阅 [与 NumPy 的交互](https://docs.pytorch.ac.cn/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label) ）。张量还针对自动求导进行了优化（稍后我们将在 [自动求导](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) 部分详细介绍）。
```python
import torch
import numpy as np
```

# 初始化张量

可以通过多种方式初始化张量。请看以下示例：

## **直接从数据创建**
可以直接从数据创建张量。数据类型会自动推断。
```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

## **从 NumPy 数组创建**
可以从 NumPy 数组创建张量（反之亦然 - 请参阅 [与 NumPy 的交互](https://docs.pytorch.ac.cn/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label) ）。
```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

## **从另一个张量创建**
新张量保留了原张量的属性（形状、数据类型），除非显式重写。
```python
x_ones = torch.ones_like(x_data) # 保留 x_data 的属性
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # 覆盖 x_data 的数据类型
print(f"Random Tensor: \n {x_rand} \n")

"""
_like是一种语义命名约定，意思是：创建一个新张量，其属性（形状、数据类型、设备）与参数x相同。
例如：torch.ones_like(x)：全 1 张量，形状同x
"""
```

```Output
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.3789, 0.9786],
        [0.2249, 0.3116]])
```

## **使用随机或常量值创建**
`shape` 是一个元组，代表张量的维度。在下面的函数中，它决定了输出张量的维度。
```python
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

```Output
Random Tensor:
 tensor([[0.0530, 0.3384, 0.6637],
        [0.8872, 0.5051, 0.0376]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```


## 总结

PyTorch 创建张量的函数分两大阵营：

| 阵营      | 函数                                                                               | 第一个参数的含义 | 语法                 |
| ------- | -------------------------------------------------------------------------------- | -------- | ------------------ |
| **传数据** | `torch.tensor()`                                                                 | 实际的数值    | `torch.tensor(数据)` |
| **传形状** | `torch.zeros()` ，`torch.ones()`，`torch.randn()` ，`torch.empty()`， `torch.full()` | 张量的形状    | `torch.zeros(形状)`  |
### ## `torch.tensor()` —— 传数据
规则：第一个参数是你要的数字本身

```python
# ① 直接数字 → 0 维标量
torch.tensor(5)          # tensor(5)              shape: ()

# ② [ ] 列表 → 1 维向量
torch.tensor([5])        # tensor([5])            shape: (1,)
torch.tensor([1, 2, 3])  # tensor([1, 2, 3])      shape: (3,)

# ③ ( ) 元组 → 同样是 1 维向量（效果等同于列表）
torch.tensor((5,))       # tensor([5])            shape: (1,)
torch.tensor((1, 2, 3))  # tensor([1, 2, 3])      shape: (3,)

# ④ 嵌套 → 多维
torch.tensor([[1, 2], [3, 4]])   # shape: (2, 2)  二维矩阵
```
**一句话**：`torch.tensor()` 只看你传的数据"长什么样"，不看用的是 `[]` 还是 `()`。

```python
torch.tensor(5)      # 标量，shape ()
torch.tensor([5])    # 向量，shape (1,)  ← 有一个元素
torch.tensor((5,))   # 向量，shape (1,)  ← 同上

# (5) 不加逗号就是普通括号，等于 5
torch.tensor((5))    # 标量，shape ()   ← Python 把它当括号里的 5
```
> Python 语法细节：`(5)` 不是元组，就是加了括号的数字 5。`(5,)` 才是单元素元组。逗号是关键。


#### 一图流
| #   | 输入               | Python 解析        | `tensor()` 结果              | shape    |
| --- | ---------------- | ---------------- | -------------------------- | -------- |
| 1   | `5`              | 裸整数              | `tensor(5)`                | `()` 标量  |
| 2   | `[5]`            | 单元素列表            | `tensor([5])`              | `(1,)`   |
| 3   | `(5)`            | ⚠️ 括号不构成元组，= `5` | `tensor(5)`                | `()` 标量  |
| 4   | `[5,]`           | 单元素列表（尾逗号合法）     | `tensor([5])`              | `(1,)`   |
| 5   | `(5,)`           | 单元素元组（逗号是元组本体）   | `tensor([5])`              | `(1,)`   |
| 6   | `[1,2,3]`        | 三元素列表            | `tensor([1, 2, 3])`        | `(3,)`   |
| 7   | `(1,2,3)`        | 三元素元组            | `tensor([1, 2, 3])`        | `(3,)`   |
| 8   | `[[1,2], [3,4]]` | 嵌套列表             | `tensor([[1, 2], [3, 4]])` | `(2, 2)` |
| 9   | `((1,2), (3,4))` | 嵌套元组             | `tensor([[1, 2], [3, 4]])` | `(2, 2)` |
| 10  | `[(1,2), (3,4)]` | 列表包元组            | `tensor([[1, 2], [3, 4]])` | `(2, 2)` |
| 11  | `([1,2], [3,4])` | 元组包列表            | `tensor([[1, 2], [3, 4]])` | `(2, 2)` |
| 12  | 1,2,3,4          | ❌报错              |                            |          |


### `torch.zeros()` / `torch.ones()` / `torch.randn()` 等 —— 传形状
规则：参数指定的是"几行几列"，不是数据

这些函数有两种传形状的方式：
```python
# 方式 A：多个独立参数
torch.zeros(3)         # 1 维，3 个元素
torch.zeros(3, 4)      # 2 维，3 行 4 列
torch.zeros(3, 4, 5)   # 3 维，3×4×5

# 方式 B：一个元组（或列表）
torch.zeros((3,))      # 等同 zeros(3)
torch.zeros((3, 4))    # 等同 zeros(3, 4)
torch.zeros([3, 4])    # 列表也可以
```

如果给 `zeros()` 传一个裸数字 5？
```python
torch.zeros(5)
# → tensor([0., 0., 0., 0., 0.])   1 维，5 个零
```
它**不会**创建一个标量零。`5` 被理解为"长度为 5 的一维向量"。



---

# 张量的属性
张量属性描述了其形状、数据类型以及存储它的设备。
```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

```Output
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

---

# 张量运算
这里全面介绍了 1200 多种张量运算，包括算术运算、线性代数、矩阵操作（转置、索引、切片）、采样等，详见 [此处](https://pytorch.ac.cn/docs/stable/torch.html) 。

所有这些操作都可以在 CPU 以及 CUDA、MPS、MTIA 或 XPU 等 [加速器](https://pytorch.ac.cn/docs/stable/torch.html#accelerators) 上运行。如果你正在使用 Colab，可以通过“运行时 > 更改运行时类型 > GPU”来分配加速器。

默认情况下，张量是在 CPU 上创建的。我们需要在检查加速器可用性后，使用 `.to` 方法将张量显式移动到加速器上。请记住，在不同设备间复制大张量在时间和内存方面可能非常昂贵！
```python
# 如果可用，我们将张量移动到当前加速器。
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
```


## 索引和切片
==:（冒号）切片操作符==，表示取该维度上的"全部"，可以配合起止范围、步长使用。
	通用形式：start:stop:step，左闭右开[start, stop)

==...（省略号）==，含义是剩下的所有维度，全取。它是一个占位符，自动补全为所需数量的:。
```
	2D: tensor[..., -1]  等价于  tensor[:, -1]
	3D: tensor[..., -1]  等价于  tensor[:, :, -1]
	4D：tensor[..., -1]  等价于  tensor[:, :, :, -1]
```
比如处理图像 `(batch, channel, height, width)` 时：
```
	image[0, ...]         # 取第 0 张图的所有通道、所有像素
	image[..., :10, :10]  # 取每张图左上角 10×10 区域
```


```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

```Output
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```


## 拼接张量
你可以使用 `torch.cat` 沿指定维度连接一系列张量。
另请参阅 [torch.stack](https://pytorch.ac.cn/docs/stable/generated/torch.stack.html) ，这是另一种与 `torch.cat` 略有不同的张量拼接运算符。

```python
torch.cat(tensors, dim=0)

"""
参数：
	tensors：张量序列（list/tuple），要拼接的张量们
	dim：沿哪个维度拼接，默认0。按哪一维拼接，就是该维数值求和，其他维对齐不变。
"""
```
核心规则：**被拼接的张量在非 `dim` 维度上的大小必须完全一致。**
```
# 图解（2D 张量）
dim=0: 上下堆                dim=1: 左右拼

┌──────────┐                 ┌────┬────┐
│  tensor1 │                 │ t1 │ t2 │
├──────────┤                 └────┴────┘
│  tensor2 │                   列数相加
├──────────┤
│  tensor3 │
└──────────┘
   行数相加
```

```python
# 3D张量[2, 3, 4]：
a = torch.randn(2, 3, 4)   # [batch, height, width]
b = torch.randn(2, 3, 4)

torch.cat([a, b], dim=0)   # [4, 3, 4]   batch 翻倍
torch.cat([a, b], dim=1)   # [2, 6, 4]   height 翻倍
torch.cat([a, b], dim=2)   # [2, 3, 8]   width 翻倍
```


```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```Output
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```


## 算术运算
```python
# 下面是3个计算两个张量之间矩阵乘法的方式
# tensor.T返回张量的转置
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)  # 张量自带的matmul方法，和 @ 完全等价

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)  # out= 参数：原地输出，不分配新内存

# 这将计算元素级乘积。z1、z2、z3 将具有相同的值
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

```Output
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```


## **单元素张量** 
如果你有一个单元素张量（例如通过将张量的所有值聚合为一个值得到），你可以使用 `item()` 将其转换为 Python 数值。
```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

```Output
12.0 <class 'float'>
```


## **原地操作 (In-place operations)** 
将结果存储在操作数中的操作称为“原地”操作。它们以 `_` 后缀表示。例如： `x.copy_(y)` 或 `x.t_()` 将会改变 `x` 。
```python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```

```Output
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

> [!note] 注意
> 原地操作可以节省一些内存，但在计算导数时可能会因为立即丢失计算历史而产生问题。因此，通常不鼓励使用它们。

---

# 与 NumPy 的交互

CPU 上的张量和 NumPy 数组可以共享底层内存位置，改变其中一个也会改变另一个。

## 张量转 NumPy 数组

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

```Output
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

张量的变化会反映在 NumPy 数组中。

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

```Output
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

## NumPy 数组转张量

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

NumPy 数组的变化会反映在张量中。
```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```Output
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```
