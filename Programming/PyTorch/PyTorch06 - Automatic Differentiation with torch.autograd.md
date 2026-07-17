---
title: 自动求导 with torch.autograd
source: https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html
tags:
  - clippings
  - PyTorch
  - Automatic-Differentiation
  - Gradient
---
[基础入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || [转换 (Transforms)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html) || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || **自动微分 (Autograd)** || [优化 (Optimization)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || [保存与加载模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

# 使用 torch.autograd 进行自动微分

在训练神经网络时，最常用的算法是 **反向传播 (back propagation)** 。在此算法中，参数（模型权重）会根据损失函数相对于给定参数的 **梯度 (gradient)** 进行调整。

为了计算这些梯度，PyTorch 内置了一个名为 `torch.autograd` 的微分引擎。它支持自动计算任意计算图的梯度。

考虑最简单的单层神经网络，具有输入 `x` ，参数 `w` 和 `b` ，以及某个损失函数。它可以在 PyTorch 中以以下方式定义：
```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```
带 Logits 的二元交叉熵损失[[Loss Function#交叉熵]]



# 张量、函数和计算图

这段代码定义了以下 **计算图** ：

![](https://docs.pytorch.ac.cn/tutorials/_images/comp-graph.png)

在该网络中， `w` 和 `b` 是我们需要优化的 **参数** 。因此，我们需要能够计算损失函数相对于这些变量的梯度。为此，我们设置这些张量的 `requires_grad` 属性。

> [!note] 注意
> 您可以在创建张量时设置 `requires_grad` 的值，或者稍后使用 `x.requires_grad_(True)` 方法进行设置。

我们应用于张量以构建计算图的函数实际上是一个 `Function` 类的对象。该对象知道如何进行 *前向* 传播计算，以及如何在 *反向传播* 步骤中计算其导数。反向传播函数的引用存储在张量的 `grad_fn` 属性中。您可以在 [官方文档](https://pytorch.ac.cn/docs/stable/autograd.html#function) 中找到关于 `Function` 的更多信息。
```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

```
Gradient function for z = <AddBackward0 object at 0x7efe7db46e30>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7efe7db80340>
```


# 计算梯度

为了优化神经网络中的参数权重，我们需要计算损失函数相对于参数的导数，即在 `x` 和 `y` 固定值下的 $\frac{\partial loss}{\partial w}$ 和 $\frac{\partial loss}{\partial b}$ 。为了计算这些导数，我们调用 `loss.backward()` ，然后从 `w.grad` 和 `b.grad` 中获取值。
```python
loss.backward()
print(w.grad)
print(b.grad)
```

```
tensor([[0.1918, 0.0777, 0.2000],
        [0.1918, 0.0777, 0.2000],
        [0.1918, 0.0777, 0.2000],
        [0.1918, 0.0777, 0.2000],
        [0.1918, 0.0777, 0.2000]])
tensor([0.1918, 0.0777, 0.2000])
```

> [!note] 注意
> - 我们只能获取计算图中叶子节点的 `grad` 属性，且这些节点的 `requires_grad` 属性必须设置为 `True` 。对于图中所有其他节点，梯度将不可用。
> - 出于性能考虑，我们只能在给定的图上执行一次 `backward` 计算。如果我们需要在同一个图上多次进行 `backward` 调用，我们需要在调用 `backward` 时传入 `retain_graph=True` 。


# 禁用梯度跟踪

默认情况下，所有 `requires_grad=True` 的张量都会跟踪其计算历史并支持梯度计算。
然而，在某些情况下我们不需要这样做，例如当我们已经训练好模型并只想将其应用于某些输入数据时（即我们**只想在网络中进行 *前向* 计算**）。

我们可以通过将计算代码包裹在 `torch.no_grad()` 块中来停止跟踪计算。
```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

```
True
False
```

实现相同结果的另一种方法是在张量上使用 `detach()` 方法。
```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

```
False
```

您可能希望禁用梯度跟踪的原因包括：
- 将神经网络中的某些参数标记为 **冻结参数 (frozen parameters)** 。
- 当您仅进行前向传播时， **加快计算速度** ，因为对不跟踪梯度的张量进行计算效率更高。



# 代码示例：完整的训练循环 — 梯度生命周期

这是最核心的示例，展示梯度从计算、累加到清零的完整流程：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 准备数据 ---
torch.manual_seed(42)

# 模拟一个二分类问题：100 个样本，每个 5 维特征
X = torch.randn(100, 5)
y = torch.randint(0, 2, (100,)).float()
"""
生成一个包含 100 个随机 0/1 的二分类标签：
	torch.randint(0, 2, (100,)) → 形状(100,)的一维张量，值随机取 0 或 1（[0, 2) 左闭右开）
	.float()→ 将整数 0、1 转为浮点 0.0、1.0，因为后续 F.binary_cross_entropy要求标签是 float 类型
"""

# --- 2. 定义可训练参数 ---
w = torch.randn(5, 1, requires_grad=True)   # 权重，需要梯度
b = torch.randn(1, requires_grad=True)      # 偏置，需要梯度

# --- 3. 训练循环 ---
learning_rate = 0.01
num_epochs = 10

for epoch in range(num_epochs):
    # ----- 前向传播 -----
    logits = X @ w + b                      # (100, 1)
    preds = torch.sigmoid(logits).squeeze() # (100,)
    loss = F.binary_cross_entropy(preds, y) # 标量损失

    # ----- 反向传播 -----
    loss.backward()  # 计算 ∂loss/∂w 和 ∂loss/∂b，并累加到 w.grad / b.grad

    # ----- 手动梯度下降（等价于 optimizer.step()）-----
    with torch.no_grad():           # 参数更新不需要计算图
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # ----- 每次迭代结束后，清零梯度（否则会累加！）-----
    w.grad.zero_()
    b.grad.zero_()

    # ----- 监控 -----
    if epoch % 3 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | "
              f"||grad(w)||: {w.grad.norm().item():.6f} | "
              f"||grad(b)||: {b.grad.norm().item():.6f}")
        # 注意：grad 已被 zero_()，这里输出的是 0
```
**关键点**：
- `loss.backward()` 计算并**累加**梯度
- 参数更新必须在 `torch.no_grad()` 下进行（否则会建图）
- `grad.zero_()` 必须每次迭代调用，否则梯度越滚越大







# 选读：张量梯度与雅可比积

在许多情况下，我们有一个标量损失函数，需要计算相对于某些参数的梯度。然而，有时输出函数是一个任意的张量。在这种情况下，PyTorch 允许您计算所谓的 **雅可比积 (Jacobian product)** ，而不是实际的梯度。

对于向量函数 $\vec{y}=f(\vec{x})$ ，其中 $\vec{x}=\langle x_1,\dots,x_n\rangle$ 和 $\vec{y}=\langle y_1,\dots,y_m\rangle$ ， $\vec{y}$ 相对于 $\vec{x}$ 的梯度由 **雅可比矩阵 (Jacobian matrix)** 给出：

$$
J=\left(\begin{array}{ccc} \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\ \vdots & \ddots & \vdots\\ \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}} \end{array}\right)
$$

PyTorch 允许您计算给定输入向量 $v=(v_1 \dots v_m)$ 的 **雅可比积** $v^T\cdot J$ ，而不是直接计算雅可比矩阵本身。这是通过调用带有参数 $v$ 的 `backward` 来实现的。 $v$ 的大小应与我们想要对其计算乘积的原始张量的大小相同。

```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

```
First call
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])

Second call
tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.]])

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])
```

请注意，当我们第二次使用相同的参数调用 `backward` 时，梯度的值是不同的。这是因为在执行 `backward` 传播时，PyTorch 会 **累加梯度** ，即计算出的梯度值会被添加到计算图所有叶子节点的 `grad` 属性中。如果您想计算准确的梯度，则需要预先将 `grad` 属性归零。在实际训练中， *优化器 (optimizer)* 会帮助我们完成这项工作。

> [!note] 注意
> 之前我们调用了不带参数的 `backward()` 函数。这本质上等同于调用 `backward(torch.tensor(1.0))` ，这是一种在标量值函数（例如神经网络训练期间的损失）的情况下计算梯度的有效方法。