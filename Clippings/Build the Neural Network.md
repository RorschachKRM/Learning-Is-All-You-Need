---
title: "Build the Neural Network"
source: "https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html"
author:
  - "[[PyTorch Contributors]]"
published: 2023-01-01
created: 2026-07-15
description: "PyTorch Documentation. Explore PyTorch, an open-source machine learning library that accelerates the path from research prototyping to production deployment."
tags:
  - "clippings"
---
> [!note] 注意
> 下载完整示例代码。

[基础入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || [转换 (Transforms)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html) || **构建模型** || [自动求导 (Autograd)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || [模型优化](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || [保存与加载模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

## 构建神经网络

创建日期：2021年2月9日 | 最后更新：2025年1月24日 | 最后验证：未验证

神经网络由对数据执行操作的层/模块组成。 [torch.nn](https://pytorch.ac.cn/docs/stable/nn.html) 命名空间提供了构建神经网络所需的所有构建块。PyTorch 中的每个模块都是 [nn.Module](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Module.html) 的子类。神经网络本身也是一个模块，由其他模块（层）组成。这种嵌套结构使得构建和管理复杂的架构变得非常容易。

在接下来的部分中，我们将构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类。

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

## 获取训练设备

我们希望能够在 [加速器](https://pytorch.ac.cn/docs/stable/torch.html#accelerators) （如 CUDA、MPS、MTIA 或 XPU）上训练模型。如果当前有可用的加速器，我们将使用它；否则，我们将使用 CPU。

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
```

```
Using cuda device
```

## 定义类

我们通过继承 `nn.Module` 来定义神经网络，并在 `__init__` 中初始化神经网络层。每个 `nn.Module` 的子类都会在 `forward` 方法中实现对输入数据的操作。

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

我们创建 `NeuralNetwork` 的实例，将其移动到 `device` 上，并打印其结构。

```python
model = NeuralNetwork().to(device)
print(model)
```

```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

要使用模型，我们将输入数据传递给它。这会执行模型的 `forward` 方法，以及一些 [后台操作](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866) 。请勿直接调用 `model.forward()` ！

在输入上调用模型会返回一个二维张量，其中 dim=0 对应每个输出的 10 个原始预测值，dim=1 对应每个输出的各个数值。我们可以通过将结果传递给 `nn.Softmax` 模块的实例来获取预测概率。

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

```
Predicted class: tensor([1], device='cuda:0')
```

---

## 模型层

让我们拆解一下 FashionMNIST 模型中的各层。为了说明这一点，我们将取一个包含 3 张 28x28 图像的小批量样本，看看当它通过网络时会发生什么。

```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```

```
torch.Size([3, 28, 28])
```

### nn.Flatten

我们初始化 [nn.Flatten](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Flatten.html) 层，将每张 2D 的 28x28 图像转换为 784 个像素值的连续数组（保持小批量维度 dim=0 不变）。

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

```
torch.Size([3, 784])
```

### nn.Linear

[线性层 (Linear layer)](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Linear.html) 是一个使用存储的权重和偏置对输入应用线性变换的模块。

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

```
torch.Size([3, 20])
```

### nn.ReLU

非线性激活函数是创建模型输入和输出之间复杂映射的关键。它们在线性变换之后应用，引入了 *非线性* ，帮助神经网络学习各种复杂现象。

在此模型中，我们在线性层之间使用 [nn.ReLU](https://pytorch.ac.cn/docs/stable/generated/torch.nn.ReLU.html) ，但还有其他激活函数可用于在模型中引入非线性。

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

```
Before ReLU: tensor([[-0.2425, -0.3731,  0.2111, -0.1093,  0.2408,  0.0319,  0.1608, -0.1215,
          0.0888,  0.0190,  0.4148,  0.0951, -0.0991,  0.1071, -0.0810,  0.1462,
         -0.0128,  0.7347, -0.2259, -0.0735],
        [ 0.0814, -0.2649,  0.0630, -0.0509,  0.3664,  0.0797,  0.2221, -0.2124,
          0.4771,  0.0351,  0.5515,  0.3524, -0.0283,  0.1853, -0.1430,  0.0072,
         -0.1168,  0.2577, -0.1757, -0.2775],
        [-0.4642, -0.3023,  0.5033,  0.0885,  0.2315,  0.1466,  0.3365,  0.1565,
         -0.0457, -0.0341,  0.5199,  0.0510,  0.0798,  0.1739, -0.1685, -0.0074,
          0.2030,  0.1695, -0.3834, -0.1319]], grad_fn=<AddmmBackward0>)

After ReLU: tensor([[0.0000, 0.0000, 0.2111, 0.0000, 0.2408, 0.0319, 0.1608, 0.0000, 0.0888,
         0.0190, 0.4148, 0.0951, 0.0000, 0.1071, 0.0000, 0.1462, 0.0000, 0.7347,
         0.0000, 0.0000],
        [0.0814, 0.0000, 0.0630, 0.0000, 0.3664, 0.0797, 0.2221, 0.0000, 0.4771,
         0.0351, 0.5515, 0.3524, 0.0000, 0.1853, 0.0000, 0.0072, 0.0000, 0.2577,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.5033, 0.0885, 0.2315, 0.1466, 0.3365, 0.1565, 0.0000,
         0.0000, 0.5199, 0.0510, 0.0798, 0.1739, 0.0000, 0.0000, 0.2030, 0.1695,
         0.0000, 0.0000]], grad_fn=<ReluBackward0>)
```

### nn.Sequential

[nn.Sequential](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Sequential.html) 是一个有序的模块容器。数据按照定义的顺序通过所有模块。你可以使用顺序容器来快速组装像 `seq_modules` 这样的网络。

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### nn.Softmax

神经网络的最后一层线性层返回 logits （即 \[-无穷, +无穷\] 范围内的原始值），这些值被传递给 [nn.Softmax](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Softmax.html) 模块。Logits 被缩放为 \[0, 1\] 范围内的值，代表模型对每个类别的预测概率。 `dim` 参数表示数值必须相加为 1 的维度。

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## 模型参数

神经网络中的许多层都是 *参数化的* ，即拥有在训练过程中优化的权重和偏置。继承 `nn.Module` 会自动跟踪模型对象内定义的所有字段，并使得所有参数都可以通过模型的 `parameters()` 或 `named_parameters()` 方法进行访问。

在此示例中，我们遍历每个参数，并打印其大小及其值的预览。

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

```
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)

Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0274,  0.0280,  0.0355,  ..., -0.0078, -0.0040,  0.0162],
        [ 0.0129, -0.0076, -0.0281,  ...,  0.0321,  0.0219, -0.0333]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([0.0004, 0.0166], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0200, -0.0004,  0.0245,  ...,  0.0144, -0.0094,  0.0234],
        [ 0.0200,  0.0397, -0.0316,  ..., -0.0132,  0.0302,  0.0189]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0316, -0.0325], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0261,  0.0318,  0.0055,  ..., -0.0228, -0.0117,  0.0003],
        [ 0.0359, -0.0181, -0.0252,  ...,  0.0087, -0.0234,  0.0434]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([-0.0221,  0.0197], device='cuda:0', grad_fn=<SliceBackward0>)
```