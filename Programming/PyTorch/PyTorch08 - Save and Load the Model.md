---
title: 保存和加载模型
source: https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html
tags:
  - clippings
  - PyTorch
---
[学习基础知识](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || [转换 (Transforms)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html) || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || [自动求导 (Autograd)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || [优化](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || **保存与加载模型**

# 保存和加载模型

在本节中，我们将介绍如何通过保存、加载和运行模型预测来持久化模型状态。
```python
import torch
import torchvision.models as models
```


# 保存和加载模型权重

PyTorch 模型将其学习到的**参数存储在内部状态字典**中，称为 `state_dict` 。这些参数可以通过 `torch.save` 方法进行持久化。
```python
"""
	创建一个 VGG16 模型实例，VGG16 是一个经典的卷积神经网络。
	weights='IMAGENET1K_V1'表示加载在ImageNet-1K（1000 类图像分类数据集）上预训练好的权重。也就是说，这个模型不是随机初始化的，而是已经在大规模图像数据上训练过的，可以直接用于推理或微调（transfer learning）。
"""
model = models.vgg16(weights='IMAGENET1K_V1')
"""
	model.state_dict()返回模型内部状态字典（state dictionary），它是一个 Python 字典，包含了模型所有的可学习参数（权重 weight 和偏置 bias 的 tensor）。
	torch.save(...)将这些参数序列化保存到磁盘文件model_weights.pth中.
	这里只保存了参数（权重），没有保存模型的结构定义。要加载它时，得先重新创建一个 VGG16 实例，再用load_state_dict()把权重加载进去
"""
torch.save(model.state_dict(), 'model_weights.pth')
```


要加载模型权重，你需要先创建一个相同模型的实例，然后使用 `load_state_dict()` 方法加载参数。

在下面的代码中，我们设置 `weights_only=True` ，以将反序列化过程中执行的函数限制为仅加载权重所必需的函数。
使用 `weights_only=True` 被认为是加载权重时的最佳实践。
```python
model = models.vgg16() # 我们不指定权重，即创建未经训练的“空”模型
"""
1. torch.load('model_weights.pth', weights_only=True)：安全地从磁盘读取之前保存的权重 tensor（安全模式下只反序列化张量数据，不执行代码）。
2. model.load_state_dict(...)：把读取到的权重字典覆盖到刚才创建的空模型上，替换掉随机初始化的参数。
"""
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```

```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

> [!note] 注意
> 请务必在进行推理之前调用 `model.eval()` 方法，将 dropout 层和批归一化（batch normalization）层设置为评估模式。如果不这样做，将会导致推理结果不一致。


# 保存和加载包含结构的模型

在加载模型权重时，我们需要先实例化模型类，因为该类定义了网络的结构。我们可能希望将此类的结构与模型一起保存，在这种情况下，我们可以将 `model` （而不是 `model.state_dict()` ）传递给保存函数。
```python
torch.save(model, 'model.pth')
```

然后，我们可以按下述方式加载模型。
如 [保存和加载 torch.nn.Modules](https://pytorch.ac.cn/docs/stable/notes/serialization.html#saving-and-loading-torch-nn-modules) 中所述，保存 `state_dict` 被认为是最佳实践。然而，在下文中我们使用了 `weights_only=False` ，因为这涉及加载整个模型，这是 `torch.save` 的一种遗留用法。
```python
model = torch.load('model.pth', weights_only=False)
```

|           | 方法一（推荐 ✅）                             | 方法二（遗留 ⚠️）                   |
| --------- | ------------------------------------- | ---------------------------- |
| **保存**    | `torch.save(model.state_dict(), ...)` | `torch.save(model, ...)`     |
| **存的是什么** | 只有权重张量（tensor）                        | 整个模型对象（结构 + 权重 + Python 类信息） |
| **序列化方式** | 纯张量                                   | pickle（序列化整个 Python 对象）      |
| **加载**    | `weights_only=True` ✅                 | `weights_only=False` ❌       |

## 为什么保存整个模型就必须用 `False`？

`torch.save(model, ...)` 用 pickle 把整个 Python 对象序列化了。文件里不只有权重 tensor，还有**模型类的定义、方法引用、甚至代码逻辑**。

`weights_only=True` 只允许反序列化 tensor 这种安全的数据类型，它不认 pickle 打包的复杂 Python 对象，所以**直接加载会报错**。必须用 `weights_only=False` 放开限制，让 pickle 重建完整对象。

> [!note] 注意
> 此方法在序列化模型时使用 Python 的 [pickle](https://docs.pythonlang.cn/3/library/pickle.html) 模块，因此在加载模型时，它依赖于原始的类定义必须是可用的。