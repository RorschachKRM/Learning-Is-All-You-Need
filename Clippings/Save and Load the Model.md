---
title: "Save and Load the Model"
source: "https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html"
author:
  - "[[PyTorch Contributors]]"
published: 2023-01-01
created: 2026-07-17
description: "PyTorch Documentation. Explore PyTorch, an open-source machine learning library that accelerates the path from research prototyping to production deployment."
tags:
  - "clippings"
---
> [!note] 注意
> 下载完整示例代码。

[学习基础知识](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || [转换 (Transforms)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html) || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || [自动求导 (Autograd)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || [优化](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || **保存与加载模型**

## 保存和加载模型

创建日期：2021年2月9日 | 最后更新：2025年9月25日 | 最后验证：2024年11月5日

在本节中，我们将介绍如何通过保存、加载和运行模型预测来持久化模型状态。

```python
import torch
import torchvision.models as models
```

## 保存和加载模型权重

PyTorch 模型将其学习到的参数存储在内部状态字典中，称为 `state_dict` 。这些参数可以通过 `torch.save` 方法进行持久化。

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

```
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/ci-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth

  0%|          | 0.00/528M [00:00<?, ?B/s]
  7%|▋         | 34.8M/528M [00:00<00:01, 364MB/s]
 15%|█▍        | 77.2M/528M [00:00<00:01, 412MB/s]
 23%|██▎       | 120M/528M [00:00<00:01, 427MB/s]
 31%|███       | 162M/528M [00:00<00:00, 434MB/s]
 39%|███▊      | 204M/528M [00:00<00:00, 437MB/s]
 47%|████▋     | 247M/528M [00:00<00:00, 440MB/s]
 55%|█████▍    | 289M/528M [00:00<00:00, 441MB/s]
 63%|██████▎   | 332M/528M [00:00<00:00, 442MB/s]
 71%|███████   | 374M/528M [00:00<00:00, 441MB/s]
 79%|███████▉  | 416M/528M [00:01<00:00, 442MB/s]
 87%|████████▋ | 459M/528M [00:01<00:00, 443MB/s]
 95%|█████████▍| 501M/528M [00:01<00:00, 443MB/s]
100%|██████████| 528M/528M [00:01<00:00, 437MB/s]
```

要加载模型权重，你需要先创建一个相同模型的实例，然后使用 `load_state_dict()` 方法加载参数。

在下面的代码中，我们设置 `weights_only=True` ，以将反序列化过程中执行的函数限制为仅加载权重所必需的函数。使用 `weights_only=True` 被认为是加载权重时的最佳实践。

```python
model = models.vgg16() # we do not specify \`\`weights\`\`, i.e. create untrained model
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

## 保存和加载包含结构的模型

在加载模型权重时，我们需要先实例化模型类，因为该类定义了网络的结构。我们可能希望将此类的结构与模型一起保存，在这种情况下，我们可以将 `model` （而不是 `model.state_dict()` ）传递给保存函数。

```python
torch.save(model, 'model.pth')
```

然后，我们可以按下述方式加载模型。

如 [保存和加载 torch.nn.Modules](https://pytorch.ac.cn/docs/stable/notes/serialization.html#saving-and-loading-torch-nn-modules) 中所述，保存 `state_dict` 被认为是最佳实践。然而，在下文中我们使用了 `weights_only=False` ，因为这涉及加载整个模型，这是 `torch.save` 的一种遗留用法。

```python
model = torch.load('model.pth', weights_only=False)
```

> [!note] 注意
> 此方法在序列化模型时使用 Python 的 [pickle](https://docs.pythonlang.cn/3/library/pickle.html) 模块，因此在加载模型时，它依赖于原始的类定义必须是可用的。