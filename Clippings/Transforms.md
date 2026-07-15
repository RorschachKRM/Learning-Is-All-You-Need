---
title: "Transforms"
source: "https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html"
author:
  - "[[PyTorch Contributors]]"
published: 2023-01-01
created: 2026-07-15
description: "PyTorch Documentation. Explore PyTorch, an open-source machine learning library that accelerates the path from research prototyping to production deployment."
tags:
  - "clippings"
---
> [!note] 注意
> 下载完整的示例代码。

[基础入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || **Transforms（变换）** || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || [自动求导](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || [优化](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || [保存与加载模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

## 转换

创建日期：2021年2月9日 | 最后更新：2026年5月7日 | 最后验证：未验证

数据并不总是以机器学习算法训练所需的最终处理形式出现。我们使用 **transforms（变换）** 来对数据进行一些操作，使其适合训练。

所有的 TorchVision 数据集都有两个参数—— `transform` （用于修改特征）和 `target_transform` （用于修改标签）——它们接收包含变换逻辑的可调用对象。 [torchvision.transforms](https://pytorch.ac.cn/vision/stable/transforms.html) 模块提供了多种开箱即用的常用变换。

FashionMNIST 的特征采用 PIL 图像格式，标签为整数。在训练时，我们需要将特征处理为归一化的张量，将标签处理为独热编码（one-hot encoded）张量。为了进行这些变换，我们使用 `torchvision.transforms.v2` API 以及 `torch.nn.functional.one_hot` 。

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

```
0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:10, 376kB/s]
  1%|          | 197k/26.4M [00:00<00:43, 597kB/s]
  3%|▎         | 852k/26.4M [00:00<00:12, 2.03MB/s]
 13%|█▎        | 3.41M/26.4M [00:00<00:03, 7.00MB/s]
 35%|███▍      | 9.21M/26.4M [00:00<00:01, 16.4MB/s]
 57%|█████▋    | 15.1M/26.4M [00:01<00:00, 22.3MB/s]
 80%|███████▉  | 21.1M/26.4M [00:01<00:00, 26.2MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 20.0MB/s]

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 336kB/s]

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|▏         | 65.5k/4.42M [00:00<00:11, 374kB/s]
  5%|▌         | 229k/4.42M [00:00<00:05, 703kB/s]
 20%|██        | 885k/4.42M [00:00<00:01, 2.08MB/s]
 80%|████████  | 3.54M/4.42M [00:00<00:00, 7.93MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.28MB/s]

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 45.1MB/s]
```

## ToImage() 和 ToDtype()

`torchvision.transforms.v2` API 用一个两步流水线取代了传统的 `ToTensor` 变换。 [v2.ToImage](https://pytorch.ac.cn/vision/stable/generated/torchvision.transforms.v2.ToImage.html) 将 PIL 图像或 NumPy `ndarray` 转换为 `torchvision.tv_tensors.Image` 张量，而带有 `scale=True` 参数的 [v2.ToDtype](https://pytorch.ac.cn/vision/stable/generated/torchvision.transforms.v2.ToDtype.html) 将其转换为 `float32` 类型，并将像素强度值缩放到 \[0., 1.\] 区间。

## Lambda 变换

Lambda 变换应用任何用户定义的 lambda 函数。在此，我们使用 [torch.nn.functional.one\_hot](https://pytorch.ac.cn/docs/stable/generated/torch.nn.functional.one_hot.html) 将整数标签转换为大小为 10（即我们数据集中的标签数量）的独热编码张量，然后将其转换为 `float` 类型以匹配预期的 dtype。

```python
target_transform = v2.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)
```