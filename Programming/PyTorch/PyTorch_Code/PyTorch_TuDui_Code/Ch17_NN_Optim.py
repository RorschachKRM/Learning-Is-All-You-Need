"""
要使用 torch.optim，您必须构造一个优化器对象，该对象将保存当前状态并根据计算出的梯度更新参数。

要构造一个 Optimizer，您必须给它一个包含要优化的参数（都应为 Parameter）或命名参数（(str, Parameter) 元组）的可迭代对象。然后，您可以指定优化器特定的选项，例如学习率、权重衰减等。
示例：
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)


class torch.optim.Optimizer(params, defaults)
    params (iterable) – torch.Tensor 或 dict 的可迭代对象。指定应优化的 Tensor。
    defaults (dict[str, Any]) – (dict)：包含优化选项默认值的字典（当参数组未指定它们时使用）。

Optimizer.step():有优化器都实现了一个 step() 方法，用于更新参数。
示例：
    for input, target in dataset:
        optimizer.zero_grad()  # 重置所有已优化 torch.Tensor 的梯度
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


具体算法：
1. Adam优化器：
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None, decoupled_weight_decay=False)

    params (iterable) – 要优化的参数或命名参数的迭代器，或者是定义参数组的字典的迭代器。使用命名参数时，所有组中的所有参数都应该命名。

    lr (float, Tensor, optional) – 学习率 (默认: 1e-3)。

"""


import torch
import torch.nn as nn
import torchvision
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader



dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)

        result_loss = loss(outputs, targets)

        optim.zero_grad()

        result_loss.backward()
        optim.step()

        running_loss = running_loss + result_loss  # 一轮中的loss求和
    print(running_loss)