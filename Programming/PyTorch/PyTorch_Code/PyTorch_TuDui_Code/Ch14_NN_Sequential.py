import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

"""
Sequential序列：一个顺序容器。

模块将按照它们在构造函数中传入的顺序被添加到容器中。或者，也可以传入一个模块的 OrderedDict。
Sequential 的 forward() 方法接收任意输入并将其传递给它包含的第一个模块。然后，它将输出按顺序“链式”传给后续的每个模块，最后返回最后一个模块的输出。

class torch.nn.Sequential(*args: Module)
class torch.nn.Sequential(arg: OrderedDict[str, Module])

示例：
# 使用 Sequential 创建一个小型模型。当运行 `model` 时，输入首先传递给 `Conv2d(1,20,5)`。
# `Conv2d(1,20,5)` 的输出将用作第一个 `ReLU` 的输入；
# 第一个 `ReLU` 的输出将成为输入，传递给 `Conv2d(20,64,5)`。最后，`Conv2d(20,64,5)` 的输出将用作第二个 `ReLU` 的输入。
model = nn.Sequential(
    nn.Conv2d(1, 20, 5), 
    nn.ReLU(), 
    nn.Conv2d(20, 64, 5), 
    nn.ReLU()
)

# 将 Sequential 与 OrderedDict 结合使用。这在功能上与上面的代码相同
model = nn.Sequential(
    OrderedDict(
        [
            ("conv1", nn.Conv2d(1, 20, 5)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(20, 64, 5)),
            ("relu2", nn.ReLU()),
        ]
    )
)
"""

class EasySeqNet(nn.Module):
    def __init__(self):
        super().__init__()
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

MySeqNet = EasySeqNet()
print(MySeqNet)
input = torch.ones((64, 3, 32, 32))
output = MySeqNet(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(MySeqNet, input)
writer.close()