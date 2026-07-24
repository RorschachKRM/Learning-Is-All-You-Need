import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

"""
最大池化（下采样）
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

参数：
    kernel_size (int | tuple[int, int]) – 执行最大值运算的窗口大小
    stride (int | tuple[int, int]) – 窗口的步长。默认值为 kernel_size
    padding (int | tuple[int, int]) – 在两侧添加的隐式负无穷填充
    dilation (int | tuple[int, int]) – 控制窗口中元素步长的参数
    return_indices (bool) – 如果为 True，将返回最大值的索引以及输出。这对于后续的 torch.nn.MaxUnpool2d 非常有用
    ceil_mode (bool) – 当为 True 时，将使用 ceil 而不是 floor 来计算输出形状（floor指剩下边缘像素、不够组成完整池化窗口，就不处理）
"""

test_dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=False)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)

class EasyMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.MaxPool1 = MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        output = self.MaxPool1(x)
        return output

writer = SummaryWriter("logs_CIFAR10")

MyMaxPool = EasyMaxPool()

for step, (imgs, labels) in enumerate(test_dataloader):
    writer.add_images("imgs", imgs, step)
    # imgs.shape

    output = MyMaxPool(imgs)
    # output.shape
    writer.add_images("maxpooling", output, step)

writer.close()