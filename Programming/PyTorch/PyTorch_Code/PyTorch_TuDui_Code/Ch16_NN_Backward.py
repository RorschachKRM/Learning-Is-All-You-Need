import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)

class EasyNN(nn.Module):
    def __init__(self):
        super(EasyNN, self).__init__()
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
EasyNN = EasyNN()
for data in dataloader:
    imgs, targets = data
    outputs = EasyNN(imgs)
    result_loss = loss(outputs, targets)
    # 反向传播，计算每个参数的梯度
    result_loss.backward()

    print("loss:", result_loss.item())
    print("第一层卷积的梯度：")
    print(EasyNN.model1[0].weight.grad)  # 第一层卷积权重经过反向传播得到的梯度

    print("ok")