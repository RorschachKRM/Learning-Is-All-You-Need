import torch
from torch import nn

# 搭建神经网络
class EasyNN(nn.Module):
    def __init__(self):
        super(EasyNN, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    EasyNN = EasyNN()
    input = torch.ones((64, 3, 32, 32))
    output = EasyNN(input)
    print(output.shape)