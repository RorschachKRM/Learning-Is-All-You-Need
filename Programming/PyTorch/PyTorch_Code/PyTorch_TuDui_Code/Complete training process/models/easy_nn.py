from torch import nn


class EasyNN(nn.Module):
    """用于 CIFAR-10 分类的简单卷积神经网络。"""

    def __init__(self):
        super().__init__()
        # 保留 model1 命名，使旧版 model.py 保存的 state_dict 仍可加载。
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
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model1(x)
