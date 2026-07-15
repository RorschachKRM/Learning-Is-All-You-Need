---
tags:
  - PyTorch
related:
  - "[[Python - Numpy]]"
  - "[PyTorch官方网页教程 - 快速入门](|https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html#working-with-data)"
---
# 基础概念
**PyTorch 主要有两大特征：**
- 类似于 NumPy 的张量计算，能在 GPU 或 MPS 等硬件加速器上加速。
- 基于带自动微分系统的深度神经网络。
PyTorch 包括 torch.autograd、torch.nn、torch.optim 等子模块。
PyTorch 包含多种损失函数，包括 MSE（均方误差 = L2 范数）、交叉熵损失和负熵似然损失（对分类器有用）等。

PyTorch 主要有以下几个基础概念：张量（Tensor）、自动求导（Autograd）、神经网络模块（nn.Module）、优化器（optim）等。
- **张量（Tensor）**：PyTorch 的核心数据结构，支持多维数组，并可以在 CPU 或 GPU 上进行加速计算。
- **自动求导（Autograd）**：PyTorch 提供了自动求导功能，可以轻松计算模型的梯度，便于进行反向传播和优化。
- **神经网络（nn.Module）**：PyTorch 提供了简单且强大的 API 来构建神经网络模型，可以方便地进行前向传播和模型定义。
- **优化器（Optimizers）**：使用优化器（如 Adam、SGD 等）来更新模型的参数，使得损失最小化。
- **设备（Device）**：可以将模型和张量移动到 GPU 上以加速计算。


# 处理数据
PyTorch 提供了两个[用于处理数据的原语 (primitives)](https://pytorch.ac.cn/docs/stable/data.html)：`torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`。`Dataset` 存储样本及其对应的标签，而[DataLoader](https://docs.pytorch.ac.cn/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader")则在 `Dataset` 周围封装了一个可迭代对象。
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
```

PyTorch 提供了特定领域的库，如 [TorchText](https://pytorch.ac.cn/text/stable/index.html)、[TorchVision](https://pytorch.ac.cn/vision/stable/index.html) 和 [TorchAudio](https://pytorch.ac.cn/audio/stable/index.html)，它们都包含了数据集。在本教程中，我们将使用 TorchVision 数据集。

例`torchvision.datasets` 模块包含了许多现实世界视觉数据的 `Dataset` 对象，例如 CIFAR、COCO（[完整列表见此处](https://pytorch.ac.cn/vision/stable/datasets.html)）。在本教程中，我们使用 FashionMNIST 数据集。
每个 TorchVision `Dataset` 都包含两个参数：`transform` 和 `target_transform`，分别用于修改样本和标签。
```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
```

我们将 `Dataset` 作为参数传递给 `DataLoader`。这会封装出一个针对我们数据集的可迭代对象，并支持自动批处理、采样、重排和多进程数据加载。在这里，我们定义批次大小 (batch size) 为 64，即 dataloader 可迭代对象中的每个元素都将返回包含 64 个特征和标签的批次。[train_dataloader](https://docs.pytorch.ac.cn/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader")
```python
batch_size = 64

# Create data loaders.
train_dataloader= DataLoader(training_data, batch_size=batch_size)
test_dataloader= DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```


# 创建模型
要在 PyTorch 中定义神经网络，我们创建一个继承自 [nn.Module](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Module.html) 的类。我们在 `__init__` 函数中定义网络的层，并在 `forward` 函数中指定数据如何通过网络。为了加速神经网络中的运算，我们将其移动到 [加速器](https://pytorch.ac.cn/docs/stable/torch.html#accelerators)（如 CUDA、MPS、MTIA 或 XPU）上。如果当前有可用加速器，我们将使用它。否则，我们使用 CPU。
```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```


# 优化模型参数
为了训练模型，我们需要一个 [损失函数](https://pytorch.ac.cn/docs/stable/nn.html#loss-functions) 和一个 [优化器](https://pytorch.ac.cn/docs/stable/optim.html)。
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

在单个训练循环中，模型会对训练数据集进行预测（以批次形式输入），并反向传播预测误差以调整模型的参数。
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

我们还会针对测试数据集检查模型的性能，以确保它正在学习。
```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

训练过程在多次迭代（_周期，epochs_）中进行。在每个周期中，模型都会学习参数以做出更好的预测。我们会在每个周期打印模型的准确率和损失；我们希望看到准确率在每个周期内增加，而损失在减少。
```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```


# 保存模型
保存模型的常用方法是序列化其内部状态字典（包含模型参数）。
```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```


# 加载模型
加载模型的过程包括重新创建模型结构，并将状态字典加载到其中。
```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

现在，该模型可用于进行预测。
```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```